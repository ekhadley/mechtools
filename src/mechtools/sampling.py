import torch as t
from torch import Tensor
from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer, LinearAttentionLayer

from mechtools.bars import pbar

def eos_ids(model, tokenizer=None) -> set[int]:
    """The token ids a sample stops at: the model's generation_config.eos_token_id (an int or a list; a chat model ends its turn with a token that is not the tokenizer's eos, e.g. gemma-3-it's <end_of_turn> beside <eos>) plus the tokenizer's eos_token_id. model is a TransformerBridge, or a raw HF model with its tokenizer passed. Raises when neither is set, since a sampler with no terminator runs every row to new_toks."""
    tokenizer = model.tokenizer if tokenizer is None else tokenizer
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)  # a Bridge delegates the attribute to the HF model it wraps
    ids = {*(eos if isinstance(eos, (list, tuple)) else [eos]), tokenizer.eos_token_id} - {None}
    if not ids:
        raise ValueError("no eos token id: neither generation_config.eos_token_id nor tokenizer.eos_token_id is set, so sampling would never stop before new_toks")
    return ids

def stream_toks(model, toks: Tensor, new_toks: int = 512):
    """Yield temperature-1 sampled token ids from a TransformerBridge, one at a time, until eos (any id in eos_ids)."""
    eos, past = eos_ids(model), None
    for _ in range(new_toks):
        logits, past = model(toks, return_type="logits_and_cache", past_key_values=past, use_cache=True)
        toks = t.multinomial(t.softmax(logits[0, -1].float(), dim=-1), num_samples=1).unsqueeze(0)
        tok = toks.item()
        if tok in eos:
            break
        yield tok

def stream_toks_hf(model, tokenizer, toks: Tensor, new_toks: int = 512):
    """stream_toks for a raw HF causal LM."""
    eos, past = eos_ids(model, tokenizer), None
    for _ in range(new_toks):
        out = model(toks, past_key_values=past, use_cache=True)
        past = out.past_key_values
        toks = t.multinomial(t.softmax(out.logits[0, -1].float(), dim=-1), num_samples=1).unsqueeze(0)
        tok = toks.item()
        if tok in eos:
            break
        yield tok

def sample_batch(model, prompt_toks: Tensor, n: int, new_toks: int = 512, quiet: bool = False) -> list[list[int]]:
    """n independent temperature-1 samples from one prompt [1, seq], generated as a batch; each row is returned cut before its first eos (any id in eos_ids); a row of length new_toks hit the cap without stopping. quiet hides the progress bar."""
    eos = t.tensor(sorted(eos_ids(model)), device=prompt_toks.device)
    toks = prompt_toks.repeat(n, 1)
    past = None
    gen = t.zeros(n, new_toks, dtype=t.long, device=prompt_toks.device)
    alive = t.ones(n, dtype=t.bool, device=prompt_toks.device)
    lengths = t.full((n,), new_toks, device=prompt_toks.device)
    for step in pbar(range(new_toks), desc="sampling", disable=quiet):
        logits, past = model(toks, return_type="logits_and_cache", past_key_values=past, use_cache=True)
        toks = t.multinomial(t.softmax(logits[:, -1].float(), dim=-1), num_samples=1)
        gen[:, step] = toks.squeeze(1)
        ended = alive & t.isin(toks.squeeze(1), eos)
        lengths[ended] = step
        alive &= ~ended
        if not alive.any(): break
    return [row[:length] for row, length in zip(gen.tolist(), lengths.tolist())]

def stream_rolling(model, prompt_toks: Tensor, n: int, batch_size: int, new_toks: int, quiet: bool = False):
    """Yield n independent temperature-1 samples from one prompt [1, seq] as each finishes, generated as a rolling batch: a row that ends (eos or new_toks) is restarted in place from the prompt while samples remain to start, else dropped from the batch. The cache is as wide as the longest live row, whatever n is: a shorter row is left-padded to it, with the mask and position ids covering only its real tokens. A restart rewrites keys and values, or the conv and recurrent state, so the cache layers must be exactly DynamicLayer, DynamicSlidingWindowLayer or LinearAttentionLayer; any other type (the indexer keys of sparse attention, hybrid layers holding both kinds of state, DeepSeek-V4's compressed-attention caches) raises TypeError right after the prefill. Each sample is cut before its first eos (any id in eos_ids). quiet hides the progress bar.
    Save each sample as it arrives and a crash or interrupt keeps everything that finished; another call with the remaining n tops up. Samples come in completion order, so a run stopped early lacks up to batch_size samples that skew long, the rows in flight. The body runs when consumed, not when called: hooks, seed and grad contexts must enclose the consuming loop, and a generator kept but not exhausted keeps the batch's cache on the device."""
    eos, last, plen = eos_ids(model), prompt_toks[0, -1], prompt_toks.shape[1] - 1
    B = min(batch_size, n)
    _, cache = model(prompt_toks[:, :-1].repeat(B, 1), return_type="logits_and_cache", use_cache=True)
    if other := [type(l).__name__ for l in cache.layers if type(l) not in (DynamicLayer, DynamicSlidingWindowLayer, LinearAttentionLayer)]:
        raise TypeError(f"stream_rolling cannot restart a row of a {other[0]} cache: it rewrites keys and values, or the conv and recurrent state, and that layer holds other per-row state")
    template = [(l.keys[0].clone(), l.values[0].clone()) if isinstance(l, DynamicLayer) else (l.conv_states[0][0].clone(), l.recurrent_states[0][0].clone()) for l in cache.layers]
    toks = last.repeat(B, 1)  # next token fed to each row
    n_real = t.full((B,), plen, device=prompt_toks.device)  # unpadded cache entries per row, also the next token's position
    gen = [[] for _ in range(B)]
    n_started = B
    with pbar(total=n, desc="sampling", disable=quiet) as bar:
        while gen:
            S = int(n_real.max())  # cache entries of the longest live row: every row masks the entries left of them, so they are dropped
            for layer in cache.layers:
                if isinstance(layer, DynamicLayer): layer.keys, layer.values = layer.keys[:, :, -S:], layer.values[:, :, -S:]
                if isinstance(layer, DynamicSlidingWindowLayer): layer.cumulative_length = S  # the length a sliding-window layer reports, its keys holding at most the window
            mask = (t.arange(S + 1, device=toks.device) >= (S - n_real)[:, None]).long()
            logits, cache = model(toks, return_type="logits_and_cache", past_key_values=cache, use_cache=True, attention_mask=mask, position_ids=n_real[:, None])
            toks = t.multinomial(t.softmax(logits[:, -1].float(), dim=-1), num_samples=1)
            n_real += 1
            for row, tok in zip(gen, toks.squeeze(1).tolist()):
                row.append(tok)
            keep = []
            for i, row in enumerate(gen):
                if row[-1] not in eos and len(row) < new_toks:
                    keep.append(i)
                    continue
                yield row[:-1] if row[-1] in eos else row
                bar.update()
                if n_started == n: continue
                n_started += 1
                keep.append(i)
                for layer, (a, b) in zip(cache.layers, template):  # entries left of the prompt are stale but masked
                    if isinstance(layer, DynamicLayer): layer.keys[i, :, -plen:], layer.values[i, :, -plen:] = a, b
                    else: layer.conv_states[0][i], layer.recurrent_states[0][i] = a, b
                toks[i], n_real[i], gen[i] = last, plen, []
            if len(keep) < len(gen):
                cache.reorder_cache(t.tensor(keep, dtype=t.long, device=toks.device))  # dtype: an empty keep, when the batch's last rows end together, would otherwise be float
                toks, n_real, gen = toks[keep], n_real[keep], [gen[i] for i in keep]
            t.cuda.empty_cache()

def sample_rolling(model, prompt_toks: Tensor, n: int, batch_size: int, new_toks: int, quiet: bool = False) -> list[list[int]]:
    """list(stream_rolling(...)): all n samples once the last has finished."""
    return list(stream_rolling(model, prompt_toks, n, batch_size, new_toks, quiet))

def stream_rollouts(model, toks: Tensor, cuts: list[int], batch_size: int, max_len: int, quiet: bool = False):
    """Yield (i, sample) for every cut in cuts as it finishes: sample i continues the first cuts[i] tokens of toks [1, seq] at temperature 1 until an eos (any id in eos_ids, not included) or until prefix and sample together hold max_len tokens, so a sample of max_len - cuts[i] tokens hit the cap. The cap is on the total length, the same wherever a row was cut, so a rollout's distribution does not depend on its cut, which is what lets one that reproduces the text's next tokens count as a sample from there (resample.estimate).
    One forward over toks caches every position once; a row starts from a right-aligned copy of the first cuts[i] - 1 entries of that cache and feeds the cut's last token, so no prefix is recomputed. Rows run batch_size at a time from the longest cut down, so a batch's rows are about as long and little of its cache is padding, and a row that finishes leaves its batch: a replacement would be padded to the batch's longest straggler, and a step costs in proportion to rows times cache length. Full-attention caches only: a sliding-window or linear-attention layer cannot be cut back to a prefix, so a model with one raises. Samples come in completion order, within a batch the shortest first, so a run that dies leaves every position's finished rollouts biased toward short continuations: save a position's rollouts together once all of them are in, and a rerun with the missing cuts tops up. The body runs when consumed, like stream_rolling."""
    eos = t.tensor(sorted(eos_ids(model)), device=toks.device)
    if any(not 1 <= c <= toks.shape[1] or c >= max_len for c in cuts):
        raise ValueError(f"cuts must lie in 1..{toks.shape[1]}, the length of toks, and below max_len; got cuts in {min(cuts)}..{max(cuts)} with max_len {max_len}")
    _, full = model(toks[:, :-1], return_type="logits_and_cache", use_cache=True)
    if other := [type(l).__name__ for l in full.layers if type(l) is not DynamicLayer]:
        raise TypeError(f"a {other[0]} cache cannot be cut back to a prefix: stream_rollouts needs full attention in every layer")
    kv = [(l.keys[0], l.values[0]) for l in full.layers]  # per layer [heads, seq - 1, d]: the entries every row is cut from
    order = sorted(range(len(cuts)), key=lambda i: -cuts[i])
    with pbar(total=len(cuts), desc="rollouts", disable=quiet) as bar:
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            c = t.tensor([cuts[i] - 1 for i in idx], device=toks.device)  # per row: the cache entries it starts with, and the position of the token it feeds
            S, fed, gen = int(c.max()), toks[0, c][:, None], [[] for _ in idx]
            src = (t.arange(S, device=toks.device)[None] - (S - c)[:, None]).clamp(min=0)  # [rows, S]: each row's entries right-aligned; the columns left of them are padding the mask hides
            cache = DynamicCache((k[:, src].transpose(0, 1), v[:, src].transpose(0, 1)) for k, v in kv)  # a generator, so one layer's gathered copy is alive at a time
            while idx:
                mask = (t.arange(S + 1, device=toks.device)[None] >= (S - c)[:, None]).long()
                logits, cache = model(fed, return_type="logits_and_cache", past_key_values=cache, use_cache=True, attention_mask=mask, position_ids=c[:, None])
                fed, c = t.multinomial(t.softmax(logits[:, -1].float(), dim=-1), num_samples=1), c + 1
                ended = t.isin(fed[:, 0], eos)
                for row, tok, e in zip(gen, fed[:, 0].tolist(), ended.tolist()):
                    if not e: row.append(tok)
                done = ended | (c >= max_len - 1)
                S += 1
                if not done.any(): continue
                for j in done.nonzero()[:, 0].tolist():
                    yield idx[j], gen[j]
                bar.update(int(done.sum()))
                keep = (~done).nonzero()[:, 0]
                cache.reorder_cache(keep)
                fed, c, idx, gen = fed[keep], c[keep], [idx[j] for j in keep.tolist()], [gen[j] for j in keep.tolist()]
                if idx and (pad := S - int(c.max())):  # the longest rows left: drop the columns that are padding for every row still running
                    for layer in cache.layers:
                        layer.keys, layer.values = layer.keys[:, :, pad:], layer.values[:, :, pad:]
                    S -= pad
