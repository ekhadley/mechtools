import torch as t
from torch import Tensor
from tqdm import tqdm
from transformers.cache_utils import DynamicLayer

def stream_toks(model, toks: Tensor, new_toks: int = 512):
    """Yield temperature-1 sampled token ids from a TransformerBridge, one at a time, until eos."""
    past = None
    for _ in range(new_toks):
        logits, past = model(toks, return_type="logits_and_cache", past_key_values=past, use_cache=True)
        toks = t.multinomial(t.softmax(logits[0, -1].float(), dim=-1), num_samples=1).unsqueeze(0)
        if toks.item() == model.tokenizer.eos_token_id:
            break
        yield toks.item()

def stream_toks_hf(model, tokenizer, toks: Tensor, new_toks: int = 512):
    """stream_toks for a raw HF causal LM."""
    past = None
    for _ in range(new_toks):
        out = model(toks, past_key_values=past, use_cache=True)
        past = out.past_key_values
        toks = t.multinomial(t.softmax(out.logits[0, -1].float(), dim=-1), num_samples=1).unsqueeze(0)
        if toks.item() == tokenizer.eos_token_id:
            break
        yield toks.item()

def sample_batch(model, prompt_toks: Tensor, n: int, new_toks: int = 512) -> list[list[int]]:
    """n independent temperature-1 samples from one prompt [1, seq], generated as a batch; each row is returned cut before its first eos."""
    eos = model.tokenizer.eos_token_id
    toks = prompt_toks.repeat(n, 1)
    past = None
    gen = t.zeros(n, 0, dtype=t.long, device=prompt_toks.device)
    alive = t.ones(n, dtype=t.bool, device=prompt_toks.device)
    lengths = t.full((n,), new_toks, device=prompt_toks.device)
    for step in tqdm(range(new_toks), desc="sampling", ascii=" >="):
        logits, past = model(toks, return_type="logits_and_cache", past_key_values=past, use_cache=True)
        toks = t.multinomial(t.softmax(logits[:, -1].float(), dim=-1), num_samples=1)
        gen = t.cat([gen, toks], dim=1)
        ended = alive & (toks.squeeze(1) == eos)
        lengths[ended] = step
        alive &= ~ended
        if not alive.any(): break
    return [gen[i, :lengths[i]].tolist() for i in range(n)]

def sample_rolling(model, prompt_toks: Tensor, n: int, batch_size: int, new_toks: int) -> list[list[int]]:
    """n independent temperature-1 samples from one prompt [1, seq] as a rolling batch: a row that ends (eos or new_toks) is restarted in place from the prompt while samples remain to start, else dropped from the batch. A restarted row is left-padded to the batch's cache length, with the mask and position ids covering only its real tokens. Each sample is cut before its first eos."""
    eos, last, plen = model.tokenizer.eos_token_id, prompt_toks[0, -1], prompt_toks.shape[1] - 1
    B = min(batch_size, n)
    _, cache = model(prompt_toks[:, :-1].repeat(B, 1), return_type="logits_and_cache", use_cache=True)
    template = [(l.keys[0].clone(), l.values[0].clone()) if isinstance(l, DynamicLayer) else (l.conv_states[0][0].clone(), l.recurrent_states[0][0].clone()) for l in cache.layers]
    toks = last.repeat(B, 1)  # next token fed to each row
    n_real = t.full((B,), plen, device=prompt_toks.device)  # unpadded cache entries per row, also the next token's position
    gen = [[] for _ in range(B)]
    out, n_started = [], B
    bar = tqdm(total=n, desc="sampling", ascii=" >=")
    while gen:
        S = cache.get_seq_length()
        mask = (t.arange(S + 1, device=toks.device) >= (S - n_real)[:, None]).long()
        logits, cache = model(toks, return_type="logits_and_cache", past_key_values=cache, use_cache=True, attention_mask=mask, position_ids=n_real[:, None])
        toks = t.multinomial(t.softmax(logits[:, -1].float(), dim=-1), num_samples=1)
        n_real += 1
        for row, tok in zip(gen, toks.squeeze(1).tolist()):
            row.append(tok)
        keep = []
        for i, row in enumerate(gen):
            if row[-1] != eos and len(row) < new_toks:
                keep.append(i)
                continue
            out.append(row[:-1] if row[-1] == eos else row)
            bar.update()
            if n_started == n: continue
            n_started += 1
            keep.append(i)
            for layer, (a, b) in zip(cache.layers, template):  # entries left of the prompt are stale but masked
                if isinstance(layer, DynamicLayer): layer.keys[i, :, S + 1 - plen:], layer.values[i, :, S + 1 - plen:] = a, b
                else: layer.conv_states[0][i], layer.recurrent_states[0][i] = a, b
            toks[i], n_real[i], gen[i] = last, plen, []
        if len(keep) < len(gen):
            cache.reorder_cache(t.tensor(keep, device=toks.device))
            toks, n_real, gen = toks[keep], n_real[keep], [gen[i] for i in keep]
        t.cuda.empty_cache()
    return out
