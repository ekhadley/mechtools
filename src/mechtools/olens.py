"""The oracle lens (Appendix A.9.2 of https://transformer-circuits.pub/2026/workspace/, benchmarked at https://github.com/camilablank/workspace-bench): a LoRA on the subject model that verbalizes one residual-stream activation as "- " bullet concepts.
The read contract travels as a whole, and reading a checkpoint with another's alpha, prompt, layers or marker is a different experiment: OLENS_PROMPT is rendered as one user turn through the chat template (add_generation_prompt=True, enable_thinking=False); the marker token's input-embedding row, the output of model.get_input_embeddings() at its slot, is replaced by alpha * h / ||h||; the generation is "- " bullets, one concept per line, ended by EOS, deterministic given the batch of activations, n and seed: batches consume the RNG in order, so the same olens_read call reproduces its block, while the same activation alone or in another stack samples differently.
"Layer L" is the output of decoder block L: blocks.L.hook_resid_post in the Bridge, hidden_states[L + 1] in HF (whose last entry has the final norm applied), decoder_layers(model)[L]. Activations are captured with the lens disabled (`with peft.disable_adapter(): _, cache = bridge.run_with_cache(ids)`): the lens's own activations read as plausible concepts and are wrong.
Loading order: hf = load_hf_model(base); peft, contract = load_olens(hf); bridge = boot_bridge(hf). The read functions never change which adapters are active (the caller does, with peft's API) and check_active raises when the lens is not the one active.
OLENS holds each checkpoint's contract, from its card: agu18dec/olens_and_ar (the same adapter as agu18dec/oracle-lens-ddp600, sampling as WorkspaceBench's producer) and andyx10/oracle-lens-qwen3-4b (its lens_config.json, whose top_k 0 is unrestricted; its card also stops a sample at the marker id, which inject_generate does not, so a sample that emits the marker runs on to EOS or max_new_tokens)."""

import copy
import html
from collections.abc import Iterable

import torch as t
from torch import Tensor
from IPython.display import HTML, display
from peft import PeftModel

from mechtools.hooks import inject_generate
from mechtools.lens import get_toks, readout_grid, readout_html, tabbed, token_strip
from mechtools.models import check_active, load_adapters
from mechtools.stats import normed
from mechtools.tokens import single_token_marker, to_ids

OLENS_PROMPT = "An activation vector from layer {layer} of a language model is enclosed in activation tags: <activation>{char}</activation>. Produce distinct concepts that encode this activation, each as a '- ' bullet on its own line."

OLENS = {
    "agu18dec/olens_and_ar:olens_s3d_rl600": {"base": "Qwen/Qwen3.6-27B", "layers": list(range(20, 61, 4)), "alpha": 16000.0, "marker": ("㈜", 158983, (29, 510)), "sampling": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_new_tokens": 256}},
    "andyx10/oracle-lens-qwen3-4b": {"base": "Qwen/Qwen3-4B", "layers": list(range(12, 35, 2)), "alpha": 16000.0, "marker": ("㈎", 149705, (29, 522)), "sampling": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 128}},
}

def load_olens(model, spec: str = "agu18dec/olens_and_ar:olens_s3d_rl600", name: str = "olens", contract: dict | None = None, **kwargs) -> tuple[PeftModel, dict]:
    """load_adapters(model, {name: spec}, **kwargs) and the checkpoint's contract: `contract` when given (a checkpoint not in the table), else OLENS[spec], copied, with "adapter" (the name) and "spec" added, which is what olens_read and olens_readout take. `model` is the raw HF base, booted into a Bridge afterwards (boot_bridge(model)), not before. Only the adapter files are fetched; andyx10's repo root also holds large data directories, which peft never touches."""
    if contract is None and spec not in OLENS:
        raise KeyError(f"no contract for {spec!r}: the known checkpoints are {list(OLENS)}; pass contract= for another")
    peft = load_adapters(model, {name: spec}, **kwargs)
    return peft, copy.deepcopy(OLENS[spec] if contract is None else contract) | {"adapter": name, "spec": spec}

def olens_prompt(tokenizer, layer: int, contract: dict, chars: Iterable[str] | None = None) -> tuple[list[int], int]:
    """(ids, slot): OLENS_PROMPT for `layer`, rendered as one user turn with add_generation_prompt=True and enable_thinking=False around the marker character single_token_marker finds (scanning `chars`, by default the trained range U+3200..U+33FF), and the index of the marker token. Raises for a layer outside contract["layers"] (the student never saw the others) and when the scan's character, id or neighbor ids differ from contract["marker"], which means another tokenizer or chat template than the checkpoint's."""
    if layer not in contract["layers"]:
        raise ValueError(f"layer {layer} is off-contract: the lens reads layers {contract['layers']}")
    render = lambda char: to_ids([{"role": "user", "content": OLENS_PROMPT.format(layer=layer, char=char)}], tokenizer, add_generation_prompt=True, enable_thinking=False)
    char, cid, ids, slot = single_token_marker(tokenizer, render, chars)
    mchar, mid, (left, right) = contract["marker"]
    if (char, cid, ids[slot - 1], ids[slot + 1]) != (mchar, mid, left, right):
        raise ValueError(f"the marker scan found {char!r} (id {cid}) between ids {(ids[slot - 1], ids[slot + 1])}, but the contract's marker is {mchar!r} (id {mid}) between {(left, right)}: this tokenizer or chat template is not the checkpoint's")
    return ids, slot

def parse_bullets(text: str) -> list[str]:
    """The "- " bullet lines of a lens sample as concepts, without the marker and surrounding whitespace; other lines are dropped."""
    return [line[2:].strip() for line in map(str.strip, text.splitlines()) if line.startswith("- ")]

def olens_read(model, tokenizer, h: Tensor, layer: int, contract: dict, n: int = 1, seed: int | None = 0, raw: bool = False, chars: Iterable[str] | None = None, **sampling) -> list:
    """Verbalize activations from `layer` through the lens. `h` is one activation [d], giving a list of n items, or a stack [m, d], giving m lists of n items; an item is parse_bullets of a sample, or the sample's text with raw=True. `model` is the PeftModel with the lens active (check_active raises otherwise) and `tokenizer` the base model's.
    One inject_generate call replaces the marker's input-embedding row with alpha * h / ||h|| in a batch of m * n rows ordered activation-major, sample-minor (row i * n + j is activation i's sample j), seeded by `seed` once for the batch, so the same call reproduces its samples while an activation read alone or in another stack samples differently. Sampling is contract["sampling"] updated by the kwargs (do_sample, temperature, top_p, top_k, max_new_tokens, ...). `chars` goes to olens_prompt's marker scan."""
    check_active(model, [contract["adapter"]])
    h = t.as_tensor(h)
    if h.ndim not in (1, 2):
        raise ValueError(f"h must be [d] or [m, d], got {tuple(h.shape)}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    ids, slot = olens_prompt(tokenizer, layer, contract, chars)
    kw = {**contract["sampling"], **sampling}
    max_new_tokens = kw.pop("max_new_tokens")
    texts = inject_generate(model, tokenizer, ids, model.get_input_embeddings(), [slot], t.atleast_2d(h).repeat_interleave(n, 0), lambda orig, v: contract["alpha"] * normed(v), max_new_tokens=max_new_tokens, seed=seed, **kw)
    items = texts if raw else [parse_bullets(text) for text in texts]
    rows = [items[i:i + n] for i in range(0, len(items), n)]
    return rows[0] if h.ndim == 1 else rows

def olens_readout(cache, layers, pos: int | list[int], model, tokenizer, contract: dict, n: int = 1, seed: int | None = 0, raw: bool = False, hook: str = "hook_resid_post", input_src=None, title: str = "oracle lens readout", ctx: int = 32, chars: Iterable[str] | None = None, **sampling) -> dict:
    """Tabbed oracle-lens readout of a run_with_cache cache of one prompt: a tab per layer and, when `pos` is a list, a second bar of tabs per position; a pane holds the n samples as small tables, one bullet per row (one line per row with raw=True). input_src (see get_toks) puts a token strip above the bars marking the positions read out, and clicking a marked token switches to it.
    `hook` is hook_resid_post because the contract's "layer L" is the output of block L, where the lens readouts of mechtools.lens default to hook_resid_pre. The cache must come from a forward with the lens disabled (`with model.disable_adapter(): _, cache = bridge.run_with_cache(ids)`), since the lens's own activations give plausible, wrong readouts; `model` is the PeftModel with the lens active, read through one batched olens_read per layer over all positions, with `sampling` kwargs (do_sample=False for greedy, temperature, ...) overriding the contract's. Returns {layer: {pos: [items]}} for a list of positions and {layer: [items]} for an int, like cluster_readout, with items as olens_read gives them."""
    positions = [pos] if isinstance(pos, int) else list(pos)
    if len(set(positions)) != len(positions):
        raise ValueError(f"pos has repeated positions: {positions}")
    toks, ids = get_toks(input_src, tokenizer)
    out, panes = {}, {}
    for layer in layers:
        samples = olens_read(model, tokenizer, cache[f"blocks.{layer}.{hook}"][0, positions], layer, contract, n, seed, raw, chars, **sampling)
        out[layer] = dict(zip(positions, samples))
        panes[f"L{layer}"] = {f"p{p}": readout_grid([(f"sample {j}", [([f"<div style='text-align:left;white-space:normal'>{html.escape(line)}</div>"], None) for line in (s.splitlines() if raw else s)]) for j, s in enumerate(ss)]) for p, ss in out[layer].items()}  # a bare last cell is right-aligned and kept on one line by the frame's (name, value) css, which would push the sample tables past the frame
    display(HTML(readout_html(tabbed(panes, token_strip(toks, ids, pos, ctx) if toks is not None else ""), title, n_cols=min(n, 4))))
    return out if isinstance(pos, list) else {layer: v[pos] for layer, v in out.items()}
