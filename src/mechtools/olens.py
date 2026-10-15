"""The oracle lens (Appendix A.9.2 of https://transformer-circuits.pub/2026/workspace/, benchmarked at https://github.com/camilablank/workspace-bench): a LoRA on the subject model that verbalizes one residual-stream activation. The lens proper (the distill and RL checkpoints, prompt kind concepts_raw) writes "- " bullet concepts, one per line; the inverter it is distilled from (continuation_raw) writes the text that would follow the activation.
The read contract travels as a whole, and reading a checkpoint with another's prompt, transform, layers or marker is a different experiment: PROMPTS[prompt_kind] is rendered as one user turn through the chat template (add_generation_prompt=True, enable_thinking=False); the marker token's input-embedding row, the output of model.get_input_embeddings() at its slot (after Gemma's sqrt(d) scale, which that module applies), is replaced by alpha * h / ||h|| (transform "unit") or global_scale * h (transform "scaled", one scale per run, fitted so the median injected norm is alpha); the generation ends at EOS and is deterministic given the batch of activations, n and seed: batches consume the RNG in order, so the same olens_read call reproduces its block, while the same activation alone or in another stack samples differently.
Inverters were trained on the AR's reconstructions of activations, not on activations, and their transform keeps the norm, so a massive-activation row (BOS, an attention sink) goes in at many times the norms they were trained on.
"Layer L" is the output of decoder block L: blocks.L.hook_resid_post in the Bridge, hidden_states[L + 1] in HF (whose last entry has the final norm applied), decoder_layers(model)[L]. Activations are captured with the lens disabled (`with peft.disable_adapter(): _, cache = bridge.run_with_cache(ids)`): the lens's own activations read as plausible concepts and are wrong.
Loading order: hf = load_hf_model(base); peft, contract = load_olens(hf, spec); bridge = boot_bridge(hf). The read functions never change which adapters are active (the caller does, with peft's API) and check_active raises when the lens is not the one active.
A checkpoint that ships an olens.json card (agu18dec/mOre-lenses: inverter, distill and RL checkpoints for Qwen3.5 0.8B to 9B, Qwen3.6-27B and Gemma-4-31B-it) is read from its card by card_contract, which raises on any field or value it does not implement, so a change of the card format fails instead of being read wrong. Cards name the marker's character and id but not its neighbors, which are then not checked. OLENS holds the two checkpoints without a card: agu18dec/olens_and_ar:olens_s3d_rl600 (the same adapter as agu18dec/oracle-lens-ddp600 and agu18dec/mOre-lenses:qwen3.6-27b/rl/s3d.ddp600.s0, sampling as WorkspaceBench's producer) and andyx10/oracle-lens-qwen3-4b (its lens_config.json, whose top_k 0 is unrestricted; its card also stops a sample at the marker id, which inject_generate does not, so a sample that emits the marker runs on to EOS or max_new_tokens)."""

import copy
import json
import os
from collections.abc import Iterable

import torch as t
from torch import Tensor
from huggingface_hub import hf_hub_download
from IPython.display import HTML, display
from peft import PeftModel

from mechtools.hooks import inject_generate
from mechtools.lens import get_toks, readout_html, tabbed, token_strip
from mechtools.models import check_active, load_adapters
from mechtools.stats import normed
from mechtools.tokens import single_token_marker, to_ids

PROMPTS = {
    "concepts_raw": "An activation vector from layer {layer} of a language model is enclosed in activation tags: <activation>{char}</activation>. Produce distinct concepts that encode this activation, each as a '- ' bullet on its own line.",
    "continuation_raw": "An activation vector from layer {layer} of a language model is enclosed in activation tags: <activation>{char}</activation>. Produce the text that immediately follows this activation.",
}
OLENS_PROMPT = PROMPTS["concepts_raw"]
INJECT = {"unit": "h -> alpha * h / |h|", "scaled": "h -> global_scale * h"}  # a card's read.inject for each transform
CARD_READ = {"transform", "alpha", "global_scale", "inject", "prompt_kind", "prompt_template", "slot_char", "slot_id", "layers", "activation", "sampling_of_record"}
CARD_ACTIVATION = "output of decoder block L at one token = hidden_states[L + 1]"

OLENS = {
    "agu18dec/olens_and_ar:olens_s3d_rl600": {"base": "Qwen/Qwen3.6-27B", "layers": list(range(20, 61, 4)), "prompt_kind": "concepts_raw", "transform": "unit", "alpha": 16000.0, "global_scale": None, "marker": ("㈜", 158983, (29, 510)), "sampling": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_new_tokens": 256}},
    "andyx10/oracle-lens-qwen3-4b": {"base": "Qwen/Qwen3-4B", "layers": list(range(12, 35, 2)), "prompt_kind": "concepts_raw", "transform": "unit", "alpha": 16000.0, "global_scale": None, "marker": ("㈎", 149705, (29, 522)), "sampling": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 128}},
}

def olens_card(spec: str, revision: str | None = None) -> dict:
    """The olens.json card of the checkpoint directory `spec`: "repo:dir" on the Hub, at `revision`, or "root:dir" in a local copy of such a repo."""
    root, _, sub = spec.partition(":")
    path = os.path.join(root, sub, "olens.json") if os.path.isdir(root) else hf_hub_download(root, "olens.json", subfolder=sub or None, revision=revision)
    with open(path) as f:
        return json.load(f)

def card_contract(card: dict) -> dict:
    """The read contract of an olens.json card: base, layers, prompt_kind, transform, alpha, global_scale, marker as (char, id, None), since cards do not name the neighbors, and sampling (the card's sampling_of_record with do_sample; its `k`, WorkspaceBench's samples per item, is dropped, olens_read's `n` takes its place). Raises for a card with no read section (an AR checkpoint, which is not a lens) and for any read field, prompt, transform or layer convention other than the ones this module implements."""
    if "read" not in card:
        raise ValueError(f"{card['run']!r} ({card['role']}) has no read contract; only inverter, distill and RL checkpoints are lenses")
    r = card["read"]
    if set(r) != CARD_READ:
        raise ValueError(f"{card['run']!r}: the card's read fields differ from the known ones by {sorted(set(r) ^ CARD_READ)}")
    sampling = dict(r["sampling_of_record"])
    checks = {
        "transform": INJECT.get(r["transform"]) == r["inject"] and (r["global_scale"] is None) == (r["transform"] == "unit"),
        "prompt": PROMPTS.get(r["prompt_kind"]) == r["prompt_template"],
        "activation": r["activation"] == CARD_ACTIVATION,
        "sampling": set(sampling) == {"temperature", "top_p", "top_k", "max_new_tokens", "k"},
    }
    if not all(checks.values()):
        raise ValueError(f"{card['run']!r}: unknown {[k for k, ok in checks.items() if not ok]} in the card's read contract {r}")
    del sampling["k"]
    return {"base": card["base_model"], "layers": r["layers"], "prompt_kind": r["prompt_kind"], "transform": r["transform"], "alpha": r["alpha"], "global_scale": r["global_scale"], "marker": (r["slot_char"], r["slot_id"], None), "sampling": {"do_sample": True, **sampling}}

def load_olens(model, spec: str = "agu18dec/olens_and_ar:olens_s3d_rl600", name: str = "olens", contract: dict | None = None, **kwargs) -> tuple[PeftModel, dict]:
    """The lens as an adapter named `name` (load_adapters(model, ..., **kwargs)) and its contract, copied, with "adapter" (the name) and "spec" added, which is what olens_read and olens_readout take. The contract is `contract` when given, with the adapter at `spec`; else OLENS[spec]; else the card of the checkpoint directory `spec` ("repo:dir", or "root:dir" in a local copy of the repo), whose adapter is the card's adapter_subfolder of the same repo, card and adapter both at kwargs' `revision` (pin a commit of a repo under active development). `model` is the raw HF base, booted into a Bridge afterwards (boot_bridge(model)), not before. Only the adapter files are fetched; andyx10's repo root also holds large data directories, which peft never touches."""
    adapter = spec
    if contract is None and spec in OLENS:
        contract = OLENS[spec]
    elif contract is None:
        card = olens_card(spec, kwargs.get("revision"))
        contract, adapter = card_contract(card), f"{spec.partition(':')[0]}:{card['adapter_subfolder']}"
    peft = load_adapters(model, {name: adapter}, **kwargs)
    return peft, copy.deepcopy(contract) | {"adapter": name, "spec": spec}

def olens_prompt(tokenizer, layer: int, contract: dict, chars: Iterable[str] | None = None) -> tuple[list[int], int]:
    """(ids, slot): PROMPTS[contract["prompt_kind"]] for `layer`, rendered as one user turn with add_generation_prompt=True and enable_thinking=False around the marker character single_token_marker finds (scanning `chars`, by default the trained range U+3200..U+33FF), and the index of the marker token. Raises for a layer outside contract["layers"] (the lens never saw the others) and when the scan's character or id differ from contract["marker"], or its neighbor ids where the contract names them, which means another tokenizer or chat template than the checkpoint's."""
    if layer not in contract["layers"]:
        raise ValueError(f"layer {layer} is off-contract: the lens reads layers {contract['layers']}")
    prompt = PROMPTS[contract["prompt_kind"]]
    render = lambda char: to_ids([{"role": "user", "content": prompt.format(layer=layer, char=char)}], tokenizer, add_generation_prompt=True, enable_thinking=False)
    char, cid, ids, slot = single_token_marker(tokenizer, render, chars)
    mchar, mid, near = contract["marker"]
    if (char, cid) != (mchar, mid) or near is not None and (ids[slot - 1], ids[slot + 1]) != tuple(near):
        raise ValueError(f"the marker scan found {char!r} (id {cid}) between ids {(ids[slot - 1], ids[slot + 1])}, but the contract's marker is {mchar!r} (id {mid}) between {near or 'any ids'}: this tokenizer or chat template is not the checkpoint's")
    return ids, slot

def parse_bullets(text: str) -> list[str]:
    """The "- " bullet lines of a lens sample as concepts, without the marker and surrounding whitespace; other lines are dropped."""
    return [line[2:].strip() for line in map(str.strip, text.splitlines()) if line.startswith("- ")]

def olens_read(model, tokenizer, h: Tensor, layer: int, contract: dict, n: int = 1, seed: int | None = 0, chars: Iterable[str] | None = None, **sampling) -> list[dict]:
    """Verbalize activations from `layer` through the lens: `h` is one activation [d] or a stack [m, d], read `n` times each, and the result is one record per sample, activation-major: `layer`, `i` (the row of h, 0 for a [d] vector), `j` (the sample), `h` (the activation, float32 on the cpu), `text` (the raw sample), `ids` (its sampled ids up to eos), `stopped` (it ended at eos rather than at max_new_tokens) and `bullets` (parse_bullets of the text, None for an inverter, whose text is a continuation). `model` is the PeftModel with the lens active (check_active raises otherwise) and `tokenizer` the base model's.
    One inject_generate call replaces the marker's input-embedding row with the contract's transform of h (alpha * h / ||h|| or global_scale * h) in a batch of m * n rows ordered activation-major, sample-minor (row i * n + j is activation i's sample j), seeded by `seed` once for the batch, so the same call reproduces its samples while an activation read alone or in another stack samples differently. Sampling is contract["sampling"] updated by the kwargs (do_sample, temperature, top_p, top_k, max_new_tokens, ...). `chars` goes to olens_prompt's marker scan."""
    check_active(model, [contract["adapter"]])
    h = t.as_tensor(h)
    if h.ndim not in (1, 2):
        raise ValueError(f"h must be [d] or [m, d], got {tuple(h.shape)}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    ids, slot = olens_prompt(tokenizer, layer, contract, chars)
    write = {"unit": lambda orig, v: contract["alpha"] * normed(v), "scaled": lambda orig, v: contract["global_scale"] * v}[contract["transform"]]
    kw = {**contract["sampling"], **sampling}
    max_new_tokens = kw.pop("max_new_tokens")
    hs = t.atleast_2d(h)
    rows = inject_generate(model, tokenizer, ids, model.get_input_embeddings(), [slot], hs.repeat_interleave(n, 0), write, max_new_tokens=max_new_tokens, seed=seed, **kw)
    hc = hs.float().cpu()
    bullets = contract["prompt_kind"] == "concepts_raw"
    return [{"layer": layer, "i": k // n, "j": k % n, "h": hc[k // n], **r, "bullets": parse_bullets(r["text"]) if bullets else None} for k, r in enumerate(rows)]

def olens_readout(cache, layers, pos: int | list[int], model, tokenizer, contract: dict, n: int = 1, seed: int | None = 0, raw: bool = False, hook: str = "hook_resid_post", input_src=None, title: str = "oracle lens readout", ctx: int = 32, chars: Iterable[str] | None = None, **sampling) -> list[dict]:
    """Tabbed oracle-lens readout of a run_with_cache cache of one prompt: a tab per layer and, when `pos` is a list, a second bar of tabs per position; a pane holds the n samples as small tables, one bullet per row (one line per row with raw=True, and always for an inverter). input_src (see get_toks) puts a token strip above the bars marking the positions read out, and clicking a marked token switches to it.
    `hook` is hook_resid_post because the contract's "layer L" is the output of block L, where the lens readouts of mechtools.lens default to hook_resid_pre. The cache must come from a forward with the lens disabled (`with model.disable_adapter(): _, cache = bridge.run_with_cache(ids)`), since the lens's own activations give plausible, wrong readouts; `model` is the PeftModel with the lens active, read through one batched olens_read per layer over all positions, with `sampling` kwargs (do_sample=False for greedy, temperature, ...) overriding the contract's. Returns olens_read's records with `pos` (as given) added, layer-major, then position-major."""
    positions = [pos] if isinstance(pos, int) else list(pos)
    if len(set(positions)) != len(positions):
        raise ValueError(f"pos has repeated positions: {positions}")
    toks, ids = get_toks(input_src, tokenizer)
    lines = lambda r: r["text"].splitlines() if raw or r["bullets"] is None else r["bullets"]
    recs, panes = [], {}
    for layer in layers:
        rs = [{"layer": layer, "pos": positions[r["i"]], **r} for r in olens_read(model, tokenizer, cache[f"blocks.{layer}.{hook}"][0, positions], layer, contract, n, seed, chars, **sampling)]
        recs += rs
        panes[f"L{layer}"] = {f"p{p}": [(f"sample {r['j']}", [[line] for line in lines(r)], None, ["w"]) for r in rs if r["pos"] == p] for p in positions}  # one wrapping column ('w'): the bullets wrap inside the frame instead of pushing the sample tables past it
    display(HTML(readout_html(tabbed(panes, token_strip(toks, ids, pos, ctx) if toks is not None else ""), title, n_cols=min(n, 4))))
    return recs
