import html
import os

import torch as t
import yaml
from torch import Tensor
from huggingface_hub import hf_hub_download
from IPython.display import HTML, display
from peft import PeftModel

from mechtools.hooks import decoder_layers, inject_generate
from mechtools.lens import get_toks, readout_grid, readout_html, tabbed, token_strip
from mechtools.models import check_active, load_adapters
from mechtools.tokens import to_ids

NLA_REPO = "ceselder/qwen3.6-27b-nla-rl"
NLA_WARM = "av_warmstart_lora/iter_0003864"

def load_nla_meta(repo: str = NLA_REPO) -> dict:
    """The checkpoint's nla_meta.yaml as a dict, from `repo`, a local directory or a Hub repo id: `extraction` (layer_index, base_model, d_model, norm), `tokens` (injection_char, injection_token_id, injection_left_neighbor_id, injection_right_neighbor_id) and `prompt_templates` (actor, the verbalizer prompt with an {injection_char} placeholder, and critic, the reconstructor's)."""
    path = os.path.join(repo, "nla_meta.yaml") if os.path.isdir(repo) else hf_hub_download(repo, "nla_meta.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

def load_nla(model, repo: str = NLA_REPO, step: int = 400, name: str = "nla", meta: dict | None = None, **kwargs) -> tuple[PeftModel, dict]:
    """The natural language autoencoder's verbalizer as two toggleable LoRA adapters on `model`, the plain base model (a raw HF model, or the PeftModel of an earlier load_adapters call): `{name}_warm`, the warm-start LoRA at NLA_WARM, and `{name}_rl`, the RL LoRA at av_rl_adapters/iter_{step:06d} (steps 100 to 800; the repo's fve_trajectory.json gives each step's held-out FVE, and 400 is the step WorkspaceBench used). Two adapters because the repo's av_base is the plain base with the warm-start LoRA merged in, and the RL LoRA was trained on top of that; LoRA deltas add, so the plain base plus both adapters is that merged model plus the RL adapter. Extra kwargs go to load_adapters. The yaml is read before any adapter is loaded; a load that then fails on the RL adapter (no such step) leaves `{name}_warm` loaded, so `model.delete_adapter` it or use another name.
    Returns (PeftModel, meta): `meta` is load_nla_meta(repo) (or `meta` when given) plus `adapters`, the two names, and `layer`, extraction.layer_index, the only layer the checkpoint reads. Loading sets no adapter state of its own: on a raw HF model peft leaves the first adapter, `{name}_warm`, active alone, which is neither the base nor the verbalizer (check_active refuses it), and on an existing PeftModel whatever was active stays. So run `model.base_model.set_adapter(meta["adapters"])` before reading, and `with model.disable_adapter():` around the forward that captures activations. boot_bridge(hf) comes after this call."""
    names = [f"{name}_warm", f"{name}_rl"]
    meta = load_nla_meta(repo) if meta is None else meta
    layer = meta["extraction"]["layer_index"]
    model = load_adapters(model, dict(zip(names, [f"{repo}:{NLA_WARM}", f"{repo}:av_rl_adapters/iter_{step:06d}"])), **kwargs)
    return model, {**meta, "adapters": names, "layer": layer}

def nla_prompt(tokenizer, meta: dict, enable_thinking: bool = False) -> tuple[list[int], int]:
    """(ids, slot) of the verbalizer prompt: the actor template with the injection character in its <concept> tags, rendered as one user turn with the generation prompt and, by default, thinking off, which the checkpoint card recommends (Qwen3.6's template otherwise ends the prompt with an open <think> block). EasyNLA rendered the training prompts with the template's default, so enable_thinking=True reproduces the training render. `slot` is the index of the one token whose id is the trained injection_token_id, whose neighbors must be the trained left and right neighbor ids: the marker is not scanned for, since its embedding runs through blocks 0 and 1 before the injection, so the id is part of the checkpoint. Raises when the render holds any other number of that id or other neighbors (another tokenizer or template)."""
    tk = meta["tokens"]
    ids = to_ids([{"role": "user", "content": meta["prompt_templates"]["actor"].format(injection_char=tk["injection_char"])}], tokenizer, add_generation_prompt=True, enable_thinking=enable_thinking)
    slots = [i for i, x in enumerate(ids) if x == tk["injection_token_id"]]
    if len(slots) != 1:
        raise ValueError(f"the rendered actor prompt holds {len(slots)} tokens with the injection id {tk['injection_token_id']} ({tk['injection_char']!r}), not one")
    slot = slots[0]
    neighbors, trained = (ids[slot - 1], ids[slot + 1]), (tk["injection_left_neighbor_id"], tk["injection_right_neighbor_id"])
    if neighbors != trained:
        raise ValueError(f"the marker's neighbors are {neighbors} ({tokenizer.decode(neighbors[0])!r}, {tokenizer.decode(neighbors[1])!r}), not the trained {trained}: the tokenizer or chat template differs from the checkpoint's")
    return ids, slot

def parse_explanation(text: str) -> str:
    """The text after "<explanation>" (all of it when the tag is absent) up to "</explanation>" when present, stripped; a sample can end at eos before the closing tag."""
    return text.split("<explanation>", 1)[-1].split("</explanation>", 1)[0].strip()

def nla_write(orig: Tensor, vecs: Tensor) -> Tensor:
    """The norm-matched add the NLA was trained with (Karvonen et al. 2025, eq. 1): orig + ||orig|| * vecs / ||vecs||, norms along the last dim, on raw residual vectors."""
    return orig + orig.norm(dim=-1, keepdim=True) * vecs / vecs.norm(dim=-1, keepdim=True)

def nla_read(model, tokenizer, h: Tensor, meta: dict, n: int = 1, seed: int | None = 0, raw: bool = False, enable_thinking: bool = False, do_sample: bool = True, temperature: float = 1.0, top_p: float = 0.95, top_k: int = 64, max_new_tokens: int = 256) -> list[str] | list[list[str]]:
    """The verbalizer's descriptions of residual activations `h` from the checkpoint's layer (meta["layer"], the output of that decoder block): `n` samples per activation, a list of n strings for an [d] vector or m lists of n for [m, d], each the text inside the <explanation> tags (parse_explanation) unless `raw`.
    Each activation is norm-match added (nla_write) to the output of decoder block 1 at the marker slot of the actor prompt (nla_prompt), and all m * n rows are generated as one batch, activation-major (row i * n + j is activation i's sample j). `seed` seeds torch before generating; `enable_thinking` goes to nla_prompt; the sampling arguments are WorkspaceBench's and all go to generate. `model` is the PeftModel from load_nla with both adapters active (check_active raises otherwise, before anything is generated), or that model merged and unloaded."""
    check_active(model, meta["adapters"])
    ids, slot = nla_prompt(tokenizer, meta, enable_thinking)
    hs = t.atleast_2d(h)
    if hs.ndim != 2 or hs.shape[1] != meta["extraction"]["d_model"]:
        raise ValueError(f"h must be [d] or [m, d] with d = {meta['extraction']['d_model']}, got {tuple(h.shape)}")
    zero = (hs.norm(dim=-1) == 0).nonzero().flatten().tolist()
    if zero:
        raise ValueError(f"activations {zero} of h are zero vectors, which have no direction to norm-match (a padded or masked position?)")
    out = inject_generate(model, tokenizer, ids, decoder_layers(model)[1], [slot], hs.repeat_interleave(n, dim=0), nla_write, max_new_tokens, seed, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
    texts = out if raw else [parse_explanation(s) for s in out]
    rows = [texts[i * n:(i + 1) * n] for i in range(len(hs))]
    return rows[0] if h.ndim == 1 else rows

def nla_readout(cache, pos: int | list[int], model, tokenizer, meta: dict, n: int = 1, seed: int | None = 0, raw: bool = False, hook: str = "hook_resid_post", input_src=None, title: str | None = "NLA readout", ctx: int = 32, **sampling) -> dict[int, list[str]]:
    """nla_read of the residual at `pos` (an int, or a list for a bar of tabs, one per position) in a Bridge run_with_cache cache of a single prompt, shown as a table of the n samples per position and returned as {pos: samples}. The cache must come from a forward with the adapters disabled (`with model.disable_adapter(): _, cache = bridge.run_with_cache(ids)`), or the verbalizer reads its own activations.
    The layer is meta["layer"], fixed by the checkpoint. The default hook is hook_resid_post because the checkpoint's "layer L" is the output of decoder block L, which is blocks.L.hook_resid_post (hook_resid_pre would be block L - 1's output). input_src (see lens.get_toks) shows the token strip, whose marked tokens switch position. Extra kwargs are nla_read's enable_thinking and sampling arguments."""
    positions = [pos] if isinstance(pos, int) else list(pos)
    if len(set(positions)) != len(positions):
        raise ValueError(f"pos repeats a position: {positions}")
    texts = nla_read(model, tokenizer, cache[f"blocks.{meta['layer']}.{hook}"][0, positions], meta, n, seed, raw, **sampling)
    toks, ids = get_toks(input_src, tokenizer)
    cell = lambda s: f"<div style='text-align:left;white-space:pre-wrap;max-width:100ch'>{html.escape(s)}</div>"
    panes = {f"p{p}": readout_grid([(f"p{p}" + (f" &middot; {html.escape(repr(toks[p]))}" if toks else ""), [([f"<span style='color:#999'>#{j}</span>", cell(s)], None) for j, s in enumerate(samples, 1)])]) for p, samples in zip(positions, texts)}
    display(HTML(readout_html(tabbed({"": panes}, token_strip(toks, ids, pos, ctx) if toks is not None else ""), title, n_cols=1)))
    return dict(zip(positions, texts))
