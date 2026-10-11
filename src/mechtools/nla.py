import os

import torch as t
import yaml
from torch import Tensor
from huggingface_hub import hf_hub_download
from IPython.display import HTML, display
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from mechtools.hooks import decoder_layers, inject_generate
from mechtools.lens import fmt3, get_toks, readout_html, tabbed, token_strip
from mechtools.models import check_active, load_adapters
from mechtools.stats import normed
from mechtools.tokens import to_ids

NLA_REPO = "ceselder/qwen3.6-27b-nla-rl"
NLA_WARM = "av_warmstart_lora/iter_0003864"
NLA_AR = "ar_reconstructor"

def load_nla_meta(repo: str = NLA_REPO) -> dict:
    """The checkpoint's nla_meta.yaml as a dict, from `repo`, a local directory or a Hub repo id: `extraction` (layer_index, base_model, d_model, norm), `tokens` (injection_char, injection_token_id, injection_left_neighbor_id, injection_right_neighbor_id) and `prompt_templates` (actor, the verbalizer prompt with an {injection_char} placeholder, and critic, the reconstructor's, with an {explanation} placeholder)."""
    path = os.path.join(repo, "nla_meta.yaml") if os.path.isdir(repo) else hf_hub_download(repo, "nla_meta.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

def load_nla(model, repo: str = NLA_REPO, step: int = 600, name: str = "nla", meta: dict | None = None, **kwargs) -> tuple[PeftModel, dict]:
    """The natural language autoencoder's verbalizer as two toggleable LoRA adapters on `model`, the plain base model (a raw HF model, or the PeftModel of an earlier load_adapters call): `{name}_warm`, the warm-start LoRA at NLA_WARM, and `{name}_rl`, the RL LoRA at av_rl_adapters/iter_{step:06d} (steps 100 to 800). The default, 600, is the step of the one saved reconstructor (load_nla_ar), which was trained alongside the verbalizer, so the default encoder and decoder match; the repo's fve_trajectory.json gives each step's held-out FVE, measured with that step's reconstructor, and WorkspaceBench read step 400. Two adapters because the repo's av_base is the plain base with the warm-start LoRA merged in, and the RL LoRA was trained on top of that; LoRA deltas add, so the plain base plus both adapters is that merged model plus the RL adapter. Extra kwargs go to load_adapters. The yaml is read before any adapter is loaded; a load that then fails on the RL adapter (no such step) leaves `{name}_warm` loaded, so `model.delete_adapter` it or use another name.
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

def load_nla_ar(repo: str = NLA_REPO, **kwargs) -> tuple:
    """The activation reconstructor (the AR, EasyNLA's critic) as (model, head): `model` is ar_reconstructor, Qwen3.5 truncated to blocks 0..42 (a full fine-tune, 35 GB in bf16, sharing no weights with the base), with its final norm and lm_head replaced by the identity, so its decoder's last hidden state is block 42's raw output; `head` is the [d, d] float32 value head, on the model's device. One reconstructor was saved, at RL step 600 (ar_reconstructor/saved_at_step.txt), load_nla's default step. `repo` is a Hub repo id or a local directory; kwargs (dtype, device_map) go to from_pretrained. The verbalizer and the reconstructor take about 89 GB together in bf16; nla_score needs only the reconstructor, so the verbalizer can be freed first."""
    path = os.path.join(repo, NLA_AR) if os.path.isdir(repo) else None
    model = AutoModelForCausalLM.from_pretrained(path or repo, **({} if path else {"subfolder": NLA_AR}), **kwargs).eval().requires_grad_(False)
    model.get_decoder().norm = t.nn.Identity()
    model.lm_head = t.nn.Identity()  # never run: the value head reads the decoder's output
    head = load_file(os.path.join(path, "value_head.safetensors") if path else hf_hub_download(repo, f"{NLA_AR}/value_head.safetensors"))["weight"]
    return model, head.float().to(model.device)

@t.no_grad()
def nla_reconstruct(ar: tuple, tokenizer, explanations: list[str], meta: dict, batch_size: int = 16) -> Tensor:
    """[N, d] float32 on the cpu: the reconstructor's prediction of the activation each explanation describes, as EasyNLA's critic_predict computes it. Each explanation goes into meta["prompt_templates"]["critic"], tokenized as is (no chat template, no special tokens); a batch is right-padded, as EasyNLA batches it, and each row's last real token's output of the truncated model is scaled to norm sqrt(d) and multiplied by the value head. Only the direction was trained: the loss scales prediction and target to the same norm. `ar` is load_nla_ar's (model, head)."""
    model, head = ar
    out = []
    for b in range(0, len(explanations), batch_size):
        enc = tokenizer([meta["prompt_templates"]["critic"].format(explanation=e) for e in explanations[b:b + batch_size]], add_special_tokens=False, padding=True, padding_side="right", return_tensors="pt")
        hid = model.get_decoder()(input_ids=enc.input_ids.to(model.device), attention_mask=enc.attention_mask.to(model.device), use_cache=False).last_hidden_state
        last = hid[t.arange(len(hid)), enc.attention_mask.sum(1) - 1].float().to(head.device)  # cpu indices, wherever a device_map puts the last block
        out.append((normed(last) * head.shape[0] ** 0.5 @ head.T).cpu())
    return t.cat(out)

def nla_score(recs: list[dict], ar: tuple, tokenizer, meta: dict, batch_size: int = 16) -> list[dict]:
    """nla_read's records with three fields added: `recon`, nla_reconstruct of the record's explanation, `cos`, its cosine with the record's activation `h`, and `mse`, 2 * (1 - cos), the MSE of the two scaled to norm sqrt(d), which is EasyNLA's loss, the negative of the RL reward, and what nla_fve averages."""
    recon = nla_reconstruct(ar, tokenizer, [r["explanation"] for r in recs], meta, batch_size)
    cos = t.cosine_similarity(recon, t.stack([r["h"] for r in recs]), dim=-1).tolist()
    return [{**r, "recon": v, "cos": c, "mse": 2 * (1 - c)} for r, v, c in zip(recs, recon, cos)]

def nla_fve(mse, acts: Tensor) -> float:
    """Fraction of variance explained as fve_trajectory.json defines it: 1 - mean(mse) / the MSE of predicting the mean, every vector scaled to norm sqrt(d). That baseline is 1 - ||mean of the unit activations||^2, so it depends on `acts` [N, d], the activations the mean is taken over (the checkpoint's figures used held-out layer-42 activations of FineFineWeb); `mse` is nla_score's per-sample values."""
    return 1 - t.as_tensor(mse, dtype=t.float32).mean().item() / (1 - normed(acts.float()).mean(0).norm() ** 2).item()

def nla_read(model, tokenizer, h: Tensor, meta: dict, n: int = 1, seed: int | None = 0, ar: tuple | None = None, enable_thinking: bool = False, do_sample: bool = True, temperature: float = 1.0, top_p: float = 0.95, top_k: int = 64, max_new_tokens: int = 256) -> list[dict]:
    """The verbalizer's descriptions of residual activations `h` ([d] or [m, d]) from the checkpoint's layer (meta["layer"], the output of that decoder block), `n` samples per activation, as one record per sample, activation-major: `i` (the row of h, 0 for a [d] vector), `j` (the sample), `h` (the activation, float32 on the cpu), `text` (the raw sample), `ids` (its sampled ids up to eos), `stopped` (it ended at eos rather than at max_new_tokens) and `explanation` (parse_explanation of the text). With `ar`, load_nla_ar's reconstructor, nla_score adds recon, cos and mse.
    Each activation is norm-match added (nla_write) to the output of decoder block 1 at the marker slot of the actor prompt (nla_prompt), and all m * n rows are generated as one batch, activation-major (row i * n + j is activation i's sample j). `seed` seeds torch before generating; `enable_thinking` goes to nla_prompt; the sampling arguments are WorkspaceBench's and all go to generate. `model` is the PeftModel from load_nla with both adapters active (check_active raises otherwise, before anything is generated), or that model merged and unloaded."""
    check_active(model, meta["adapters"])
    ids, slot = nla_prompt(tokenizer, meta, enable_thinking)
    hs = t.atleast_2d(h)
    if hs.ndim != 2 or hs.shape[1] != meta["extraction"]["d_model"]:
        raise ValueError(f"h must be [d] or [m, d] with d = {meta['extraction']['d_model']}, got {tuple(h.shape)}")
    zero = (hs.norm(dim=-1) == 0).nonzero().flatten().tolist()
    if zero:
        raise ValueError(f"activations {zero} of h are zero vectors, which have no direction to norm-match (a padded or masked position?)")
    rows = inject_generate(model, tokenizer, ids, decoder_layers(model)[1], [slot], hs.repeat_interleave(n, dim=0), nla_write, max_new_tokens, seed, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
    hc = hs.float().cpu()
    recs = [{"i": k // n, "j": k % n, "h": hc[k // n], **r, "explanation": parse_explanation(r["text"])} for k, r in enumerate(rows)]
    return recs if ar is None else nla_score(recs, ar, tokenizer, meta)

def nla_readout(cache, pos: int | list[int], model, tokenizer, meta: dict, n: int = 1, seed: int | None = 0, ar: tuple | None = None, raw: bool = False, hook: str = "hook_resid_post", input_src=None, title: str | None = "NLA readout", ctx: int = 32, **sampling) -> list[dict]:
    """nla_read of the residual at `pos` (an int, or a list for a bar of tabs, one per position) in a Bridge run_with_cache cache of a single prompt, shown as a table of the n samples per position (the explanation, or the raw text with raw=True, and with `ar` each sample's cosine with the activation), and returned as nla_read's records with `pos` (as given) added, position-major. The cache must come from a forward with the adapters disabled (`with model.disable_adapter(): _, cache = bridge.run_with_cache(ids)`), or the verbalizer reads its own activations.
    The layer is meta["layer"], fixed by the checkpoint. The default hook is hook_resid_post because the checkpoint's "layer L" is the output of decoder block L, which is blocks.L.hook_resid_post (hook_resid_pre would be block L - 1's output). input_src (see lens.get_toks) shows the token strip, whose marked tokens switch position. Extra kwargs are nla_read's enable_thinking and sampling arguments."""
    positions = [pos] if isinstance(pos, int) else list(pos)
    if len(set(positions)) != len(positions):
        raise ValueError(f"pos repeats a position: {positions}")
    recs = [{"pos": positions[r["i"]], **r} for r in nla_read(model, tokenizer, cache[f"blocks.{meta['layer']}.{hook}"][0, positions], meta, n, seed, ar, **sampling)]
    toks, ids = get_toks(input_src, tokenizer)
    scored = ar is not None
    panes = {f"p{p}": [([f"p{p}", repr(toks[p])] if toks else f"p{p}", [[f"#{r['j'] + 1}", r["text" if raw else "explanation"]] + ([fmt3(r["cos"])] if scored else []) for r in recs if r["pos"] == p], None, ["d", "w"] + ["v"] * scored)] for p in positions}  # a dim sample number, the text in a wrapping column, and with the reconstructor its cosine
    display(HTML(readout_html(tabbed({"": panes}, token_strip(toks, ids, pos, ctx) if toks is not None else ""), title, n_cols=1)))
    return recs
