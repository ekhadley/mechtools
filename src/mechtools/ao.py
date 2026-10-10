"""Activation oracles (Karvonen et al., "Activation Oracles", arXiv 2512.15674): a LoRA on the subject model (r 64, alpha 128, every linear module) that answers a natural-language question about one or more of the subject model's own activation vectors. Checkpoints for 12 subject models across Gemma-2, Gemma-3, Qwen3 and Llama-3 are in the Hub collection adamkarvonen/activation-oracles, e.g. adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B; each repo holds the adapter and an ao_config.json, which load_ao_config reads.

A read: ao_prompt renders one user turn, "Layer: {layer}\\n" + k placeholders (config["special_token"], " ?", one token on these tokenizers) + " \\n" + the question, through the chat template with the generation prompt and thinking off; ao_read generates from it with the oracle adapter active and the k vectors written into the output of decoder block config["hook_onto_layer"] (1) at the placeholder positions on the prefill. A vector v enters the placeholder's residual h as ||h|| * v / ||v|| * config["steering_coefficient"], added to h (config["op"] "add", the paper's eq. 1) or in place of it ("replace"): checkpoints trained since the repo's addition commit of 2025-10-26 expect the add and checkpoints trained before it expect the replacement, so set config["op"] = "replace" for one of those.

The activations are the subject model's, captured with the adapter disabled at the output of decoder block L, "layer L" in the checkpoints' convention (`blocks.L.hook_resid_post` in the Bridge, HF hidden_states[L + 1]): hf = load_hf_model(base); model, config = load_ao(hf, repo); bridge = boot_bridge(hf); with model.disable_adapter(): _, cache = bridge.run_with_cache(ids). The read functions never change which adapters are active: they raise unless the oracle's adapter is exactly the active one, and the caller switches adapters with peft's own API."""

import html
import json
import os

import torch as t
from torch import Tensor
from huggingface_hub import hf_hub_download
from IPython.display import HTML, display
from peft import PeftModel

from mechtools.hooks import decoder_layers, inject_generate
from mechtools.lens import get_toks, readout_grid, readout_html, tabbed, token_strip
from mechtools.models import adapter_spec, check_active, load_adapters
from mechtools.stats import normed
from mechtools.tokens import to_ids

OPS = {"add": lambda orig, v: orig + orig.norm(dim=-1, keepdim=True) * v, "replace": lambda orig, v: orig.norm(dim=-1, keepdim=True) * v}

def load_ao_config(repo: str, **hub_kwargs) -> dict:
    """The checkpoint's ao_config.json as a dict; `repo` is a local directory, a Hub repo or `repo:subdir` (see adapter_spec), and `hub_kwargs` (revision, token) go to hf_hub_download. Its keys: special_token (" ?"), hook_onto_layer (1), steering_coefficient (1.0), act_layer_combinations (the source layers it was trained on, as outputs of block L, e.g. [[8], [16], [24]]; the oracle generalizes to other layers), layer_combinations (the same as percent of depth), model_name (the base), generation_kwargs (the training-time eval's), prefix_template, created_at_utc and git_commit (which date the checkpoint against the addition commit), the LoRA settings and the dataset names."""
    model_id, hub = adapter_spec(repo)
    path = os.path.join(model_id, "ao_config.json") if os.path.isdir(model_id) else hf_hub_download(model_id, "ao_config.json", **hub, **hub_kwargs)
    with open(path) as f:
        return json.load(f)

def load_ao(model, repo: str, name: str = "ao", config: dict | None = None, **kwargs) -> tuple[PeftModel, dict]:
    """The oracle adapter loaded onto `model` (a raw HF model, or the PeftModel of an earlier load) under `name` through load_adapters, which gets the kwargs (`revision` and `token` also reach the config download), and its config: load_ao_config(repo) unless one is given, copied with "adapter" set to `name` and "op" to "add" unless the config has one. Returns (PeftModel, config). On a raw HF model the adapter is active afterwards; on a PeftModel the adapter already active stays active (peft's load_adapter does not switch), so call model.set_adapter(name) before reading. boot_bridge(model) comes after this."""
    peft_model = load_adapters(model, {name: repo}, **kwargs)
    return peft_model, {"op": "add", **(load_ao_config(repo, **{k: kwargs[k] for k in ("revision", "token") if k in kwargs}) if config is None else config), "adapter": name}

def ao_prompt(tokenizer, config: dict, layer: int, k: int, question: str) -> tuple[list[int], list[int]]:
    """(ids, placeholder positions) of the oracle prompt for `k` activations read from layer `layer`: one user turn, "Layer: {layer}\\n" + config["special_token"] * k + " \\n" + question, rendered with add_generation_prompt=True and enable_thinking=False (the Llama-3 template adds its default system header with the date, which does not matter: the placeholders are found by id). The special token must be one token, and exactly k consecutive ids must equal it, so a question that contains the placeholder string raises."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    special = tokenizer.encode(config["special_token"], add_special_tokens=False)
    if len(special) != 1:
        raise ValueError(f"special_token {config['special_token']!r} is {len(special)} tokens {special} on this tokenizer, not one")
    ids = to_ids([{"role": "user", "content": f"Layer: {layer}\n" + config["special_token"] * k + " \n" + question}], tokenizer, add_generation_prompt=True, enable_thinking=False)
    positions = [i for i, x in enumerate(ids) if x == special[0]]
    if len(positions) != k or positions != list(range(positions[0], positions[0] + k)):
        raise ValueError(f"need {k} consecutive placeholder tokens (id {special[0]}, {config['special_token']!r}) in the rendered prompt, found {len(positions)} at {positions}; a question holding the placeholder string adds to the count")
    return ids, positions

def ao_read(model, tokenizer, hs: Tensor, layer: int, question: str, config: dict, n: int = 1, seed: int | None = 0, do_sample: bool = False, max_new_tokens: int = 50, **sampling) -> list[str]:
    """`n` answers of the oracle to `question` about the activations `hs`, [k, d] (k vectors, one per placeholder, in order) or [d], read from layer `layer` of the subject model. config["adapter"] must be exactly the active adapter of `model` (check_active raises otherwise, also inside a disable_adapter() block; the caller sets adapters). The vectors are written into the output of decoder block config["hook_onto_layer"] at the placeholder positions by config["op"]: "add" gives h + ||h|| * v / ||v|| * coeff and "replace" gives ||h|| * v / ||v|| * coeff, with coeff = config["steering_coefficient"]. Greedy with 50 new tokens by default, like the paper's demo; any other sampling kwarg (temperature, top_p, top_k) reaches generate only when passed, and with do_sample=True each one not passed comes from the model's generation config (Llama-3.2-1B-Instruct's 0.6 and 0.9, Qwen3's 0.6, 0.95 and top_k 20, HF's own top_k 50), so pass all three. `seed` seeds torch first. The n rows run as one batch, all from the same prompt."""
    check_active(model, [config["adapter"]])
    if config["op"] not in OPS:
        raise ValueError(f"config['op'] is {config['op']!r}, not 'add' or 'replace'")
    if hs.ndim not in (1, 2):
        raise ValueError(f"hs must be [k, d] or [d], got {tuple(hs.shape)}")
    hs = t.atleast_2d(hs)
    ids, positions = ao_prompt(tokenizer, config, layer, hs.shape[0], question)
    write = lambda orig, vecs: OPS[config["op"]](orig, normed(vecs) * config["steering_coefficient"])
    return inject_generate(model, tokenizer, ids, decoder_layers(model)[config["hook_onto_layer"]], positions, hs[None].repeat(n, 1, 1), write, max_new_tokens, seed, do_sample=do_sample, **sampling)

def ao_readout(cache, layers, pos: int | list[int], model, tokenizer, config: dict, questions: list[str], window: int = 1, seed: int | None = 0, hook: str = "hook_resid_post", input_src=None, title: str = "activation oracle readout", ctx: int = 32, **sampling) -> dict:
    """The oracle's answer to each of `questions` at every layer in `layers` and position in `pos` (an int or a list), from a run_with_cache cache of a single prompt captured with the adapter disabled (`with model.disable_adapter(): _, cache = bridge.run_with_cache(ids)`; with it active the cache holds the oracle's activations, not the subject's). `model` is the PeftModel with the oracle adapter active. The `window` activations at positions p - window + 1 .. p go to the oracle together, one placeholder each.
    The default hook is hook_resid_post, the output of block L, which is what the checkpoints' "Layer: L" means; hook_resid_pre at L is block L - 1's output, and the prompt would name the wrong layer. One greedy generation per (layer, position, question), since every question is its own prompt; `seed` and `sampling` go to ao_read.
    Displays a tab per layer and, when pos is a list, a second bar per position, each pane a table with a row per question; the token strip of `input_src` (see get_toks) marks the positions, and clicking a marked token switches to it, so input_src must hold as many tokens as the cache. Returns {layer: {pos: {question: answer}}} for a list of positions, keyed as given, and {layer: {question: answer}} for an int, like cluster_readout."""
    if isinstance(questions, str):
        raise TypeError(f"questions is a list of strings, got the one string {questions!r}")
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    toks, ids = get_toks(input_src, tokenizer)
    answers, panes = {}, {}
    for layer in layers:
        key = f"blocks.{layer}.{hook}"
        if cache[key].shape[0] != 1:
            raise ValueError(f"{key} is a batch of {cache[key].shape[0]}; ao_readout takes the cache of one prompt")
        acts = cache[key][0]
        if toks is not None and len(toks) != acts.shape[0]:
            raise ValueError(f"input_src has {len(toks)} tokens but the cache holds {acts.shape[0]} positions")
        answers[layer], panes[f"L{layer}"] = {}, {}
        for p in [pos] if isinstance(pos, int) else pos:
            q = p % acts.shape[0]
            if not -acts.shape[0] <= p < acts.shape[0] or q - window + 1 < 0:
                raise ValueError(f"a window of {window} at position {p} reaches outside the {acts.shape[0]}-token sequence")
            answers[layer][p] = {question: ao_read(model, tokenizer, acts[q - window + 1:q + 1], layer, question, config, 1, seed, **sampling)[0] for question in questions}
            header = f"L{layer} &middot; p{q}" if window == 1 else f"L{layer} &middot; p{q - window + 1}..p{q}"
            panes[f"L{layer}"][f"p{p}"] = readout_grid([(header, [([html.escape(question), f"<div style='white-space:normal;text-align:left;max-width:80ch'>{html.escape(answer)}</div>"], None) for question, answer in answers[layer][p].items()])])
    display(HTML(readout_html(tabbed(panes, token_strip(toks, ids, pos, ctx) if toks is not None else ""), title, n_cols=1)))
    return answers if isinstance(pos, list) else {layer: v[pos] for layer, v in answers.items()}
