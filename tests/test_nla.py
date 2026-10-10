import copy
import html
import re
from types import SimpleNamespace

import pytest
import torch as t
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

from conftest import TINY_MODEL, load_tokenizer
from mechtools import nla
from mechtools.hooks import decoder_layers
from mechtools.models import boot_bridge, check_active
from mechtools.nla import *

# the actor template of ceselder/qwen3.6-27b-nla-rl's nla_meta.yaml, verbatim
ACTOR = "You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.\n\nWe will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.\n\nHere is the vector:\n\n<concept>{injection_char}</concept>\n\nPlease provide an explanation."
D = 16
TARGETS = {NLA_WARM: ["q_proj", "v_proj"], "av_rl_adapters/iter_000400": ["q_proj", "k_proj", "v_proj", "o_proj"]}

def render(tok, char: str) -> list[int]:
    return tok.apply_chat_template([{"role": "user", "content": ACTOR.format(injection_char=char)}], tokenize=True, add_generation_prompt=True, return_dict=False)

@pytest.fixture(scope="module")
def repo(tmp_path_factory) -> str:
    """A fake NLA repo on TINY_MODEL: two random LoRAs with different target sets at the checkpoint's subdirectories, and a nla_meta.yaml with the real structure whose token ids are measured on the tiny tokenizer's render of the actor template with 'Z' as the injection character."""
    tok = load_tokenizer(TINY_MODEL)
    path = tmp_path_factory.mktemp("nla")
    t.manual_seed(0)
    for sub, targets in TARGETS.items():
        m = get_peft_model(AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32), LoraConfig(r=4, lora_alpha=8, target_modules=targets))
        for name, p in m.named_parameters():
            if "lora_B" in name:
                p.data.normal_()
        m.save_pretrained(path / sub)
    ids = render(tok, "Z")
    slot, = [i for i, x in enumerate(ids) if tok.decode(x) == "Z"]
    meta = {"extraction": {"layer_index": 1, "base_model": TINY_MODEL, "d_model": D, "norm": "none"}, "tokens": {"injection_char": "Z", "injection_token_id": ids[slot], "injection_left_neighbor_id": ids[slot - 1], "injection_right_neighbor_id": ids[slot + 1]}, "prompt_templates": {"actor": ACTOR, "critic": "reconstruct {injection_char}"}}
    (path / "nla_meta.yaml").write_text(yaml.safe_dump(meta))
    return str(path)

@pytest.fixture(scope="module")
def loaded(repo):
    """(PeftModel, meta, tokenizer): load_nla on a fresh tiny model."""
    model, meta = load_nla(AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32), repo=repo)
    return model, meta, load_tokenizer(TINY_MODEL)

@pytest.fixture
def stack(loaded):
    """`loaded` with both adapters active, whatever the previous test left."""
    model, meta, tok = loaded
    model.base_model.set_adapter(meta["adapters"])
    return model, meta, tok

@pytest.fixture(scope="module")
def booted(repo):
    """The real loading order: hf model, load_nla, boot_bridge(hf). (PeftModel, Bridge, meta, tokenizer) with the stack active."""
    hf = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32)
    model, meta = load_nla(hf, repo=repo)
    model.base_model.set_adapter(meta["adapters"])
    return model, boot_bridge(hf), meta, load_tokenizer(TINY_MODEL)

@pytest.fixture
def shown(monkeypatch):
    out = []
    monkeypatch.setattr(nla, "display", lambda x: out.append(x.data))
    return out

@pytest.fixture
def captured(monkeypatch):
    """inject_generate replaced by a recorder returning one tagged string per row of vecs."""
    calls = []
    def fake(model, tokenizer, ids, module, positions, vecs, write, max_new_tokens=256, seed=0, **generate_kwargs):
        calls.append(SimpleNamespace(model=model, ids=ids, module=module, positions=positions, vecs=vecs, write=write, max_new_tokens=max_new_tokens, seed=seed, kwargs=generate_kwargs))
        return [f"<explanation> s{i} </explanation> junk" for i in range(len(vecs))]
    monkeypatch.setattr(nla, "inject_generate", fake)
    return calls

@pytest.mark.hf
def test_load_nla_meta(repo, tmp_path, monkeypatch):
    meta = load_nla_meta(repo)
    assert meta["extraction"]["layer_index"] == 1 and meta["extraction"]["d_model"] == D and meta["prompt_templates"]["actor"] == ACTOR and "critic" in meta["prompt_templates"]
    assert (meta["tokens"]["injection_char"], meta["tokens"]["injection_token_id"], meta["tokens"]["injection_left_neighbor_id"], meta["tokens"]["injection_right_neighbor_id"]) == ("Z", 29999, 29958, 829)  # 'Z' between '>' and '</' on the tiny Llama-2 tokenizer
    seen = []
    monkeypatch.setattr(nla, "hf_hub_download", lambda r, f: seen.append((r, f)) or f"{repo}/nla_meta.yaml")
    assert load_nla_meta("org/not-a-dir") == meta and seen == [("org/not-a-dir", "nla_meta.yaml")]  # a Hub id goes through hf_hub_download
    with pytest.raises(FileNotFoundError):
        load_nla_meta(str(tmp_path))  # a directory without the file

@pytest.mark.hf
def test_load_nla(loaded, repo):
    model, meta, _ = loaded
    assert isinstance(model, PeftModel) and list(model.peft_config) == ["nla_warm", "nla_rl"] and meta["adapters"] == ["nla_warm", "nla_rl"] and meta["layer"] == 1
    assert meta["extraction"] == load_nla_meta(repo)["extraction"] and meta["tokens"] == load_nla_meta(repo)["tokens"] and meta["prompt_templates"]["actor"] == ACTOR
    assert model.peft_config["nla_warm"].target_modules == set(TARGETS[NLA_WARM]) and model.peft_config["nla_rl"].target_modules == set(TARGETS["av_rl_adapters/iter_000400"])
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    ids = t.tensor([render(load_tokenizer(TINY_MODEL), "Z")])
    model.base_model.set_adapter(meta["adapters"])
    stacked = model(ids).logits
    model.set_adapter("nla_warm")
    warm = model(ids).logits
    with model.disable_adapter():
        base = model(ids).logits
    assert not t.allclose(stacked, warm) and not t.allclose(warm, base) and not t.allclose(stacked, base)  # both adapters change the model, and the stack is neither alone

@pytest.mark.hf
def test_load_nla_options(repo, tmp_path):
    hf = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32)
    with pytest.raises(FileNotFoundError):
        load_nla(hf, repo=str(tmp_path))  # a repo without nla_meta.yaml
    assert not any("lora_" in n for n, _ in hf.named_parameters())  # the yaml is read before any adapter is loaded
    fake_meta = {"extraction": {"layer_index": 7, "d_model": D}, "tokens": {}, "prompt_templates": {}}
    model, meta = load_nla(hf, repo=repo, name="v", meta=fake_meta)
    assert list(model.peft_config) == ["v_warm", "v_rl"] and meta["adapters"] == ["v_warm", "v_rl"] and meta["layer"] == 7 and meta["extraction"] is fake_meta["extraction"]  # a given meta is used, not the file
    assert "adapters" not in fake_meta  # returned as a copy
    assert model.active_adapters == ["v_warm"]  # peft leaves the first adapter active on a raw model: neither the base nor the verbalizer
    with pytest.raises(RuntimeError, match=r"active adapters \['v_warm'\] are not"):
        check_active(model, meta["adapters"])
    with pytest.raises(ValueError, match="already loaded"):
        load_nla(model, repo=repo, name="v")
    with pytest.raises(ValueError, match="adapter_config.json"):
        load_nla(model, repo=repo, name="w", step=200)  # no such RL step in the repo
    assert list(model.peft_config) == ["v_warm", "v_rl", "w_warm"]  # load_adapters loads name by name, so the warm adapter of the failed stack stays loaded
    model.base_model.set_adapter(meta["adapters"])
    model2, _ = load_nla(model, repo=repo, name="x", meta=fake_meta)  # a second stack on the same PeftModel
    assert model2 is model and list(model.peft_config) == ["v_warm", "v_rl", "w_warm", "x_warm", "x_rl"] and model.active_adapters == ["v_warm", "v_rl"]  # what was active stays active
    lora_dtypes = lambda m: {p.dtype for n, p in m.named_parameters() if "lora_" in n}
    assert lora_dtypes(load_nla(AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.bfloat16), repo=repo, meta=fake_meta)[0]) == {t.float32}  # peft's default casts the adapters up
    assert lora_dtypes(load_nla(AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.bfloat16), repo=repo, meta=fake_meta, autocast_adapter_dtype=False)[0]) == {t.bfloat16}  # kwargs reach load_adapters

@pytest.mark.hf
def test_nla_prompt(stack):
    _, meta, tok = stack
    ids, slot = nla_prompt(tok, meta)
    assert ids == render(tok, "Z") and slot == 108 and ids[slot] == meta["tokens"]["injection_token_id"] == 29999 and (ids[slot - 1], ids[slot + 1]) == (29958, 829)
    assert tok.decode(ids[slot - 4:slot + 5]) == "<concept>Z</concept>"
    wrong = copy.deepcopy(meta)
    wrong["tokens"]["injection_right_neighbor_id"] = 7
    with pytest.raises(ValueError, match=r"\(29958, 829\).*\(29958, 7\)"):
        nla_prompt(tok, wrong)
    wrong = copy.deepcopy(meta)
    wrong["tokens"]["injection_left_neighbor_id"] = 7
    with pytest.raises(ValueError, match=r"not the trained \(7, 829\)"):
        nla_prompt(tok, wrong)
    absent = copy.deepcopy(meta)
    absent["prompt_templates"]["actor"] = "No marker here."
    with pytest.raises(ValueError, match="0 tokens with the injection id 29999"):
        nla_prompt(tok, absent)
    twice = copy.deepcopy(meta)
    twice["prompt_templates"]["actor"] = ACTOR + " <concept>{injection_char}</concept>"
    with pytest.raises(ValueError, match="2 tokens"):
        nla_prompt(tok, twice)
    other = copy.deepcopy(meta)
    other["tokens"]["injection_token_id"] = 1577  # ' ?', a token the prompt does not hold
    with pytest.raises(ValueError, match="0 tokens with the injection id 1577"):
        nla_prompt(tok, other)

def test_parse_explanation():
    assert parse_explanation("<explanation>\n a dog \n</explanation>") == "a dog"
    assert parse_explanation("pre <explanation> cut at eos") == "cut at eos"
    assert parse_explanation("  no tags  ") == "no tags"
    assert parse_explanation("before </explanation> after") == "before"
    assert parse_explanation("<explanation>a</explanation> <explanation>b</explanation>") == "a"  # the first pair
    assert parse_explanation("") == "" and parse_explanation("<explanation></explanation>") == ""

def test_nla_write():
    t.manual_seed(0)
    orig, v = t.randn(6, 1, D) * 3, t.randn(6, 1, D)
    out = nla_write(orig, v)
    unit = v / v.norm(dim=-1, keepdim=True)
    assert out.shape == orig.shape and t.allclose(out, orig + orig.norm(dim=-1, keepdim=True) * unit, atol=1e-6)
    delta = out - orig
    assert t.allclose(delta.norm(dim=-1), orig.norm(dim=-1), atol=1e-5)  # the step has the residual's norm
    assert t.allclose(t.cosine_similarity(delta, v, dim=-1), t.ones(6, 1), atol=1e-6)  # and points along v
    assert not t.allclose(out.norm(dim=-1), orig.norm(dim=-1), atol=1e-2)  # an add, not a replacement: the norm changes
    assert t.allclose(nla_write(orig, 100 * v), out) and t.allclose(nla_write(orig, 0.01 * v), out, atol=1e-5)  # v's scale does not matter
    assert t.allclose(nla_write(orig, orig), 2 * orig) and t.allclose(nla_write(orig, -orig), t.zeros_like(orig), atol=1e-5)

@pytest.mark.hf
def test_nla_read_refuses_wrong_adapter_state(loaded, monkeypatch):
    model, meta, tok = loaded
    monkeypatch.setattr(nla, "inject_generate", lambda *a, **k: pytest.fail("generated with the wrong adapters active"))
    h = t.randn(D)
    for name in meta["adapters"]:
        model.set_adapter(name)
        with pytest.raises(RuntimeError, match=f"active adapters \\['{name}'\\] are not"):
            nla_read(model, tok, h, meta)
    model.base_model.set_adapter([])
    with pytest.raises(RuntimeError, match=r"active adapters \[\] are not"):
        nla_read(model, tok, h, meta)
    model.base_model.set_adapter(meta["adapters"])
    with model.disable_adapter():
        with pytest.raises(RuntimeError, match="disabled"):
            nla_read(model, tok, h, meta)
    with pytest.raises(RuntimeError, match=r"\['nla_warm', 'nla_rl'\] are not \['nla_warm'\]"):
        nla_readout({"blocks.1.hook_resid_post": t.randn(1, 5, D)}, 2, model, tok, {**meta, "adapters": ["nla_warm"]})  # an extra active adapter fails too

@pytest.mark.hf
def test_nla_read_plumbing(stack, captured):
    model, meta, tok = stack
    ids, slot = nla_prompt(tok, meta)
    t.manual_seed(1)
    h = t.randn(2, D)
    out = nla_read(model, tok, h, meta, n=3, seed=5, temperature=0.7, top_p=0.9, top_k=20, max_new_tokens=11, do_sample=False)
    call, = captured
    assert call.model is model and call.ids == ids and call.module is decoder_layers(model)[1] and call.positions == [slot]
    assert call.write is nla_write and call.max_new_tokens == 11 and call.seed == 5 and call.kwargs == {"do_sample": False, "temperature": 0.7, "top_p": 0.9, "top_k": 20}
    assert call.vecs.shape == (6, D) and t.equal(call.vecs, h.repeat_interleave(3, dim=0)) and all(t.equal(call.vecs[i * 3 + j], h[i]) for i in range(2) for j in range(3))  # activation-major
    assert out == [["s0", "s1", "s2"], ["s3", "s4", "s5"]]
    orig, v = t.randn(6, 1, D), t.randn(6, 1, D)
    assert t.allclose(call.write(orig, v), orig + orig.norm(dim=-1, keepdim=True) * v / v.norm(dim=-1, keepdim=True))
    assert nla_read(model, tok, h[0], meta, n=2) == ["s0", "s1"] and captured[-1].vecs.shape == (2, D) and t.equal(captured[-1].vecs, h[:1].repeat(2, 1))  # [d]: a flat list of n
    assert nla_read(model, tok, h, meta) == [["s0"], ["s1"]] and captured[-1].vecs.shape == (2, D)  # n=1 keeps the nesting of [m, d]
    assert nla_read(model, tok, h[0], meta, raw=True) == ["<explanation> s0 </explanation> junk"]
    assert captured[-1].kwargs == {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64} and captured[-1].max_new_tokens == 256 and captured[-1].seed == 0  # WorkspaceBench's defaults
    nla_read(model, tok, h, meta, seed=None)
    assert captured[-1].seed is None
    assert nla_read(model, tok, h.bfloat16(), meta, n=2) == [["s0", "s1"], ["s2", "s3"]] and captured[-1].vecs.dtype == t.bfloat16  # the dtype is inject_generate's to cast
    for bad in (t.randn(D + 1), t.randn(2, D - 1), t.randn(2, 2, D)):
        with pytest.raises(ValueError, match=f"d = {D}, got {re.escape(str(tuple(bad.shape)))}"):
            nla_read(model, tok, bad, meta)
    with pytest.raises(ValueError, match=r"activations \[1, 2\] of h are zero vectors"):
        nla_read(model, tok, t.stack([h[0], t.zeros(D), t.zeros(D)]), meta)  # a zero vector has no direction, so it would be NaN after nla_write
    with pytest.raises(ValueError, match=r"activations \[0\] of h are zero vectors"):
        nla_read(model, tok, t.zeros(D), meta)
    assert len(captured) == 6  # the bad shapes and the zero vectors raised before generating

@pytest.mark.hf
def test_nla_read_end_to_end(booted):
    """Shapes, determinism, hook removal and batch independence on the booted tiny model. Not proof that the injection reaches the output: block 1 is the tiny model's last block, so a write at its output at the marker slot never feeds a later block or the last position's logits. That the written rows reach later blocks is inject_generate's tests (tests/test_adapters.py, through output_hidden_states)."""
    model, bridge, meta, tok = booted
    layer = decoder_layers(model)[1]
    assert layer is decoder_layers(bridge)[1] and not layer._forward_hooks
    t.manual_seed(0)
    h = t.randn(2, D)
    out = nla_read(model, tok, h, meta, n=2, max_new_tokens=3)
    assert len(out) == 2 and all(len(row) == 2 and all(isinstance(s, str) for s in row) for row in out) and not layer._forward_hooks  # the hook is gone afterwards
    assert nla_read(model, tok, h, meta, n=2, max_new_tokens=3) == out  # the seed makes it deterministic
    assert len(nla_read(model, tok, h[0], meta, n=2, max_new_tokens=3)) == 2
    raw = nla_read(model, tok, h, meta, n=2, max_new_tokens=3, raw=True)
    assert [[parse_explanation(s) for s in row] for row in raw] == out  # raw is the same sample before parsing
    assert nla_read(model, tok, h, meta, n=2, max_new_tokens=3, seed=1) != out or nla_read(model, tok, h, meta, n=2, max_new_tokens=3, seed=2) != out  # another seed samples otherwise
    greedy = nla_read(model, tok, h, meta, n=2, max_new_tokens=3, do_sample=False, seed=None)
    assert greedy[0][0] == greedy[0][1] and greedy[1][0] == greedy[1][1]  # greedy rows of one activation agree
    assert nla_read(model, tok, h[1], meta, max_new_tokens=3, do_sample=False, seed=None) == [greedy[1][0]]  # and do not depend on the batch around them
    plain = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32)
    assert len(nla_read(plain, tok, h[0], meta, n=2, max_new_tokens=2)) == 2  # a non-peft model (merged and unloaded) passes check_active

@pytest.mark.hf
def test_nla_readout(booted, shown, captured):
    model, bridge, meta, tok = booted
    ids, _ = nla_prompt(tok, meta)
    with model.disable_adapter():
        _, cache = bridge.run_with_cache(t.tensor([ids]))
    resid = cache["blocks.1.hook_resid_post"]
    assert resid.shape == (1, len(ids), D)
    out = nla_readout(cache, [3, -1], model, tok, meta, n=2, input_src=ids, temperature=0.5, ctx=2)
    call, = captured
    assert t.equal(call.vecs, resid[0, [3, -1]].repeat_interleave(2, dim=0)) and call.kwargs["temperature"] == 0.5 and call.kwargs["top_k"] == 64
    assert out == {3: ["s0", "s1"], -1: ["s2", "s3"]}
    h = shown[-1]
    assert h.count("class='pane'") == 2 and h.count("class='tb' hidden") == 1 and "<button>p3</button><button>p-1</button>" in h and "NLA readout</h3>" in h
    assert "data-p=0 data-h='pos 3 " in h and f"data-p=1 data-h='pos {len(ids) - 1} " in h and h.count("data-p=") == 2  # the strip marks both positions
    assert "data-h='pos 1 " in h and "data-h='pos 0 " not in h  # ctx reaches the strip: the window starts ctx before the first position
    with pytest.raises(ValueError, match=r"repeats a position: \[3, 3\]"):
        nla_readout(cache, [3, 3], model, tok, meta, input_src=ids)
    assert len(captured) == 1 and len(shown) == 1  # before generating or showing anything
    p3, pl = re.findall(r"<table>.*?</table>", h)
    assert p3.count("<tr") == 3 and f"<th colspan=2>p3 &middot; {html.escape(repr(tok.decode(ids[3])))}</th>" in p3 and ">s0<" in p3 and ">s1<" in p3 and "#1</span>" in p3 and "#2</span>" in p3
    assert "<th colspan=2>p-1 &middot;" in pl and ">s2<" in pl and ">s3<" in pl
    assert "--n:1;" in h
    out = nla_readout(cache, 5, model, tok, meta, input_src=ids, title="T", raw=True, hook="hook_resid_pre")
    assert out == {5: ["<explanation> s0 </explanation> junk"]} and t.equal(captured[-1].vecs, cache["blocks.1.hook_resid_pre"][0, [5]])
    h = shown[-1]
    assert h.count("class='pane'") == 1 and h.count("class='tb' hidden") == 2 and "data-p=" not in h and h.count("outline:1px solid #fc6") == 1 and "T</h3>" in h and "&lt;explanation&gt; s0" in h
    out = nla_readout(cache, [2], model, tok, meta, title=None)
    assert out == {2: ["s0"]} and "<h3" not in shown[-1] and "class='tk'" not in shown[-1]  # no input_src, no strip
    with pytest.raises(KeyError):
        nla_readout(cache, 2, model, tok, {**meta, "layer": 9})

@pytest.mark.hf
def test_nla_prompt_qwen36():
    """The real checkpoint's marker on the Qwen3.6-27B tokenizer: ㈜ (158983) between '>' (29) and '</' (510), and thinking off in the render."""
    tok = load_tokenizer("Qwen/Qwen3.6-27B")
    meta = {"tokens": {"injection_char": "㈜", "injection_token_id": 158983, "injection_left_neighbor_id": 29, "injection_right_neighbor_id": 510}, "prompt_templates": {"actor": ACTOR}}
    ids, slot = nla_prompt(tok, meta)
    assert ids[slot] == 158983 and (ids[slot - 1], ids[slot + 1]) == (29, 510) and ids.count(158983) == 1
    assert tok.decode(ids[slot - 1:slot + 2]) == ">㈜</" and tok.decode(ids).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert ids == tok.apply_chat_template([{"role": "user", "content": ACTOR.format(injection_char="㈜")}], tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False)
