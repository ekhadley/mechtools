import copy
import html
import re
from types import SimpleNamespace

import pytest
import torch as t
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from conftest import TINY_MODEL, load_tokenizer, widget_data
from mechtools import nla
from mechtools.hooks import decoder_layers
from mechtools.lens import fmt3
from mechtools.models import boot_bridge, check_active
from mechtools.nla import *
from mechtools.stats import normed

# the actor template of ceselder/qwen3.6-27b-nla-rl's nla_meta.yaml, verbatim
ACTOR = "You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.\n\nWe will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.\n\nHere is the vector:\n\n<concept>{injection_char}</concept>\n\nPlease provide an explanation."
CRITIC = "Summary of the following text: <text>{explanation}</text> <summary>"  # the critic template of the same file
D = 16
TARGETS = {NLA_WARM: ["q_proj", "v_proj"], "av_rl_adapters/iter_000600": ["q_proj", "k_proj", "v_proj", "o_proj"]}
EXPLANATIONS = ["a dog", "a long explanation of several words about the weather", ""]
texts = lambda recs: [r["text"] for r in recs]

def render(tok, char: str) -> list[int]:
    return tok.apply_chat_template([{"role": "user", "content": ACTOR.format(injection_char=char)}], tokenize=True, add_generation_prompt=True, return_dict=False)

def critic_ids(tok, explanation: str) -> list[int]:
    return tok.encode(CRITIC.format(explanation=explanation), add_special_tokens=False)

def save_ar(path, model):
    """`model` saved as a reconstructor checkpoint at path/ar_reconstructor, with a random value head."""
    model.save_pretrained(path / NLA_AR)
    save_file({"weight": t.randn(D, D)}, path / NLA_AR / "value_head.safetensors")

@pytest.fixture(scope="module")
def repo(tmp_path_factory) -> str:
    """A fake NLA repo on TINY_MODEL: two random LoRAs with different target sets at the checkpoint's subdirectories, the tiny model itself as the reconstructor at ar_reconstructor with a random value head, and a nla_meta.yaml with the real structure whose token ids are measured on the tiny tokenizer's render of the actor template with 'Z' as the injection character."""
    tok = load_tokenizer(TINY_MODEL)
    path = tmp_path_factory.mktemp("nla")
    t.manual_seed(0)
    for sub, targets in TARGETS.items():
        m = get_peft_model(AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32), LoraConfig(r=4, lora_alpha=8, target_modules=targets))
        for name, p in m.named_parameters():
            if "lora_B" in name:
                p.data.normal_()
        m.save_pretrained(path / sub)
    save_ar(path, AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32))
    ids = render(tok, "Z")
    slot, = [i for i, x in enumerate(ids) if tok.decode(x) == "Z"]
    meta = {"extraction": {"layer_index": 1, "base_model": TINY_MODEL, "d_model": D, "norm": "none"}, "tokens": {"injection_char": "Z", "injection_token_id": ids[slot], "injection_left_neighbor_id": ids[slot - 1], "injection_right_neighbor_id": ids[slot + 1]}, "prompt_templates": {"actor": ACTOR, "critic": CRITIC}}
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

@pytest.fixture(scope="module")
def ar(repo):
    """load_nla_ar of the fake repo: the tiny model as the reconstructor."""
    return load_nla_ar(repo)

@pytest.fixture
def shown(monkeypatch):
    out = []
    monkeypatch.setattr(nla, "display", lambda x: out.append(x.data))
    return out

@pytest.fixture
def captured(monkeypatch):
    """inject_generate replaced by a recorder returning one row per row of vecs, its text a tagged string."""
    calls = []
    def fake(model, tokenizer, ids, module, positions, vecs, write, max_new_tokens=256, seed=0, **generate_kwargs):
        calls.append(SimpleNamespace(model=model, ids=ids, module=module, positions=positions, vecs=vecs, write=write, max_new_tokens=max_new_tokens, seed=seed, kwargs=generate_kwargs))
        return [{"text": f"<explanation> s{i} </explanation> junk", "ids": [i], "stopped": i % 2 == 0} for i in range(len(vecs))]
    monkeypatch.setattr(nla, "inject_generate", fake)
    return calls

@pytest.mark.hf
def test_load_nla_meta(repo, tmp_path, monkeypatch):
    meta = load_nla_meta(repo)
    assert meta["extraction"]["layer_index"] == 1 and meta["extraction"]["d_model"] == D and meta["prompt_templates"]["actor"] == ACTOR and meta["prompt_templates"]["critic"] == CRITIC
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
    assert model.peft_config["nla_warm"].target_modules == set(TARGETS[NLA_WARM]) and model.peft_config["nla_rl"].target_modules == set(TARGETS["av_rl_adapters/iter_000600"])  # the RL step defaults to 600, the reconstructor's
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
    assert [list(r) for r in out] == [["i", "j", "h", "text", "ids", "stopped", "explanation"]] * 6  # one record per sample
    assert [(r["i"], r["j"], r["explanation"], r["ids"], r["stopped"]) for r in out] == [(i, j, f"s{3 * i + j}", [3 * i + j], (3 * i + j) % 2 == 0) for i in range(2) for j in range(3)]  # activation-major, the row's fields passed through
    assert out[4]["text"] == "<explanation> s4 </explanation> junk" and all(t.equal(r["h"], h[r["i"]]) and r["h"].device.type == "cpu" for r in out)
    orig, v = t.randn(6, 1, D), t.randn(6, 1, D)
    assert t.allclose(call.write(orig, v), orig + orig.norm(dim=-1, keepdim=True) * v / v.norm(dim=-1, keepdim=True))
    one = nla_read(model, tok, h[0], meta, n=2)
    assert [(r["i"], r["explanation"]) for r in one] == [(0, "s0"), (0, "s1")] and t.equal(one[1]["h"], h[0]) and captured[-1].vecs.shape == (2, D) and t.equal(captured[-1].vecs, h[:1].repeat(2, 1))  # [d] is activation 0
    assert [(r["i"], r["j"]) for r in nla_read(model, tok, h, meta)] == [(0, 0), (1, 0)] and captured[-1].vecs.shape == (2, D)
    assert captured[-1].kwargs == {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64} and captured[-1].max_new_tokens == 256 and captured[-1].seed == 0  # WorkspaceBench's defaults
    nla_read(model, tok, h, meta, seed=None)
    assert captured[-1].seed is None
    half = nla_read(model, tok, h.bfloat16(), meta, n=2)
    assert captured[-1].vecs.dtype == t.bfloat16 and all(r["h"].dtype == t.float32 and t.equal(r["h"], h.bfloat16()[r["i"]].float()) for r in half)  # the dtype is inject_generate's to cast; the records hold float32
    for bad in (t.randn(D + 1), t.randn(2, D - 1), t.randn(2, 2, D)):
        with pytest.raises(ValueError, match=f"d = {D}, got {re.escape(str(tuple(bad.shape)))}"):
            nla_read(model, tok, bad, meta)
    with pytest.raises(ValueError, match=r"activations \[1, 2\] of h are zero vectors"):
        nla_read(model, tok, t.stack([h[0], t.zeros(D), t.zeros(D)]), meta)  # a zero vector has no direction, so it would be NaN after nla_write
    with pytest.raises(ValueError, match=r"activations \[0\] of h are zero vectors"):
        nla_read(model, tok, t.zeros(D), meta)
    assert len(captured) == 5  # the bad shapes and the zero vectors raised before generating

@pytest.mark.hf
def test_nla_read_end_to_end(booted):
    """Shapes, determinism, hook removal and batch independence on the booted tiny model. Not proof that the injection reaches the output: block 1 is the tiny model's last block, so a write at its output at the marker slot never feeds a later block or the last position's logits. That the written rows reach later blocks is inject_generate's tests (tests/test_adapters.py, through output_hidden_states)."""
    model, bridge, meta, tok = booted
    layer = decoder_layers(model)[1]
    assert layer is decoder_layers(bridge)[1] and not layer._forward_hooks
    t.manual_seed(0)
    h = t.randn(2, D)
    out = nla_read(model, tok, h, meta, n=2, max_new_tokens=3)
    assert len(out) == 4 and all(isinstance(r["text"], str) and r["explanation"] == parse_explanation(r["text"]) and len(r["ids"]) <= 3 and (r["stopped"] or len(r["ids"]) == 3) for r in out) and not layer._forward_hooks  # the hook is gone afterwards
    assert all(r["text"] == tok.decode(r["ids"], skip_special_tokens=True) for r in out)
    assert texts(nla_read(model, tok, h, meta, n=2, max_new_tokens=3)) == texts(out)  # the seed makes it deterministic
    assert len(nla_read(model, tok, h[0], meta, n=2, max_new_tokens=3)) == 2
    assert texts(nla_read(model, tok, h, meta, n=2, max_new_tokens=3, seed=1)) != texts(out) or texts(nla_read(model, tok, h, meta, n=2, max_new_tokens=3, seed=2)) != texts(out)  # another seed samples otherwise
    greedy = texts(nla_read(model, tok, h, meta, n=2, max_new_tokens=3, do_sample=False, seed=None))
    assert greedy[0] == greedy[1] and greedy[2] == greedy[3]  # greedy rows of one activation agree
    assert texts(nla_read(model, tok, h[1], meta, max_new_tokens=3, do_sample=False, seed=None)) == [greedy[2]]  # and do not depend on the batch around them
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
    assert [(r["pos"], r["i"], r["j"], r["explanation"]) for r in out] == [(3, 0, 0, "s0"), (3, 0, 1, "s1"), (-1, 1, 0, "s2"), (-1, 1, 1, "s3")] and t.equal(out[2]["h"], resid[0, -1])  # nla_read's records with the position as given
    h = shown[-1]
    d = widget_data(h)
    assert d["tabs"] == [""] and d["inner"] == ["p3", "p-1"] and h.count("class='tb' hidden") == 1 and "<button class=on>p3</button><button>p-1</button>" in h and "NLA readout</h3>" in h
    assert "data-p=0 class=on data-h='pos 3 " in h and f"data-p=1 data-h='pos {len(ids) - 1} " in h and h.count("data-p=") == 2  # the strip marks both positions, the first selected
    assert "data-h='pos 1 " in h and "data-h='pos 0 " not in h  # ctx reaches the strip: the window starts ctx before the first position
    with pytest.raises(ValueError, match=r"repeats a position: \[3, 3\]"):
        nla_readout(cache, [3, 3], model, tok, meta, input_src=ids)
    assert len(captured) == 1 and len(shown) == 1  # before generating or showing anything
    none, (pl,) = d["panes"][0]  # one table per position: the position and its token in the header, a dim sample number and the text in a wrapping column per row; the first pane rendered, not shipped
    assert none is None and pl == [["p-1", repr(tok.decode(ids[-1]))], [["#1", "s2"], ["#2", "s3"]], None, ["d", "w"]]
    assert h.count("<table>") == 1 and h.count("<tr") == 3 and f"<th colspan=2>p3 &middot; {html.escape(repr(tok.decode(ids[3])), False)}</th>" in h and "<td class=d>#1</td><td class=w><div>s0</div></td>" in h and "s2" not in h.split("application/json")[0]  # the first pane rendered
    assert "--n:1;" in h
    out = nla_readout(cache, 5, model, tok, meta, input_src=ids, title="T", raw=True, hook="hook_resid_pre")
    assert [(r["pos"], r["text"]) for r in out] == [(5, "<explanation> s0 </explanation> junk")] and t.equal(captured[-1].vecs, cache["blocks.1.hook_resid_pre"][0, [5]])
    h = shown[-1]
    assert "application/json" not in h and "<th colspan=2>p5 &middot; " in h and h.count("class='tb' hidden") == 2 and "data-p=" not in h and h.count("outline:1px solid #fc6") == 1 and "T</h3>" in h and "&lt;explanation&gt; s0" in h
    out = nla_readout(cache, [2], model, tok, meta, title=None)
    assert [(r["pos"], r["explanation"]) for r in out] == [(2, "s0")] and "<h3" not in shown[-1] and "class='tk'" not in shown[-1]  # no input_src, no strip
    with pytest.raises(KeyError):
        nla_readout(cache, 2, model, tok, {**meta, "layer": 9})

@pytest.mark.hf
def test_load_nla_ar(repo, ar, monkeypatch):
    model, head = ar
    assert head.dtype == t.float32 and head.shape == (D, D) and t.equal(head, load_file(f"{repo}/{NLA_AR}/value_head.safetensors")["weight"])
    assert isinstance(model.get_decoder().norm, t.nn.Identity) and isinstance(model.lm_head, t.nn.Identity) and not model.training and not any(p.requires_grad for p in model.parameters())
    seen = []
    class Auto:
        def from_pretrained(name, **kw):
            seen.append((name, kw))
            return AutoModelForCausalLM.from_pretrained(f"{repo}/{NLA_AR}")
    monkeypatch.setattr(nla, "AutoModelForCausalLM", Auto)
    monkeypatch.setattr(nla, "hf_hub_download", lambda r, f: seen.append((r, f)) or f"{repo}/{NLA_AR}/value_head.safetensors")
    assert t.equal(load_nla_ar("org/nla", dtype=t.bfloat16)[1], head)
    assert seen == [("org/nla", {"subfolder": NLA_AR, "dtype": t.bfloat16}), ("org/nla", f"{NLA_AR}/value_head.safetensors")]  # a Hub id: the reconstructor's subfolder and the head's path in the repo; kwargs reach from_pretrained

@pytest.mark.hf
def test_nla_reconstruct_is_the_value_head_on_the_last_block_output(ar, loaded):
    """A padded batch against each explanation alone through a fresh copy of the model, built by hand: the last block's raw output (the final norm skipped) at the last token of the critic template, scaled to norm sqrt(d), times the head."""
    _, meta, tok = loaded
    _, head = ar
    assert len({len(critic_ids(tok, e)) for e in EXPLANATIONS}) == 3  # the rows of a batch differ in length, so they are padded
    fresh = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32)
    outs = []
    handle = decoder_layers(fresh)[-1].register_forward_hook(lambda m, a, o: outs.append(o[0] if isinstance(o, tuple) else o))
    with t.no_grad():
        for e in EXPLANATIONS:
            fresh(t.tensor([critic_ids(tok, e)]))
    handle.remove()
    ref = t.stack([normed(o[0, -1]) * D ** 0.5 @ head.T for o in outs])
    got = nla_reconstruct(ar, tok, EXPLANATIONS, meta, batch_size=2)
    assert got.shape == (3, D) and got.dtype == t.float32 and got.device.type == "cpu"
    t.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    t.testing.assert_close(nla_reconstruct(ar, tok, EXPLANATIONS, meta, batch_size=1), got, rtol=1e-5, atol=1e-5)

@pytest.mark.hf
def test_nla_reconstruct_on_a_qwen35_reconstructor(tmp_path, loaded):
    """The real reconstructor's architecture, a random 3-layer Qwen3.5 with linear-attention layers, saved and loaded through load_nla_ar: a padded batch gives each explanation's prediction alone."""
    _, meta, tok = loaded
    t.manual_seed(0)
    save_ar(tmp_path, Qwen3_5ForCausalLM(Qwen3_5TextConfig(vocab_size=32000, hidden_size=D, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=8, linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2, linear_num_value_heads=4, layer_types=["linear_attention", "full_attention", "linear_attention"])))
    ar = load_nla_ar(str(tmp_path))
    assert type(ar[0]) is Qwen3_5ForCausalLM and isinstance(ar[0].get_decoder().norm, t.nn.Identity)
    t.testing.assert_close(nla_reconstruct(ar, tok, EXPLANATIONS, meta, batch_size=3), nla_reconstruct(ar, tok, EXPLANATIONS, meta, batch_size=1), rtol=1e-5, atol=1e-5)

@pytest.mark.hf
def test_nla_score_and_fve(ar, loaded):
    """cos against h, mse = 2(1 - cos) as EasyNLA's per-sample MSE of both vectors scaled to norm sqrt(d), and nla_fve against EasyNLA's predict-the-mean baseline computed elementwise."""
    _, meta, tok = loaded
    t.manual_seed(2)
    h = t.randn(3, D)
    recs = [{"i": i, "h": h[i], "explanation": e} for i, e in enumerate(EXPLANATIONS)]
    scored = nla_score(recs, ar, tok, meta)
    recon = nla_reconstruct(ar, tok, EXPLANATIONS, meta)
    assert [list(r) for r in scored] == [["i", "h", "explanation", "recon", "cos", "mse"]] * 3 and all(s[k] is r[k] for r, s in zip(recs, scored) for k in r)  # fields added, the others passed through
    assert t.equal(t.stack([r["recon"] for r in scored]), recon) and all(abs(r["cos"] - c) < 1e-6 and r["mse"] == 2 * (1 - r["cos"]) for r, c in zip(scored, t.cosine_similarity(recon, h, dim=-1).tolist()))
    scale = lambda v: v / v.norm(dim=-1, keepdim=True) * D ** 0.5  # EasyNLA's normalize_activation to mse_scale, sqrt(d)
    mse = [r["mse"] for r in scored]
    t.testing.assert_close(t.tensor(mse), ((scale(recon) - scale(h)) ** 2).mean(-1), rtol=1e-5, atol=1e-5)
    acts = t.randn(50, D) + 1  # a reference set with a mean direction
    baseline = ((scale(acts) - scale(acts).mean(0)) ** 2).mean()  # compute_predict_mean_baselines' raw-variance baseline, the FVE denominator
    assert abs(nla_fve(mse, acts) - (1 - t.tensor(mse).mean() / baseline).item()) < 1e-5
    assert abs(nla_fve([0.0], acts) - 1) < 1e-6 and nla_fve([2.0], acts) < 0  # a perfect reconstruction explains everything; an orthogonal one does worse than predicting the mean

@pytest.mark.hf
def test_nla_read_and_readout_with_the_reconstructor(stack, ar, captured, shown):
    model, meta, tok = stack
    t.manual_seed(3)
    h = t.randn(2, D)
    out = nla_read(model, tok, h, meta, n=2, ar=ar)
    plain = nla_score(nla_read(model, tok, h, meta, n=2), ar, tok, meta)
    assert [list(r) for r in out] == [["i", "j", "h", "text", "ids", "stopped", "explanation", "recon", "cos", "mse"]] * 4
    assert [r["cos"] for r in out] == [r["cos"] for r in plain] and all(t.equal(a["recon"], b["recon"]) for a, b in zip(out, plain))  # ar= is nla_score of the records
    cache = {"blocks.1.hook_resid_post": t.randn(1, 6, D)}
    recs = nla_readout(cache, [1, 4], model, tok, meta, n=2, ar=ar)
    assert [(r["pos"], r["explanation"]) for r in recs] == [(1, "s0"), (1, "s1"), (4, "s2"), (4, "s3")] and all(isinstance(r["cos"], float) for r in recs)
    (pl,) = widget_data(shown[-1])["panes"][0][1]
    assert pl[1] == [["#1", "s2", fmt3(recs[2]["cos"])], ["#2", "s3", fmt3(recs[3]["cos"])]] and pl[3] == ["d", "w", "v"]  # a value column of cosines
    assert f"<td class=w><div>s0</div></td><td class=v>{fmt3(recs[0]['cos'])}</td>" in shown[-1]  # the first pane rendered

@pytest.mark.hf
def test_nla_prompt_qwen36():
    """The real checkpoint's marker on the Qwen3.6-27B tokenizer: ㈜ (158983) between '>' (29) and '</' (510), and thinking off in the render."""
    tok = load_tokenizer("Qwen/Qwen3.6-27B")
    meta = {"tokens": {"injection_char": "㈜", "injection_token_id": 158983, "injection_left_neighbor_id": 29, "injection_right_neighbor_id": 510}, "prompt_templates": {"actor": ACTOR}}
    ids, slot = nla_prompt(tok, meta)
    assert ids[slot] == 158983 and (ids[slot - 1], ids[slot + 1]) == (29, 510) and ids.count(158983) == 1
    assert tok.decode(ids[slot - 1:slot + 2]) == ">㈜</" and tok.decode(ids).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert ids == tok.apply_chat_template([{"role": "user", "content": ACTOR.format(injection_char="㈜")}], tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False)
