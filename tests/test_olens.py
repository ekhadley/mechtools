import json
import shutil

import pytest
import torch as t
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma4Config, Gemma4ForConditionalGeneration

from conftest import TINY_MODEL, first_pane, load_tokenizer, widget_data
from mechtools import olens
from mechtools.hooks import decoder_layers
from mechtools.models import boot_bridge, load_adapters
from mechtools.lens import readout_grid
from mechtools.olens import *
from mechtools.stats import normed
from mechtools.tokens import single_token_marker, to_ids

D = 16
GREEDY = {"do_sample": False, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 3}
td = lambda s: f"<td class=w><div>{s}</div></td>"  # a readout cell in the wrapping column: its text in a div the frame caps at 100ch
texts = lambda recs: [r["text"] for r in recs]

# olens.json cards of agu18dec/mOre-lenses, verbatim: an inverter (scaled transform, continuation prompt), an RL lens on Gemma-4, and an AR, which has no read section
QWEN_INVERTER_CARD = {'repo': 'agu18dec/mOre-lenses', 'subject': 'qwen3.5-0.8b', 'base_model': 'Qwen/Qwen3.5-0.8B', 'role': 'inverter', 'run': 'ao.qwen3.5-0.8b.s0', 'label': 'inverter', 'adapter_subfolder': 'qwen3.5-0.8b/inverter/ao.qwen3.5-0.8b.s0/lora', 'source': 'olens-qwen3.5-0.8b-runs:ao_checkpoints/ao.qwen3.5-0.8b.s0/step6805/lora', 'workspacebench_tag': 'qwen3.5-0.8b.inv.end', 'read': {'transform': 'scaled', 'alpha': 16000.0, 'global_scale': 458.5681356729765, 'inject': 'h -> global_scale * h', 'prompt_kind': 'continuation_raw', 'prompt_template': 'An activation vector from layer {layer} of a language model is enclosed in activation tags: <activation>{char}</activation>. Produce the text that immediately follows this activation.', 'slot_char': '㈜', 'slot_id': 158983, 'layers': [8, 10, 12, 14, 16, 18, 20, 22], 'activation': 'output of decoder block L at one token = hidden_states[L + 1]', 'sampling_of_record': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 64, 'max_new_tokens': 256, 'k': 1}}}
GEMMA_RL_CARD = {'repo': 'agu18dec/mOre-lenses', 'subject': 'gemma-4-31b-it', 'base_model': 'google/gemma-4-31B-it', 'role': 'rl', 'run': 'rl.gemma4-31b.arwsd.ddp600.s0', 'label': 'RL i600 (fixed)', 'adapter_subfolder': 'gemma-4-31b-it/rl/rl.gemma4-31b.arwsd.ddp600.s0/iter_000600', 'source': 'olens-gemma-4-31b-it-runs:rl_runs/rl.gemma4-31b.arwsd.ddp600.s0/iter_000600', 'workspacebench_tag': 'gemma.gfix.arwsd.rl.i600', 'read': {'transform': 'unit', 'alpha': 16000.0, 'global_scale': None, 'inject': 'h -> alpha * h / |h|', 'prompt_kind': 'concepts_raw', 'prompt_template': "An activation vector from layer {layer} of a language model is enclosed in activation tags: <activation>{char}</activation>. Produce distinct concepts that encode this activation, each as a '- ' bullet on its own line.", 'slot_char': '㈜', 'slot_id': 246566, 'layers': [20, 24, 28, 32, 36, 40, 44, 48, 52, 56], 'activation': 'output of decoder block L at one token = hidden_states[L + 1]', 'sampling_of_record': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 64, 'max_new_tokens': 256, 'k': 1}}}
AR_CARD = {'repo': 'agu18dec/mOre-lenses', 'subject': 'qwen3.5-0.8b', 'base_model': 'Qwen/Qwen3.5-0.8B', 'role': 'ar', 'run': 'ar.qwen3.5-0.8b.ptag.s0', 'label': 'AR', 'adapter_subfolder': 'qwen3.5-0.8b/ar/ar.qwen3.5-0.8b.ptag.s0/lora', 'source': 'olens-qwen3.5-0.8b-runs:ml_checkpoints/ar.qwen3.5-0.8b.ptag.s0/end/lora', 'workspacebench_tag': None, 'ar': {'layers': [8, 10, 12, 14, 16, 18, 20, 22, 23], 'head_mode': 'prompt_tag', 'tag_template': '[Layer {n:02d}]', 'head_output': 'raw', 'loss_space': 'whiten', 'whitener_prefix': 'whitening_iolens_chat', 'note': "LoRA + linear per-layer heads (heads.pt) predicting the residual at each layer from a text span; the subject's whitener is not included."}}

@pytest.fixture(scope="module")
def tok():
    """TINY_MODEL's Llama-2 sentencepiece tokenizer with add_prefix_space=False, so that a letter is the same token alone and between the activation tags (with the dummy prefix, 'Z' alone is '▁Z', which single_token_marker's scan compares against the bare 'Z' of the render and never matches)."""
    load_tokenizer(TINY_MODEL)
    return AutoTokenizer.from_pretrained(TINY_MODEL, add_prefix_space=False)

@pytest.fixture(scope="module")
def contract(tok):
    """A contract for the tiny model: the marker from single_token_marker's scan over 'ZQXJ' of the real prompt render, layers 0 and 1, alpha 7, greedy sampling of 3 tokens."""
    render = lambda c: tok.apply_chat_template([{"role": "user", "content": OLENS_PROMPT.format(layer=0, char=c)}], tokenize=True, return_dict=False, add_generation_prompt=True)
    char, cid, ids, slot = single_token_marker(tok, render, "ZQXJ")
    assert (char, tok.decode(ids[slot - 1]), tok.decode(ids[slot + 1])) == ("Z", ">", "</")
    return {"base": TINY_MODEL, "layers": [0, 1], "prompt_kind": "concepts_raw", "transform": "unit", "alpha": 7.0, "global_scale": None, "marker": (char, cid, (ids[slot - 1], ids[slot + 1])), "sampling": dict(GREEDY)}

@pytest.fixture(scope="module")
def adapter_dir(tmp_path_factory, tok):
    """A random LoRA on the tiny model's q_proj and v_proj, saved with peft. lora_B is zero at init, so it is randomized, else the adapter would be the identity. `tok` is requested first so a missing cache skips the test instead of raising OSError here. save_embedding_layers=False, since peft's "auto" asks the Hub whether the base has a config.json."""
    t.manual_seed(0)
    pm = get_peft_model(AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    for name, p in pm.named_parameters():
        if "lora_B" in name:
            p.data.normal_()
    path = tmp_path_factory.mktemp("lora")
    pm.save_pretrained(path, save_embedding_layers=False)
    return str(path)

@pytest.fixture
def lens(adapter_dir, contract):
    """(peft, contract, bridge): the tiny model with the adapter loaded as 'olens' and active, booted into a Bridge around the same HF model."""
    hf = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32)
    peft, c = load_olens(hf, adapter_dir, contract=contract)
    return peft, c, boot_bridge(hf)

@pytest.fixture
def shown(monkeypatch):
    out = []
    monkeypatch.setattr(olens, "display", lambda x: out.append(x.data))
    return out

@pytest.fixture
def fake_generate(monkeypatch):
    """inject_generate replaced by a recorder returning a row with the text '- {i}a\\n- {i}b\\nnoise' for row i; `seen` holds the last call's arguments and, under "calls", every call's vecs."""
    seen = {"calls": []}
    def fake(model, tokenizer, ids, module, positions, vecs, write, max_new_tokens, seed, **kw):
        seen.update(model=model, tokenizer=tokenizer, ids=ids, module=module, positions=positions, vecs=vecs.clone(), write=write, max_new_tokens=max_new_tokens, seed=seed, kw=kw)
        seen["calls"].append(vecs.clone())
        return [{"text": f"- {i}a\n- {i}b\nnoise", "ids": [i], "stopped": True} for i in range(vecs.shape[0])]
    monkeypatch.setattr(olens, "inject_generate", fake)
    return seen

def test_contract_table():
    assert list(OLENS) == ["agu18dec/olens_and_ar:olens_s3d_rl600", "andyx10/oracle-lens-qwen3-4b"]
    for c in OLENS.values():
        assert set(c) == {"base", "layers", "prompt_kind", "transform", "alpha", "global_scale", "marker", "sampling"} and set(c["sampling"]) == {"do_sample", "temperature", "top_p", "top_k", "max_new_tokens"} and c["alpha"] == 16000.0 and c["sampling"]["do_sample"] is True
        assert (c["prompt_kind"], c["transform"], c["global_scale"]) == ("concepts_raw", "unit", None)
    agu, andy = OLENS.values()
    assert agu["base"] == "Qwen/Qwen3.6-27B" and agu["layers"] == list(range(20, 61, 4)) and agu["marker"] == ("㈜", 158983, (29, 510)) and agu["sampling"] == {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_new_tokens": 256}
    assert andy["base"] == "Qwen/Qwen3-4B" and andy["layers"] == list(range(12, 35, 2)) and andy["marker"] == ("㈎", 149705, (29, 522)) and andy["sampling"] == {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 128}
    assert OLENS_PROMPT.format(layer=44, char="㈜") == "An activation vector from layer 44 of a language model is enclosed in activation tags: <activation>㈜</activation>. Produce distinct concepts that encode this activation, each as a '- ' bullet on its own line."
    assert OLENS_PROMPT is PROMPTS["concepts_raw"] and PROMPTS["continuation_raw"].format(layer=20, char="㈜") == "An activation vector from layer 20 of a language model is enclosed in activation tags: <activation>㈜</activation>. Produce the text that immediately follows this activation."

def test_parse_bullets():
    assert parse_bullets("Concepts:\n- cats\n\n  - dogs and  \n-not a bullet\n- \n-\n--dash\nplain line\n- last") == ["cats", "dogs and", "last"]
    assert parse_bullets("") == [] and parse_bullets("- one") == ["one"] and parse_bullets("- a - b\n") == ["a - b"]

def test_card_contract():
    sampling = {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_new_tokens": 256}
    assert card_contract(QWEN_INVERTER_CARD) == {"base": "Qwen/Qwen3.5-0.8B", "layers": list(range(8, 23, 2)), "prompt_kind": "continuation_raw", "transform": "scaled", "alpha": 16000.0, "global_scale": 458.5681356729765, "marker": ("㈜", 158983, None), "sampling": sampling}
    assert card_contract(GEMMA_RL_CARD) == {"base": "google/gemma-4-31B-it", "layers": list(range(20, 57, 4)), "prompt_kind": "concepts_raw", "transform": "unit", "alpha": 16000.0, "global_scale": None, "marker": ("㈜", 246566, None), "sampling": sampling}
    assert QWEN_INVERTER_CARD["read"]["sampling_of_record"]["k"] == 1  # the card is not mutated
    with pytest.raises(ValueError, match=r"'ar.qwen3.5-0.8b.ptag.s0' \(ar\) has no read contract"):
        card_contract(AR_CARD)
    edit = lambda **kw: QWEN_INVERTER_CARD | {"read": QWEN_INVERTER_CARD["read"] | kw}
    with pytest.raises(ValueError, match=r"differ from the known ones by \['clip_mult'\]"):
        card_contract(edit(clip_mult=4.0))
    for kw, bad in [({"inject": "h -> clip(global_scale * h)"}, "transform"), ({"transform": "wht_unit"}, "transform"), ({"transform": "unit", "inject": "h -> alpha * h / |h|"}, "transform"), ({"global_scale": None}, "transform"), ({"prompt_template": OLENS_PROMPT}, "prompt"), ({"prompt_kind": "concepts_v2"}, "prompt"), ({"activation": "input of decoder block L"}, "activation"), ({"sampling_of_record": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_new_tokens": 256}}, "sampling")]:
        with pytest.raises(ValueError, match=rf"unknown \['{bad}'\] in the card's read contract"):
            card_contract(edit(**kw))

def test_load_olens_card_from_hub(tmp_path, monkeypatch):
    """A spec with no table entry and no contract= is a checkpoint directory: its card is fetched first, then the adapter at the card's adapter_subfolder of the same repo, both at the given revision."""
    calls = []
    (tmp_path / "olens.json").write_text(json.dumps(QWEN_INVERTER_CARD))
    monkeypatch.setattr(olens, "hf_hub_download", lambda *a, **kw: calls.append((a, kw)) or str(tmp_path / "olens.json"))
    monkeypatch.setattr(olens, "load_adapters", lambda model, adapters, **kw: calls.append((model, adapters, kw)) or "peft")
    spec = "agu18dec/mOre-lenses:qwen3.5-0.8b/inverter/ao.qwen3.5-0.8b.s0"
    peft, c = load_olens("hf", spec, revision="abc123")
    assert calls == [(("agu18dec/mOre-lenses", "olens.json"), {"subfolder": "qwen3.5-0.8b/inverter/ao.qwen3.5-0.8b.s0", "revision": "abc123"}), ("hf", {"olens": "agu18dec/mOre-lenses:qwen3.5-0.8b/inverter/ao.qwen3.5-0.8b.s0/lora"}, {"revision": "abc123"})]
    assert peft == "peft" and c == card_contract(QWEN_INVERTER_CARD) | {"adapter": "olens", "spec": spec}
    monkeypatch.setattr(olens, "hf_hub_download", lambda *a, **kw: str(tmp_path / "olens.json").replace("olens.json", "ar.json"))
    (tmp_path / "ar.json").write_text(json.dumps(AR_CARD))
    with pytest.raises(ValueError, match="has no read contract"):  # the card is read before any adapter is loaded
        load_olens("hf", "agu18dec/mOre-lenses:qwen3.5-0.8b/ar/ar.qwen3.5-0.8b.ptag.s0")
    assert len(calls) == 2

def test_load_olens_card_local(tmp_path, adapter_dir, contract, tok):
    """A local copy of a card repo, "root:dir": the card at root/dir/olens.json, the adapter at root/<adapter_subfolder>. The tiny model's card is the Gemma RL card with the tiny model's marker, layers and alpha."""
    char, cid, _ = contract["marker"]
    card = GEMMA_RL_CARD | {"base_model": TINY_MODEL, "adapter_subfolder": "tiny/rl/run/iter_000600", "read": GEMMA_RL_CARD["read"] | {"alpha": 7.0, "slot_char": char, "slot_id": cid, "layers": [0, 1]}}
    shutil.copytree(adapter_dir, tmp_path / "tiny/rl/run/iter_000600")
    (tmp_path / "tiny/rl/run/olens.json").write_text(json.dumps(card))
    spec = f"{tmp_path}:tiny/rl/run"
    peft, c = load_olens(AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32), spec)
    assert c == card_contract(card) | {"adapter": "olens", "spec": spec} and peft.active_adapters == ["olens"] and peft.peft_config["olens"].r == 4
    ids, slot = olens_prompt(tok, 1, c, chars="ZQXJ")  # a card names no neighbors: the scan's character and id are checked
    assert ids[slot] == cid
    with pytest.raises(ValueError, match=rf"contract's marker is 'Z' \(id {cid + 1}\) between any ids"):
        olens_prompt(tok, 1, c | {"marker": (char, cid + 1, None)}, chars="ZQXJ")
    with pytest.raises(FileNotFoundError, match="olens.json"):  # no card and no table entry: raises before any loading
        load_olens(object(), f"{tmp_path}:tiny/rl")

def test_load_olens_table_contract(monkeypatch):
    calls = []
    monkeypatch.setattr(olens, "load_adapters", lambda model, adapters, **kw: calls.append((model, adapters, kw)) or "peft")
    spec = "andyx10/oracle-lens-qwen3-4b"
    peft, c = load_olens("hf", spec, revision="main")
    assert peft == "peft" and calls == [("hf", {"olens": spec}, {"revision": "main"})]
    assert c == OLENS[spec] | {"adapter": "olens", "spec": spec} and c["layers"] is not OLENS[spec]["layers"] and "adapter" not in OLENS[spec]
    _, c2 = load_olens("hf", spec, name="lens2", contract={"layers": [3], "marker": ("x", 1, (2, 3))})
    assert c2 == {"layers": [3], "marker": ("x", 1, (2, 3)), "adapter": "lens2", "spec": spec} and calls[-1] == ("hf", {"lens2": spec}, {})

def test_load_olens(lens, adapter_dir, contract):
    peft, c, bridge = lens
    assert isinstance(peft, PeftModel) and peft.active_adapters == ["olens"] and peft.peft_config["olens"].r == 4 and not peft.training
    assert c == contract | {"adapter": "olens", "spec": adapter_dir} and "adapter" not in contract and c["layers"] is not contract["layers"] and c["sampling"] is not contract["sampling"]
    ids = t.tensor([[1, 5, 9, 13]])
    with peft.disable_adapter():
        base = peft(ids).logits
    assert not t.allclose(peft(ids).logits, base)  # the adapter changes the model
    hf = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32)
    peft2, c2 = load_olens(hf, adapter_dir, name="other", contract=contract)
    assert peft2.active_adapters == ["other"] and c2["adapter"] == "other"

def test_olens_prompt(tok, contract):
    char, cid, (left, right) = contract["marker"]
    ids, slot = olens_prompt(tok, 0, contract, chars="ZQXJ")
    assert ids[slot] == cid and (ids[slot - 1], ids[slot + 1]) == (left, right) and ids.count(cid) == 1 and ids[0] == tok.bos_token_id
    assert ids == tok.apply_chat_template([{"role": "user", "content": OLENS_PROMPT.format(layer=0, char="Z")}], tokenize=True, return_dict=False, add_generation_prompt=True)
    assert tok.decode(ids) == "<s>[INST] " + OLENS_PROMPT.format(layer=0, char="Z") + " [/INST]"
    ids1, slot1 = olens_prompt(tok, 1, contract, chars="ZQXJ")
    assert slot1 == slot and "layer 1 of" in tok.decode(ids1) and ids1 != ids
    with pytest.raises(ValueError, match=r"layer 5 is off-contract.*\[0, 1\]"):
        olens_prompt(tok, 5, contract, chars="ZQXJ")
    with pytest.raises(ValueError, match=rf"found 'Z' \(id {cid}\) between ids \({left}, {right}\), but the contract's marker is 'Z' \(id {cid + 1}\)"):
        olens_prompt(tok, 0, contract | {"marker": (char, cid + 1, (left, right))}, chars="ZQXJ")
    with pytest.raises(ValueError, match=rf"between \({left}, {right + 1}\)"):
        olens_prompt(tok, 0, contract | {"marker": (char, cid, (left, right + 1))}, chars="ZQXJ")
    with pytest.raises(ValueError, match="marker is 'Q'"):
        olens_prompt(tok, 0, contract | {"marker": ("Q", cid, (left, right))}, chars="ZQXJ")
    with pytest.raises(ValueError, match="found 'Q'"):  # the scan is over `chars`: Q is another token than the contract's Z
        olens_prompt(tok, 0, contract, chars="Q")
    with pytest.raises(ValueError, match="no character in the range"):  # nothing in U+3200..U+33FF is one token on this tokenizer
        olens_prompt(tok, 0, contract)

def test_olens_read_checks_the_adapter_before_generating(adapter_dir, contract, tok, monkeypatch, shown):
    monkeypatch.setattr(olens, "inject_generate", lambda *a, **kw: pytest.fail("generated with the wrong adapter state"))
    peft, c = load_olens(AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32), adapter_dir, contract=contract)
    peft = load_adapters(peft, {"other": adapter_dir})  # a second adapter goes on before any boot_bridge
    h = t.randn(D)
    peft.set_adapter("other")
    with pytest.raises(RuntimeError, match=r"active adapters \['other'\] are not \['olens'\]"):
        olens_read(peft, tok, h, 0, c, chars="ZQXJ")
    peft.base_model.set_adapter(["olens", "other"])
    with pytest.raises(RuntimeError, match=r"are not \['olens'\]"):
        olens_read(peft, tok, h, 0, c, chars="ZQXJ")
    peft.set_adapter("olens")
    with peft.disable_adapter():
        with pytest.raises(RuntimeError, match="adapters are disabled"):
            olens_read(peft, tok, h, 0, c, chars="ZQXJ")
        with pytest.raises(RuntimeError, match="adapters are disabled"):
            olens_readout({"blocks.0.hook_resid_post": t.randn(1, 4, D)}, [0], -1, peft, tok, c, chars="ZQXJ")
    assert shown == []
    with pytest.raises(ValueError, match="layer 3 is off-contract"):  # the prompt is checked before any generation too
        olens_read(peft, tok, h, 3, c, chars="ZQXJ")

def test_olens_read_plumbing(lens, tok, fake_generate):
    peft, c, _ = lens
    seen = fake_generate
    h = t.randn(2, D)
    out = olens_read(peft, tok, h, 1, c, n=3, seed=5, chars="ZQXJ", temperature=0.7, max_new_tokens=9)
    ids, slot = olens_prompt(tok, 1, c, chars="ZQXJ")
    assert seen["model"] is peft and seen["tokenizer"] is tok and seen["ids"] == ids and seen["positions"] == [slot] and seen["module"] is peft.get_input_embeddings()
    assert seen["vecs"].shape == (6, D) and t.equal(seen["vecs"], h.repeat_interleave(3, 0)) and t.equal(seen["vecs"][2], h[0]) and t.equal(seen["vecs"][3], h[1])  # activation-major, sample-minor
    assert seen["max_new_tokens"] == 9 and seen["seed"] == 5 and seen["kw"] == {"do_sample": False, "temperature": 0.7, "top_p": 1.0, "top_k": 0}
    assert c["sampling"] == GREEDY  # the contract is not mutated by the update
    orig, v = t.randn(6, 1, D), t.randn(6, 1, D) * 30
    w = seen["write"](orig, v)
    assert w.shape == (6, 1, D) and t.allclose(w, 7.0 * v / v.norm(dim=-1, keepdim=True), atol=1e-5) and t.allclose(w.norm(dim=-1), t.full((6, 1), 7.0), atol=1e-5)
    assert t.equal(seen["write"](t.zeros_like(orig), v), w) and t.equal(seen["write"](orig * 100, v), w)  # the original rows are ignored
    assert [list(r) for r in out] == [["layer", "i", "j", "h", "text", "ids", "stopped", "bullets"]] * 6  # one record per sample
    assert [(r["layer"], r["i"], r["j"], r["bullets"], r["ids"]) for r in out] == [(1, i, j, [f"{3 * i + j}a", f"{3 * i + j}b"], [3 * i + j]) for i in range(2) for j in range(3)]  # activation-major, the row's fields passed through
    assert out[5]["text"] == "- 5a\n- 5b\nnoise" and all(t.equal(r["h"], h[r["i"]]) and r["h"].device.type == "cpu" for r in out)
    one = olens_read(peft, tok, h[0], 0, c, n=2, chars="ZQXJ")
    assert texts(one) == ["- 0a\n- 0b\nnoise", "- 1a\n- 1b\nnoise"] and [(r["layer"], r["i"]) for r in one] == [(0, 0)] * 2 and seen["vecs"].shape == (2, D) and t.equal(seen["vecs"][1], h[0])  # [d] is activation 0
    assert seen["kw"] == {"do_sample": False, "temperature": 1.0, "top_p": 1.0, "top_k": 0} and seen["max_new_tokens"] == 3 and seen["seed"] == 0 and seen["ids"] == olens_prompt(tok, 0, c, chars="ZQXJ")[0]
    assert [r["bullets"] for r in olens_read(peft, tok, h[0], 0, c, chars="ZQXJ")] == [["0a", "0b"]] and seen["vecs"].shape == (1, D)
    half = olens_read(peft, tok, h[0].bfloat16(), 0, c, seed=None, chars="ZQXJ")
    assert seen["vecs"].dtype == t.bfloat16 and seen["seed"] is None and half[0]["h"].dtype == t.float32 and t.equal(half[0]["h"], h[0].bfloat16().float())  # the records hold float32
    assert [(r["i"], r["bullets"]) for r in olens_read(peft, tok, h[:1], 0, c, chars="ZQXJ")] == [(0, ["0a", "0b"])]  # [1, d] gives the records of [d]
    with pytest.raises(ValueError, match=r"\[d\] or \[m, d\], got \(2, 3, 16\)"):
        olens_read(peft, tok, t.randn(2, 3, D), 0, c, chars="ZQXJ")
    with pytest.raises(ValueError, match="n must be at least 1, got 0"):
        olens_read(peft, tok, h[0], 0, c, n=0, chars="ZQXJ")

def test_olens_read_end_to_end(lens, tok):
    peft, c, _ = lens
    h = t.randn(2, D)
    seen = []
    handle = decoder_layers(peft)[0].register_forward_pre_hook(lambda m, args, kwargs: seen.append(args[0].clone()), with_kwargs=True)
    out = olens_read(peft, tok, h, 1, c, n=3, chars="ZQXJ", max_new_tokens=1)
    handle.remove()
    ids, slot = olens_prompt(tok, 1, c, chars="ZQXJ")
    x = seen[0]  # what block 0 saw on the prefill
    assert x.shape == (6, len(ids), D) and t.allclose(x[:, slot], 7.0 * normed(h).repeat_interleave(3, 0), atol=1e-5)  # alpha * unit(h), activation-major
    emb = peft.get_input_embeddings()(t.tensor([ids]))[0]
    assert t.allclose(x[:, :slot], emb[:slot].expand(6, -1, -1), atol=1e-6) and t.allclose(x[:, slot + 1:], emb[slot + 1:].expand(6, -1, -1), atol=1e-6)  # every other row is the prompt's embedding
    assert len(out) == 6 and all(isinstance(r["text"], str) and r["bullets"] == parse_bullets(r["text"]) and r["text"] == tok.decode(r["ids"], skip_special_tokens=True) and len(r["ids"]) <= 1 for r in out)
    sample = dict(n=2, chars="ZQXJ", do_sample=True, top_p=0.95, top_k=64)
    a = texts(olens_read(peft, tok, h, 1, c, **sample))
    assert a == texts(olens_read(peft, tok, h, 1, c, **sample)) and len(a) == 4  # deterministic given (activations, n, seed): the same batch reproduces its block
    assert texts(olens_read(peft, tok, h[:1], 1, c, **sample)) == texts(olens_read(peft, tok, h[0], 1, c, **sample))  # [1, d] and [d] are the same batch
    one = olens_read(peft, tok, h[0], 0, c, chars="ZQXJ")
    assert len(one) == 1 and isinstance(one[0]["bullets"], list) and all(isinstance(s, str) for s in one[0]["bullets"])

def test_olens_read_inverter(lens, tok, fake_generate, shown):
    """An inverter contract: the continuation prompt, global_scale * h written (the norm kept), no bullets, and a readout of its text a line per row."""
    peft, c, _ = lens
    inv = c | {"prompt_kind": "continuation_raw", "transform": "scaled", "global_scale": 3.0}
    out = olens_read(peft, tok, t.randn(2, D), 1, inv, n=2, chars="ZQXJ")
    ids, slot = olens_prompt(tok, 1, inv, chars="ZQXJ")
    assert fake_generate["ids"] == ids and fake_generate["positions"] == [slot] and tok.decode(ids) == "<s>[INST] " + PROMPTS["continuation_raw"].format(layer=1, char="Z") + " [/INST]"
    v = t.randn(4, 1, D)
    assert t.equal(fake_generate["write"](t.randn(4, 1, D), v), 3.0 * v)
    assert [(r["i"], r["j"], r["bullets"]) for r in out] == [(i, j, None) for i in range(2) for j in range(2)] and out[0]["text"] == "- 0a\n- 0b\nnoise"
    recs = olens_readout({"blocks.1.hook_resid_post": t.randn(1, 4, D)}, [1], -1, peft, tok, inv, chars="ZQXJ")
    assert recs[0]["bullets"] is None and td("- 0a") in shown[-1] and td("noise") in shown[-1] and shown[-1].count("<tr") == 4

def test_olens_readout(lens, tok, fake_generate, shown, monkeypatch):
    peft, c, _ = lens
    seen, vecs = fake_generate, fake_generate["calls"]
    toks = tok.encode("Hello there my old friend")
    cache = {f"blocks.{l}.hook_resid_post": t.randn(1, len(toks), D) for l in range(2)} | {"blocks.0.hook_resid_pre": t.randn(1, len(toks), D)}
    out = olens_readout(cache, [0, 1], [2, -1], peft, tok, c, n=2, seed=7, input_src=toks, chars="ZQXJ")
    assert [(r["layer"], r["pos"], r["i"], r["j"], r["bullets"]) for r in out] == [(l, p, i, j, [f"{2 * i + j}a", f"{2 * i + j}b"]) for l in (0, 1) for i, p in enumerate((2, -1)) for j in range(2)]  # olens_read's records with the position as given, layer-major
    assert all(t.equal(r["h"], cache[f"blocks.{r['layer']}.hook_resid_post"][0, r["pos"]]) for r in out)
    assert len(vecs) == 2 and all(v.shape == (4, D) for v in vecs) and all(t.equal(v, cache[f"blocks.{l}.hook_resid_post"][0, [2, -1]].repeat_interleave(2, 0)) for l, v in enumerate(vecs))  # one batched read per layer, positions in order
    assert seen["ids"] == olens_prompt(tok, 1, c, chars="ZQXJ")[0] and seen["max_new_tokens"] == 3 and seen["seed"] == 7
    h = shown[-1]
    d = widget_data(h)
    assert d["tabs"] == ["L0", "L1"] and d["inner"] == ["p2", "p-1"] and "<button class=on>L0</button><button>L1</button>" in h and "<button class=on>p2</button><button>p-1</button>" in h and h.count("class='tb' hidden") == 0
    assert h.count("<span data-p=") == 2 and f"<span data-p=1 data-h='pos {len(toks) - 1}" in h and "data-h='pos 2 " in h  # the strip marks both positions
    assert d["panes"][0][1] == [["sample 0", [["2a"], ["2b"]], None, ["w"]], ["sample 1", [["3a"], ["3b"]], None, ["w"]]] and d["panes"][0][0] is None and all(len(pane) == 2 for row in d["panes"] for pane in row if pane)  # a table per sample, a bullet per row, in every pane; the first pane rendered, not shipped
    assert h.count("<table>") == 2 and "<th colspan=1>sample 0</th>" in h and "<th colspan=1>sample 1</th>" in h and td("0a") in h and td("1b") in h and "3b" not in first_pane(h) and "noise" not in h and "oracle lens readout</h3>" in h and "--n:2;" in h  # the first pane (L0, p2) rendered
    olens_readout(cache, [0], 2, peft, tok, c, chars="ZQXJ", do_sample=False, max_new_tokens=5)
    assert seen["kw"]["do_sample"] is False and seen["max_new_tokens"] == 5  # sampling kwargs override the contract's through olens_read
    out = olens_readout(cache, [1], 3, peft, tok, c, title="T", input_src=toks, chars="ZQXJ")
    h = shown[-1]
    assert [(r["layer"], r["pos"], r["bullets"]) for r in out] == [(1, 3, ["0a", "0b"])] and "application/json" not in h and first_pane(h) == readout_grid([("sample 0", [["0a"], ["0b"]], None, ["w"])]) and h.count("class='tb' hidden") == 2 and "<span data-p=" not in h and h.count("outline:1px solid #fc6") == 1 and "T</h3>" in h and "--n:1;" in h and seen["seed"] == 0
    out = olens_readout(cache, [0], [1], peft, tok, c, raw=True, hook="hook_resid_pre", chars="ZQXJ")
    assert [(r["layer"], r["pos"], r["text"]) for r in out] == [(0, 1, "- 0a\n- 0b\nnoise")] and t.equal(vecs[-1], cache["blocks.0.hook_resid_pre"][0, [1]]) and td("- 0a") in shown[-1] and td("noise") in shown[-1] and shown[-1].count("<tr") == 4 and "class='tk'" not in shown[-1]  # raw: a line per row; no input_src, no strip
    n_calls = len(vecs)
    with pytest.raises(ValueError, match=r"repeated positions: \[2, 2\]"):
        olens_readout(cache, [0], [2, 2], peft, tok, c, chars="ZQXJ")
    with pytest.raises(ValueError, match="input_src is empty"):  # input_src is resolved before any generation
        olens_readout(cache, [0], 0, peft, tok, c, input_src=[], chars="ZQXJ")
    assert len(vecs) == n_calls and len(shown) == 4
    monkeypatch.setattr(olens, "inject_generate", lambda *a, **kw: [{"text": "- a<b & c", "ids": [], "stopped": True}])
    olens_readout(cache, [0], 0, peft, tok, c, chars="ZQXJ")
    assert td("a&lt;b &amp; c") in shown[-1] and "application/json" not in shown[-1]  # escaped in the rendered pane; one pane, so no payload
    with pytest.raises(KeyError, match="blocks.5.hook_resid_post"):
        olens_readout(cache, [5], 0, peft, tok, c, chars="ZQXJ")
    assert len(shown) == 5

def test_olens_readout_end_to_end(lens, tok, shown):
    peft, c, bridge = lens
    ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}], tokenize=True, return_dict=False, add_generation_prompt=True)
    with peft.disable_adapter():
        _, cache = bridge.run_with_cache(t.tensor([ids]))
        hs = peft(t.tensor([ids]), output_hidden_states=True).hidden_states
        with pytest.raises(RuntimeError, match="adapters are disabled"):  # the booted path: the LoRA Linear is a LinearBridge's original_component
            olens_read(peft, tok, cache["blocks.0.hook_resid_post"][0, 0], 0, c, chars="ZQXJ")
    assert t.allclose(cache["blocks.0.hook_resid_post"], hs[1], atol=1e-5) and len(hs) == 3  # layer L is hidden_states[L + 1]; the last entry has the final norm applied
    _, cache_on = bridge.run_with_cache(t.tensor([ids]))
    assert not t.allclose(cache_on["blocks.1.hook_resid_post"], cache["blocks.1.hook_resid_post"])  # the lens's own activations differ from the base model's
    out = olens_readout(cache, [0, 1], [0, -1], peft, tok, c, n=2, input_src=ids, chars="ZQXJ")
    assert [(r["layer"], r["pos"], r["j"]) for r in out] == [(l, p, j) for l in (0, 1) for p in (0, -1) for j in range(2)] and all(isinstance(r["bullets"], list) for r in out) and t.equal(out[-1]["h"], cache["blocks.1.hook_resid_post"][0, -1])
    assert [len(row) for row in widget_data(shown[-1])["panes"]] == [2, 2] and shown[-1].count("<span data-p=") == 2 and peft.active_adapters == ["olens"]

@pytest.mark.hf
@pytest.mark.parametrize("name, spec, layer, bad", [("Qwen/Qwen3-0.6B", "andyx10/oracle-lens-qwen3-4b", 12, 13), ("Qwen/Qwen3.6-27B", "agu18dec/olens_and_ar:olens_s3d_rl600", 44, 43)])
def test_checkpoint_contracts(name, spec, layer, bad):
    """The marker scan on the base model's tokenizer lands on the card's character, id and neighbors, with the 56-token render; the other checkpoint's contract raises on this tokenizer."""
    tok = load_tokenizer(name)
    c = OLENS[spec]
    char, cid, (left, right) = c["marker"]
    ids, slot = olens_prompt(tok, layer, c)
    assert len(ids) == 56 and ids[slot] == cid and (ids[slot - 1], ids[slot + 1]) == (left, right) and tok.decode(cid) == char and tok.decode(left) == ">" and tok.decode(right) == "</"
    text = tok.decode(ids)
    assert text == f"<|im_start|>user\n{OLENS_PROMPT.format(layer=layer, char=char)}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" and text.count(char) == 1
    assert all(olens_prompt(tok, l, c)[1] == slot for l in c["layers"])
    with pytest.raises(ValueError, match=f"layer {bad} is off-contract"):
        olens_prompt(tok, bad, c)
    other = next(s for s in OLENS if s != spec)
    with pytest.raises(ValueError, match="contract's marker"):
        olens_prompt(tok, OLENS[other]["layers"][0], OLENS[other])

@pytest.mark.hf
@pytest.mark.parametrize("name, card, neighbors, template", [
    ("Qwen/Qwen3.5-0.8B", QWEN_INVERTER_CARD, (29, 510), "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("google/gemma-4-31B-it", GEMMA_RL_CARD, (236813, 954), "<bos><|turn>user\n{}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
])
def test_card_markers(name, card, neighbors, template):
    """On the subject's tokenizer the marker scan lands on the card's character and id, between '>' and '</' (pinned here, since cards do not name the neighbors), at every layer and with both prompts, in the chat template's render with thinking off."""
    tok = load_tokenizer(name)
    c = card_contract(card)
    for kind, prompt in PROMPTS.items():
        for layer in c["layers"]:
            ids, slot = olens_prompt(tok, layer, c | {"prompt_kind": kind})
            assert ids[slot] == c["marker"][1] and (ids[slot - 1], ids[slot + 1]) == neighbors and ids.count(ids[slot]) == 1
        assert tok.decode(ids) == template.format(prompt.format(layer=layer, char="㈜"))
    assert tok.decode(list(neighbors)) == "></"

@pytest.fixture(scope="module")
def tiny_gemma4(tmp_path_factory):
    """A random 2-block Gemma4ForConditionalGeneration (the 31B's class: d 16, a 1-block vision tower, the real 262k vocabulary, no per-layer inputs) saved with the gemma-4-31B-it tokenizer, which boot_bridge loads from the model's path."""
    tok = load_tokenizer("google/gemma-4-31B-it")
    text = dict(vocab_size=262144, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8, global_head_dim=8, num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=8, layer_types=["sliding_attention", "full_attention"], hidden_size_per_layer_input=0, max_position_embeddings=128, rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"}, "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}})
    vision = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8, global_head_dim=8)
    t.manual_seed(0)
    path = tmp_path_factory.mktemp("gemma4")
    Gemma4ForConditionalGeneration(Gemma4Config(text_config=text, vision_config=vision, eos_token_id=[1, 106])).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path), tok

@pytest.mark.hf
def test_gemma4_card_lens(tiny_gemma4, tmp_path):
    """A card lens on Gemma-4's multimodal class: a LoRA with the released adapters' language_model keys loads through the card, block L's output in the Bridge is hidden_states[L + 1], and the write lands in block 0's input as alpha * unit(h): it replaces the embedding module's output, which already holds the sqrt(d) scale, and is not scaled again."""
    path, tok = tiny_gemma4
    t.manual_seed(0)
    pm = get_peft_model(Gemma4ForConditionalGeneration.from_pretrained(path, dtype=t.float32), LoraConfig(r=4, lora_alpha=8, target_modules=r".*language_model.*\.(q_proj|v_proj)"))  # the vision tower's projections are not nn.Linear, and the released adapters list language-model modules only
    for name, p in pm.named_parameters():
        if "lora_B" in name:
            p.data.normal_()
    pm.save_pretrained(tmp_path / "gemma/rl/run/iter_000600", save_embedding_layers=False)
    card = GEMMA_RL_CARD | {"base_model": path, "adapter_subfolder": "gemma/rl/run/iter_000600", "read": GEMMA_RL_CARD["read"] | {"alpha": 7.0, "layers": [0, 1]}}
    (tmp_path / "gemma/rl/run/olens.json").write_text(json.dumps(card))
    hf = Gemma4ForConditionalGeneration.from_pretrained(path, dtype=t.float32)
    peft, c = load_olens(hf, f"{tmp_path}:gemma/rl/run")
    assert next(k for k in peft.state_dict() if "lora_A" in k).startswith("base_model.model.model.language_model.layers.0.")
    bridge = boot_bridge(hf)
    ids = to_ids([{"role": "user", "content": "What is the capital of France?"}], tok, add_generation_prompt=True, enable_thinking=False)
    with peft.disable_adapter():
        _, cache = bridge.run_with_cache(t.tensor([ids]))
        hs = peft(t.tensor([ids]), output_hidden_states=True).hidden_states
    assert t.allclose(cache["blocks.0.hook_resid_post"], hs[1], atol=1e-5)
    seen = []
    handle = decoder_layers(peft)[0].register_forward_pre_hook(lambda m, args, kwargs: seen.append(args[0].clone()), with_kwargs=True)
    h = cache["blocks.0.hook_resid_post"][0, -1]
    out = olens_read(peft, tok, h, 0, c, n=2, max_new_tokens=2)
    handle.remove()
    pids, slot = olens_prompt(tok, 0, c)
    emb = peft.get_input_embeddings()
    x = seen[0]
    assert x.shape == (2, len(pids), 16) and t.allclose(x[:, slot], 7.0 * normed(h).expand(2, -1), atol=1e-5)
    assert t.allclose(x[0, slot - 1], emb.weight[pids[slot - 1]] * 4.0, atol=1e-5)  # every other row is the embedding times sqrt(16)
    assert len(out) == 2 and all(isinstance(r["bullets"], list) and len(r["ids"]) <= 2 for r in out)
