import json

import pytest
import torch as t
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

from conftest import TINY_MODEL, load_tokenizer
from mechtools import ao
from mechtools.ao import *
from mechtools.hooks import decoder_layers, inject_generate
from mechtools.models import boot_bridge, load_adapters
from mechtools.tokens import to_ids

AO_CONFIG = {"special_token": " ?", "hook_onto_layer": 1, "steering_coefficient": 1.0, "act_layer_combinations": [[0], [1]], "layer_combinations": [[0], [50]], "model_name": TINY_MODEL, "generation_kwargs": {"do_sample": False, "max_new_tokens": 20}}
CONFIG = {**AO_CONFIG, "adapter": "ao", "op": "add"}  # what load_ao returns for AO_CONFIG
SPECIAL_ID = 1577  # " ?" on the tiny Llama-2 tokenizer
D = 16

def tiny_hf():
    try:
        return AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=t.float32)
    except OSError as e:
        pytest.skip(f"{TINY_MODEL} is not in the local HF cache ({type(e).__name__})")

@pytest.fixture(scope="module")
def ao_repo(tmp_path_factory) -> str:
    """A local oracle repo: a random LoRA on TINY_MODEL (lora_B is zero at init and an untouched adapter is the identity, so it is drawn random) and an ao_config.json with the real checkpoints' keys. Seeded before get_peft_model, which draws lora_A from the global RNG, so the adapter is the same every run."""
    t.manual_seed(0)
    peft_model = get_peft_model(tiny_hf(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            p.data.normal_()
    path = tmp_path_factory.mktemp("ao_repo")
    peft_model.save_pretrained(path, save_embedding_layers=False)  # the "auto" default asks the Hub whether the base repo has a config.json
    (path / "ao_config.json").write_text(json.dumps(AO_CONFIG))
    return str(path)

@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(TINY_MODEL)

@pytest.fixture(scope="module")
def oracle(ao_repo):
    """(PeftModel, config) from load_ao, not booted, with a second copy of the adapter loaded as "other" and left inactive for the wrong-adapter tests."""
    model, config = load_ao(tiny_hf(), ao_repo)
    load_adapters(model, {"other": ao_repo})
    model.set_adapter("ao")
    return model, config

@pytest.fixture(scope="module")
def booted(ao_repo):
    """(PeftModel, config, Bridge): the real stack, the oracle adapter loaded and then the HF model booted into a Bridge, so generation runs through the Bridge's wrappers and the cache comes from run_with_cache."""
    hf = tiny_hf()
    model, config = load_ao(hf, ao_repo)
    return model, config, boot_bridge(hf)

@pytest.fixture
def shown(monkeypatch):
    out = []
    monkeypatch.setattr(ao, "display", lambda x: out.append(x.data))
    return out

@pytest.fixture
def no_generate(monkeypatch):
    monkeypatch.setattr(ao, "inject_generate", lambda *a, **k: pytest.fail("inject_generate was called"))

@pytest.fixture
def captured(monkeypatch):
    """inject_generate replaced by a recorder of every call's arguments, answering one numbered string per row of vecs."""
    calls = []
    def fake(model, tokenizer, ids, module, positions, vecs, write, max_new_tokens, seed, **kw):
        calls.append(dict(model=model, tokenizer=tokenizer, ids=ids, module=module, positions=positions, vecs=vecs, write=write, max_new_tokens=max_new_tokens, seed=seed, kw=kw))
        return [f"answer {len(calls)} row {i}" for i in range(vecs.shape[0])]
    monkeypatch.setattr(ao, "inject_generate", fake)
    return calls

def test_load_ao_config(tmp_path, monkeypatch):
    (tmp_path / "ao_config.json").write_text(json.dumps(AO_CONFIG))
    assert load_ao_config(str(tmp_path)) == AO_CONFIG
    seen = []
    monkeypatch.setattr(ao, "hf_hub_download", lambda repo, filename, **kw: seen.append((repo, filename, kw)) or str(tmp_path / "ao_config.json"))
    assert load_ao_config("org/oracle") == AO_CONFIG and load_ao_config("org/base:oracle") == AO_CONFIG and load_ao_config("org/oracle", revision="v2", token="tk") == AO_CONFIG
    assert seen == [("org/oracle", "ao_config.json", {}), ("org/base", "ao_config.json", {"subfolder": "oracle"}), ("org/oracle", "ao_config.json", {"revision": "v2", "token": "tk"})]
    (tmp_path / "nocfg").mkdir()
    with pytest.raises(FileNotFoundError):
        load_ao_config(str(tmp_path / "nocfg"))

def test_load_ao_passes_kwargs(tmp_path, monkeypatch):
    """Every kwarg goes to load_adapters; revision and token also reach the config download, so a pinned adapter gets its own config."""
    (tmp_path / "ao_config.json").write_text(json.dumps(AO_CONFIG))
    seen = []
    monkeypatch.setattr(ao, "load_adapters", lambda model, adapters, **kw: seen.append(("adapters", model, adapters, kw)) or "peft")
    monkeypatch.setattr(ao, "hf_hub_download", lambda repo, filename, **kw: seen.append(("hub", repo, filename, kw)) or str(tmp_path / "ao_config.json"))
    assert load_ao("hf", "org/oracle", revision="v2", token="tk", autocast_adapter_dtype=False) == ("peft", CONFIG)
    assert seen == [("adapters", "hf", {"ao": "org/oracle"}, {"revision": "v2", "token": "tk", "autocast_adapter_dtype": False}), ("hub", "org/oracle", "ao_config.json", {"revision": "v2", "token": "tk"})]
    assert load_ao("hf", "org/oracle", name="x", config=AO_CONFIG) == ("peft", {**CONFIG, "adapter": "x"}) and len(seen) == 3  # a given config is not downloaded

def test_ao_prompt_with_a_fake_tokenizer():
    """The render goes through the chat template as one user turn with the generation prompt and thinking off; the count and the consecutive checks raise, and so does a multi-token placeholder."""
    seen = []
    class Tok:
        def encode(self, s, add_special_tokens=True): return [7] if s == "@" else [1, 2]
        def apply_chat_template(self, conv, **kw): seen.append((conv, kw)); return [7, 0, 7]
    with pytest.raises(ValueError, match=r"need 2 consecutive placeholder tokens \(id 7, '@'\).*found 2 at \[0, 2\]"):
        ao_prompt(Tok(), {"special_token": "@"}, 3, 2, "q")
    assert seen == [([{"role": "user", "content": "Layer: 3\n@@ \nq"}], {"tokenize": True, "return_dict": False, "add_generation_prompt": True, "enable_thinking": False})]
    with pytest.raises(ValueError, match="'ab' is 2 tokens"):
        ao_prompt(Tok(), {"special_token": "ab"}, 3, 1, "q")
    with pytest.raises(ValueError, match=r"need 3 .* found 2"):
        ao_prompt(Tok(), {"special_token": "@"}, 3, 3, "q")
    with pytest.raises(ValueError, match="k must be at least 1, got 0"):
        ao_prompt(Tok(), {"special_token": "@"}, 3, 0, "q")
    assert len(seen) == 2  # k < 1 raises before rendering

@pytest.mark.hf
def test_load_ao(ao_repo, tok, captured):
    hf = tiny_hf()
    model, config = load_ao(hf, ao_repo)
    assert isinstance(model, PeftModel) and list(model.peft_config) == ["ao"] and model.active_adapters == ["ao"]
    assert config == CONFIG and decoder_layers(model)[1] is hf.model.layers[1]  # peft wraps the HF model in place, so its blocks are the ones to hook
    hs = t.randn(1, 5, D)
    with model.disable_adapter():
        off = decoder_layers(model)[0].self_attn.q_proj(hs)
    assert not t.allclose(decoder_layers(model)[0].self_attn.q_proj(hs), off)  # the random adapter does something
    given = {**AO_CONFIG, "op": "replace"}
    model2, config2 = load_ao(model, ao_repo, name="old", config=given)
    assert model2 is model and sorted(model2.peft_config) == ["ao", "old"] and config2 == {**AO_CONFIG, "op": "replace", "adapter": "old"}
    assert given == {**AO_CONFIG, "op": "replace"}  # the given config is copied, not modified
    assert model2.active_adapters == ["ao"]  # a load onto a PeftModel leaves the active adapter alone
    with pytest.raises(RuntimeError, match=r"active adapters \['ao'\] are not \['old'\]"):
        ao_read(model2, tok, t.randn(D), 0, "q", config2)
    model2.set_adapter("old")
    assert ao_read(model2, tok, t.randn(D), 0, "q", config2) == ["answer 1 row 0"] and model2.active_adapters == ["old"]
    with pytest.raises(ValueError, match="already loaded"):
        load_ao(model, ao_repo)

@pytest.mark.hf
@pytest.mark.parametrize("k", [1, 3])
def test_ao_prompt_finds_consecutive_placeholders(tok, k):
    ids, positions = ao_prompt(tok, CONFIG, 7, k, "What is the model thinking about?")
    assert positions == list(range(positions[0], positions[0] + k)) and all(ids[p] == SPECIAL_ID for p in positions) and ids.count(SPECIAL_ID) == k
    text = tok.decode(ids)
    assert "Layer: 7\n" + " ?" * k + " \nWhat is the model thinking about?" in text and text.startswith("<s>[INST]") and text.endswith("[/INST]")
    assert ids[0] == tok.bos_token_id and [ids[p - 1] for p in positions] == [13] + [SPECIAL_ID] * (k - 1) and tok.decode(13) == "\n"  # the first placeholder follows the newline of the Layer line

@pytest.mark.hf
def test_ao_prompt_raises(tok):
    with pytest.raises(ValueError, match=r"'abcdef' is \d+ tokens"):
        ao_prompt(tok, {**CONFIG, "special_token": "abcdef"}, 1, 1, "q")
    with pytest.raises(ValueError, match=r"need 2 consecutive placeholder tokens \(id 1577, ' \?'\).*found 3 at \[\d+, \d+, \d+\]"):
        ao_prompt(tok, CONFIG, 1, 2, "Is this right ? Say yes")  # " ?" inside the question is one more placeholder
    with pytest.raises(ValueError, match="k must be at least 1, got 0"):
        ao_prompt(tok, CONFIG, 1, 0, "q")
    with pytest.raises(ValueError, match="k must be at least 1, got -2"):
        ao_prompt(tok, CONFIG, 1, -2, "q")

@pytest.mark.hf
def test_ao_read_checks_the_adapter_before_generating(oracle, tok, no_generate):
    model, config = oracle
    hs = t.randn(D)
    with model.disable_adapter():
        with pytest.raises(RuntimeError, match="disabled"):
            ao_read(model, tok, hs, 0, "q", config)
    model.set_adapter("other")
    try:
        with pytest.raises(RuntimeError, match=r"active adapters \['other'\] are not \['ao'\]"):
            ao_read(model, tok, hs, 0, "q", config)
    finally:
        model.set_adapter("ao")
    with pytest.raises(RuntimeError, match=r"are not \['nope'\]"):
        ao_read(model, tok, hs, 0, "q", {**config, "adapter": "nope"})
    model.base_model.set_adapter(["ao", "other"])
    try:
        with pytest.raises(RuntimeError, match="are not"):  # a stack holding the oracle plus another adapter is not the oracle
            ao_read(model, tok, hs, 0, "q", config)
    finally:
        model.set_adapter("ao")

@pytest.mark.hf
def test_ao_read_rejects_bad_op_and_shape(oracle, tok, no_generate):
    model, config = oracle
    with pytest.raises(ValueError, match="'scale', not 'add' or 'replace'"):
        ao_read(model, tok, t.randn(D), 0, "q", {**config, "op": "scale"})
    with pytest.raises(ValueError, match=r"got \(2, 1, 16\)"):
        ao_read(model, tok, t.randn(2, 1, D), 0, "q", config)
    with pytest.raises(ValueError, match="found 2"):
        ao_read(model, tok, t.randn(D), 0, "q ?", config)

@pytest.mark.hf
def test_ao_read_plumbing(oracle, tok, captured):
    model, config = oracle
    hs = t.randn(3, D)
    out = ao_read(model, tok, hs, 1, "Which language?", config, n=4, max_new_tokens=7, seed=3)
    assert out == [f"answer 1 row {i}" for i in range(4)]
    call, = captured
    ids, positions = ao_prompt(tok, config, 1, 3, "Which language?")
    assert call["model"] is model and call["tokenizer"] is tok and call["ids"] == ids and call["positions"] == positions and len(positions) == 3
    assert call["module"] is decoder_layers(model)[1] and call["module"] is model.base_model.model.model.layers[1]
    assert call["vecs"].shape == (4, 3, D) and all(t.equal(row, hs) for row in call["vecs"])
    assert call["max_new_tokens"] == 7 and call["seed"] == 3 and call["kw"] == {"do_sample": False}
    assert "Layer: 1\n" in tok.decode(ids)
    orig, v = t.randn(4, 3, D) * 5, t.randn(4, 3, D)
    unit = v / v.norm(dim=-1, keepdim=True)
    assert t.allclose(call["write"](orig, v), orig + orig.norm(dim=-1, keepdim=True) * unit, atol=1e-5)
    assert t.allclose(call["write"](orig, 3 * v), call["write"](orig, v), atol=1e-5)  # only the direction of v matters
    ao_read(model, tok, hs, 1, "q", {**config, "steering_coefficient": 2.0})
    assert t.allclose(captured[-1]["write"](orig, v), orig + 2.0 * orig.norm(dim=-1, keepdim=True) * unit, atol=1e-5)
    ao_read(model, tok, hs, 1, "q", {**config, "op": "replace"})
    assert t.allclose(captured[-1]["write"](orig, v), orig.norm(dim=-1, keepdim=True) * unit, atol=1e-5)
    ao_read(model, tok, hs, 1, "q", {**config, "op": "replace", "steering_coefficient": 2.0})
    written = captured[-1]["write"](orig, v)
    assert t.allclose(written, 2.0 * orig.norm(dim=-1, keepdim=True) * unit, atol=1e-5) and t.allclose(written.norm(dim=-1), 2.0 * orig.norm(dim=-1), atol=1e-4)
    ao_read(model, tok, hs[0], 0, "q", {**config, "hook_onto_layer": 0})  # [d] is one activation; the injection block comes from the config
    call = captured[-1]
    assert call["vecs"].shape == (1, 1, D) and t.equal(call["vecs"][0, 0], hs[0]) and len(call["positions"]) == 1 and call["module"] is decoder_layers(model)[0] and "Layer: 0\n" in tok.decode(call["ids"])
    ao_read(model, tok, hs, 1, "q", config, do_sample=True, temperature=0.7, top_p=0.9)
    assert captured[-1]["kw"] == {"do_sample": True, "temperature": 0.7, "top_p": 0.9} and captured[-1]["max_new_tokens"] == 50 and captured[-1]["seed"] == 0
    ao_read(model, tok, hs, 1, "q", config, seed=None)
    assert captured[-1]["seed"] is None

@pytest.mark.hf
def test_ao_read_generates(booted):
    model, config, bridge = booted
    tok = bridge.tokenizer
    hs = t.randn(2, D)
    out = ao_read(model, tok, hs, 1, "What is here?", config, n=3, max_new_tokens=3)
    assert len(out) == 3 and all(isinstance(s, str) for s in out) and len(set(out)) == 1  # greedy rows of one prompt agree
    assert out[:1] == ao_read(model, tok, hs, 1, "What is here?", config, max_new_tokens=3)
    assert not decoder_layers(model)[1]._forward_hooks  # the injection hook is gone
    sampled = ao_read(model, tok, hs, 1, "What is here?", config, n=2, max_new_tokens=4, seed=1, do_sample=True, temperature=1.0, top_k=0)
    assert len(sampled) == 2 and sampled == ao_read(model, tok, hs, 1, "What is here?", config, n=2, max_new_tokens=4, seed=1, do_sample=True, temperature=1.0, top_k=0)
    # The injection reaches the generated text only when a block follows the injection block: block 1 is this 2-layer model's last, so a write at a placeholder position of its output never reaches the last position's logits, whatever the op or coefficient. Injected at block 0's output, a random vector changes the greedy continuation.
    ids, positions = ao_prompt(tok, config, 0, 1, "q")
    none = inject_generate(model, tok, ids, decoder_layers(model)[0], positions, t.zeros(1, D), lambda orig, vecs: orig, 8, 0, do_sample=False)
    g = t.Generator().manual_seed(0)
    assert any(ao_read(model, tok, t.randn(D, generator=g), 0, "q", {**config, "hook_onto_layer": 0}, max_new_tokens=8) != none for _ in range(8))  # the default add at coefficient 1.0
    assert any(ao_read(model, tok, t.randn(D, generator=g), 0, "q", {**config, "hook_onto_layer": 0, "op": "replace"}, max_new_tokens=8) != none for _ in range(8))
    assert all(ao_read(model, tok, t.randn(D, generator=g), 0, "q", {**config, "op": "replace", "steering_coefficient": 50.0}, max_new_tokens=8) == none for _ in range(4))

@pytest.mark.hf
def test_ao_readout_plumbing(oracle, tok, captured, shown):
    model, config = oracle
    t.manual_seed(0)
    cache = {f"blocks.{l}.hook_resid_post": t.randn(1, 9, D) for l in range(2)} | {"blocks.0.hook_resid_pre": t.randn(1, 9, D)}
    questions = ["What topic?", "Which <language>?"]
    answers = ao_readout(cache, [0, 1], [3, -1], model, tok, config, questions, window=2, max_new_tokens=5, seed=4, input_src=list(range(10, 19)))
    assert len(captured) == 8 and all(c["vecs"].shape == (1, 2, D) and len(c["positions"]) == 2 and c["max_new_tokens"] == 5 and c["seed"] == 4 and c["kw"] == {"do_sample": False} for c in captured)
    assert list(answers) == [0, 1] and list(answers[0]) == [3, -1] and list(answers[0][3]) == questions
    assert answers[0][3] == {"What topic?": "answer 1 row 0", "Which <language>?": "answer 2 row 0"} and answers[0][-1]["What topic?"] == "answer 3 row 0" and answers[1][3]["What topic?"] == "answer 5 row 0" and answers[1][-1]["Which <language>?"] == "answer 8 row 0"
    assert t.equal(captured[0]["vecs"][0], cache["blocks.0.hook_resid_post"][0, 2:4]) and t.equal(captured[2]["vecs"][0], cache["blocks.0.hook_resid_post"][0, 7:9]) and t.equal(captured[5]["vecs"][0], cache["blocks.1.hook_resid_post"][0, 2:4])
    assert "Layer: 0\n" in tok.decode(captured[0]["ids"]) and "Layer: 1\n" in tok.decode(captured[4]["ids"])
    h = shown[-1]
    assert h.count("class='pane'") == 4 and "<button>L0</button><button>L1</button>" in h and "<button>p3</button><button>p-1</button>" in h and h.count("class='tb' hidden") == 0
    assert h.count("data-p=") == 2 and "data-p=0 data-h='pos 3 &middot; id 13" in h and "data-p=1 data-h='pos 8 &middot; id 18" in h
    assert h.count("<tr") == 12 and h.count("Which &lt;language&gt;?") == 4 and "answer 8 row 0" in h and "<th colspan=2>L1 &middot; p7..p8</th>" in h and "activation oracle readout</h3>" in h
    answers = ao_readout(cache, [0], 2, model, tok, config, ["q"], hook="hook_resid_pre", title="T")
    assert answers == {0: {"q": "answer 9 row 0"}} and t.equal(captured[-1]["vecs"][0], cache["blocks.0.hook_resid_pre"][0, 2:3])  # an int pos collapses the position level, like cluster_readout
    h = shown[-1]
    assert h.count("class='pane'") == 1 and h.count("class='tb' hidden") == 2 and "data-p=" not in h and "<th colspan=2>L0 &middot; p2</th>" in h and "T</h3>" in h
    assert ao_readout(cache, [0, 1], -1, model, tok, config, ["q"], input_src=list(range(9))) == {0: {"q": "answer 10 row 0"}, 1: {"q": "answer 11 row 0"}} and t.equal(captured[-1]["vecs"][0], cache["blocks.1.hook_resid_post"][0, 8:9]) and "data-h='pos 8 &middot; id 8" in shown[-1] and "outline:1px solid #fc6" in shown[-1] and "data-p=" not in shown[-1]  # an int pos marks the token without making it a tab
    n_calls, n_shown = len(captured), len(shown)
    with pytest.raises(ValueError, match="a window of 2 at position 0 reaches outside the 9-token sequence"):
        ao_readout(cache, [0], 0, model, tok, config, ["q"], window=2)
    with pytest.raises(ValueError, match="a window of 1 at position 9 reaches outside the 9-token sequence"):
        ao_readout(cache, [0], [9, 3], model, tok, config, ["q"])
    with pytest.raises(ValueError, match="a window of 1 at position -10 reaches outside the 9-token sequence"):
        ao_readout(cache, [0], -10, model, tok, config, ["q"])
    with pytest.raises(ValueError, match="window must be at least 1, got 0"):
        ao_readout(cache, [0], 3, model, tok, config, ["q"], window=0)
    with pytest.raises(ValueError, match="input_src has 5 tokens but the cache holds 9 positions"):
        ao_readout(cache, [0], -1, model, tok, config, ["q"], input_src=["a"] * 5)
    with pytest.raises(ValueError, match=r"blocks\.0\.hook_resid_post is a batch of 2; ao_readout takes the cache of one prompt"):
        ao_readout({"blocks.0.hook_resid_post": t.randn(2, 9, D)}, [0], 3, model, tok, config, ["q"])
    with pytest.raises(TypeError, match="questions is a list of strings, got the one string 'What topic\\?'"):
        ao_readout(cache, [0], 3, model, tok, config, "What topic?")
    with pytest.raises(KeyError):
        ao_readout(cache, [1], 0, model, tok, config, ["q"], hook="hook_resid_pre")
    with pytest.raises(ValueError, match="found 2"):
        ao_readout(cache, [0], 0, model, tok, config, ["q ?"])
    assert len(captured) == n_calls and len(shown) == n_shown  # every raise comes before any generation or display

@pytest.mark.hf
def test_ao_readout_from_bridge_cache(booted, shown):
    model, config, bridge = booted
    tok = bridge.tokenizer
    ids = to_ids([{"role": "user", "content": "The capital of France is"}], tok, add_generation_prompt=True)
    with model.disable_adapter():
        _, cache = bridge.run_with_cache(t.tensor([ids]))
    ref = tiny_hf()(t.tensor([ids]), output_hidden_states=True).hidden_states
    assert t.allclose(cache["blocks.0.hook_resid_post"], ref[1], atol=1e-5)  # the capture is the subject model's block output (block 1 is the last, which HF's hidden_states norms)
    assert not t.allclose(bridge.run_with_cache(t.tensor([ids]))[1]["blocks.0.hook_resid_post"], ref[1], atol=1e-5)  # with the adapter active it would be the oracle's
    questions = ["What topic?", "Which language?"]
    answers = ao_readout(cache, [0, 1], [-1, -3], model, tok, config, questions, input_src=ids, max_new_tokens=3)
    assert list(answers) == [0, 1] and list(answers[0]) == [-1, -3] and all(isinstance(a, str) for by_pos in answers.values() for by_q in by_pos.values() for a in by_q.values())
    assert answers[1][-1]["What topic?"] == ao_read(model, tok, cache["blocks.1.hook_resid_post"][0, -1], 1, "What topic?", config, max_new_tokens=3)[0]
    h = shown[-1]
    assert h.count("class='pane'") == 4 and "<button>L0</button><button>L1</button>" in h and "<button>p-1</button><button>p-3</button>" in h and h.count("data-p=") == 2 and h.count("<tr") == 12

@pytest.mark.hf
@pytest.mark.parametrize("name, special_id", [("meta-llama/Llama-3.2-1B-Instruct", 949), ("Qwen/Qwen3-0.6B", 937), ("Qwen/Qwen3.6-27B", 907), ("google/gemma-3-1b-it", 2360)])
def test_ao_prompt_on_the_checkpoints_tokenizers(name, special_id):
    tok = load_tokenizer(name)
    ids, positions = ao_prompt(tok, CONFIG, 8, 3, "What is the model thinking about?")
    assert positions == list(range(positions[0], positions[0] + 3)) and all(ids[p] == special_id for p in positions) and ids.count(special_id) == 3
    text = tok.decode(ids)
    assert "Layer: 8\n ? ? ? \nWhat is the model thinking about?" in text
    if "Qwen" in name:
        assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")  # enable_thinking=False renders the empty think block
