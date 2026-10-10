import pytest
import torch as t
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from conftest import TINY_MODEL, first_pane, load_tokenizer, widget_data
from mechtools import olens
from mechtools.hooks import decoder_layers
from mechtools.models import boot_bridge, load_adapters
from mechtools.lens import readout_grid
from mechtools.olens import *
from mechtools.stats import normed
from mechtools.tokens import single_token_marker

D = 16
GREEDY = {"do_sample": False, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 3}
td = lambda s: f"<td class=w><div>{s}</div></td>"  # a readout cell in the wrapping column: its text in a div the frame caps at 100ch

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
    return {"base": TINY_MODEL, "layers": [0, 1], "alpha": 7.0, "marker": (char, cid, (ids[slot - 1], ids[slot + 1])), "sampling": dict(GREEDY)}

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
    """inject_generate replaced by a recorder returning '- {i}a\\n- {i}b\\nnoise' for row i; `seen` holds the last call's arguments and, under "calls", every call's vecs."""
    seen = {"calls": []}
    def fake(model, tokenizer, ids, module, positions, vecs, write, max_new_tokens, seed, **kw):
        seen.update(model=model, tokenizer=tokenizer, ids=ids, module=module, positions=positions, vecs=vecs.clone(), write=write, max_new_tokens=max_new_tokens, seed=seed, kw=kw)
        seen["calls"].append(vecs.clone())
        return [f"- {i}a\n- {i}b\nnoise" for i in range(vecs.shape[0])]
    monkeypatch.setattr(olens, "inject_generate", fake)
    return seen

def test_contract_table():
    assert list(OLENS) == ["agu18dec/olens_and_ar:olens_s3d_rl600", "andyx10/oracle-lens-qwen3-4b"]
    for c in OLENS.values():
        assert set(c) == {"base", "layers", "alpha", "marker", "sampling"} and set(c["sampling"]) == {"do_sample", "temperature", "top_p", "top_k", "max_new_tokens"} and c["alpha"] == 16000.0 and c["sampling"]["do_sample"] is True
    agu, andy = OLENS.values()
    assert agu["base"] == "Qwen/Qwen3.6-27B" and agu["layers"] == list(range(20, 61, 4)) and agu["marker"] == ("㈜", 158983, (29, 510)) and agu["sampling"] == {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_new_tokens": 256}
    assert andy["base"] == "Qwen/Qwen3-4B" and andy["layers"] == list(range(12, 35, 2)) and andy["marker"] == ("㈎", 149705, (29, 522)) and andy["sampling"] == {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 128}
    assert OLENS_PROMPT.format(layer=44, char="㈜") == "An activation vector from layer 44 of a language model is enclosed in activation tags: <activation>㈜</activation>. Produce distinct concepts that encode this activation, each as a '- ' bullet on its own line."

def test_parse_bullets():
    assert parse_bullets("Concepts:\n- cats\n\n  - dogs and  \n-not a bullet\n- \n-\n--dash\nplain line\n- last") == ["cats", "dogs and", "last"]
    assert parse_bullets("") == [] and parse_bullets("- one") == ["one"] and parse_bullets("- a - b\n") == ["a - b"]

def test_load_olens_unknown_spec_raises_before_loading():
    with pytest.raises(KeyError, match=r"'nope/nope'.*agu18dec/olens_and_ar:olens_s3d_rl600.*andyx10/oracle-lens-qwen3-4b.*contract="):
        load_olens(object(), "nope/nope")  # object() has no modules(), so load_adapters would have raised AttributeError had it been called

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
    assert out == [[["0a", "0b"], ["1a", "1b"], ["2a", "2b"]], [["3a", "3b"], ["4a", "4b"], ["5a", "5b"]]]
    raw = olens_read(peft, tok, h[0], 0, c, n=2, raw=True, chars="ZQXJ")
    assert raw == ["- 0a\n- 0b\nnoise", "- 1a\n- 1b\nnoise"] and seen["vecs"].shape == (2, D) and t.equal(seen["vecs"][1], h[0])
    assert seen["kw"] == {"do_sample": False, "temperature": 1.0, "top_p": 1.0, "top_k": 0} and seen["max_new_tokens"] == 3 and seen["seed"] == 0 and seen["ids"] == olens_prompt(tok, 0, c, chars="ZQXJ")[0]
    assert olens_read(peft, tok, h[0], 0, c, chars="ZQXJ") == [["0a", "0b"]] and seen["vecs"].shape == (1, D)
    assert olens_read(peft, tok, h[0].bfloat16(), 0, c, seed=None, chars="ZQXJ") == [["0a", "0b"]] and seen["vecs"].dtype == t.bfloat16 and seen["seed"] is None
    assert olens_read(peft, tok, h[:1], 0, c, chars="ZQXJ") == [[["0a", "0b"]]]  # [1, d] keeps the outer list
    with pytest.raises(ValueError, match=r"\[d\] or \[m, d\], got \(2, 3, 16\)"):
        olens_read(peft, tok, t.randn(2, 3, D), 0, c, chars="ZQXJ")
    with pytest.raises(ValueError, match="n must be at least 1, got 0"):
        olens_read(peft, tok, h[0], 0, c, n=0, chars="ZQXJ")

def test_olens_read_end_to_end(lens, tok):
    peft, c, _ = lens
    h = t.randn(2, D)
    seen = []
    handle = decoder_layers(peft)[0].register_forward_pre_hook(lambda m, args, kwargs: seen.append(args[0].clone()), with_kwargs=True)
    out = olens_read(peft, tok, h, 1, c, n=3, raw=True, chars="ZQXJ", max_new_tokens=1)
    handle.remove()
    ids, slot = olens_prompt(tok, 1, c, chars="ZQXJ")
    x = seen[0]  # what block 0 saw on the prefill
    assert x.shape == (6, len(ids), D) and t.allclose(x[:, slot], 7.0 * normed(h).repeat_interleave(3, 0), atol=1e-5)  # alpha * unit(h), activation-major
    emb = peft.get_input_embeddings()(t.tensor([ids]))[0]
    assert t.allclose(x[:, :slot], emb[:slot].expand(6, -1, -1), atol=1e-6) and t.allclose(x[:, slot + 1:], emb[slot + 1:].expand(6, -1, -1), atol=1e-6)  # every other row is the prompt's embedding
    assert len(out) == 2 and all(len(row) == 3 and all(isinstance(s, str) for s in row) for row in out)
    a = olens_read(peft, tok, h, 1, c, n=2, raw=True, chars="ZQXJ", do_sample=True, top_p=0.95, top_k=64)
    b = olens_read(peft, tok, h, 1, c, n=2, raw=True, chars="ZQXJ", do_sample=True, top_p=0.95, top_k=64)
    assert a == b and len(a) == 2 and len(a[0]) == 2  # deterministic given (activations, n, seed): the same batch reproduces its block
    assert olens_read(peft, tok, h[:1], 1, c, n=2, raw=True, chars="ZQXJ", do_sample=True, top_p=0.95, top_k=64) == [olens_read(peft, tok, h[0], 1, c, n=2, raw=True, chars="ZQXJ", do_sample=True, top_p=0.95, top_k=64)]  # [1, d] and [d] are the same batch
    one = olens_read(peft, tok, h[0], 0, c, chars="ZQXJ")
    assert len(one) == 1 and isinstance(one[0], list) and all(isinstance(s, str) for s in one[0])

def test_olens_readout(lens, tok, fake_generate, shown, monkeypatch):
    peft, c, _ = lens
    seen, vecs = fake_generate, fake_generate["calls"]
    toks = tok.encode("Hello there my old friend")
    cache = {f"blocks.{l}.hook_resid_post": t.randn(1, len(toks), D) for l in range(2)} | {"blocks.0.hook_resid_pre": t.randn(1, len(toks), D)}
    out = olens_readout(cache, [0, 1], [2, -1], peft, tok, c, n=2, seed=7, input_src=toks, chars="ZQXJ")
    assert list(out) == [0, 1] and list(out[0]) == [2, -1] and out[0][2] == [["0a", "0b"], ["1a", "1b"]] and out[0][-1] == [["2a", "2b"], ["3a", "3b"]] and out[1] == out[0]
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
    assert out == {1: [["0a", "0b"]]} and "application/json" not in h and first_pane(h) == readout_grid([("sample 0", [["0a"], ["0b"]], None, ["w"])]) and h.count("class='tb' hidden") == 2 and "<span data-p=" not in h and h.count("outline:1px solid #fc6") == 1 and "T</h3>" in h and "--n:1;" in h and seen["seed"] == 0
    out = olens_readout(cache, [0], [1], peft, tok, c, raw=True, hook="hook_resid_pre", chars="ZQXJ")
    assert out == {0: {1: ["- 0a\n- 0b\nnoise"]}} and t.equal(vecs[-1], cache["blocks.0.hook_resid_pre"][0, [1]]) and td("- 0a") in shown[-1] and td("noise") in shown[-1] and shown[-1].count("<tr") == 4 and "class='tk'" not in shown[-1]  # raw: a line per row; no input_src, no strip
    n_calls = len(vecs)
    with pytest.raises(ValueError, match=r"repeated positions: \[2, 2\]"):
        olens_readout(cache, [0], [2, 2], peft, tok, c, chars="ZQXJ")
    with pytest.raises(ValueError, match="input_src is empty"):  # input_src is resolved before any generation
        olens_readout(cache, [0], 0, peft, tok, c, input_src=[], chars="ZQXJ")
    assert len(vecs) == n_calls and len(shown) == 4
    monkeypatch.setattr(olens, "inject_generate", lambda *a, **kw: ["- a<b & c"])
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
    assert list(out) == [0, 1] and list(out[1]) == [0, -1] and all(len(out[l][p]) == 2 and all(isinstance(s, list) for s in out[l][p]) for l in out for p in out[l])
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
