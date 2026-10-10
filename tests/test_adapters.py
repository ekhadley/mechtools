import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM

from conftest import TINY_MODEL, load_tokenizer
from mechtools.hooks import decoder_layers, inject_generate
from mechtools.models import adapter_spec, boot_bridge, check_active, load_adapters, load_bridge, load_hf_model
from mechtools.olens import OLENS_PROMPT
from mechtools.stats import normed
from mechtools.tokens import single_token_marker, to_ids

PROMPT = "The quick brown fox jumps over"  # 9 tokens with BOS on the tiny Llama-2 tokenizer, asserted by the ids fixture
POS = [2, 8]  # 8 is the last prompt token, where a write moves the tiny random model's greedy continuation (one at 2 and 5 alone does not)
GREEDY = dict(do_sample=False, max_new_tokens=4)
VECS = t.randn(3, 2, 16, generator=t.Generator().manual_seed(0))  # 3 rows, 2 positions, d_model 16

def tiny(dtype=t.float32):
    return AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=dtype).eval()

def logits(model, ids):
    return model(input_ids=t.tensor([ids])).logits

def cut(tok, row, eos=(1, 2)):
    """The decoded row up to its first eos id, as inject_generate returns it."""
    return tok.decode(row[:next((i for i, x in enumerate(row) if x in eos), len(row))], skip_special_tokens=True)

def save_lora(path, targets, seed) -> str:
    """A LoRA on TINY_MODEL saved to `path`, with lora_B drawn at random: it is zero at init, which would make the adapter the identity."""
    t.manual_seed(seed)
    pm = get_peft_model(tiny(), LoraConfig(r=4, lora_alpha=8, target_modules=targets))
    for name, p in pm.named_parameters():
        if "lora_B" in name:
            p.data.normal_()
    pm.save_pretrained(path)
    return str(path)

def edit_targets(path, targets):
    cfg_path = Path(path) / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["target_modules"] = targets
    cfg_path.write_text(json.dumps(cfg))

def olens_render(tok, layer):
    return lambda char: to_ids([{"role": "user", "content": OLENS_PROMPT.format(layer=layer, char=char)}], tok, add_generation_prompt=True, enable_thinking=False)

@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(TINY_MODEL)

@pytest.fixture(scope="module")
def plain():
    """TINY_MODEL without adapters, the reference for base-model logits. Skips when it is not cached."""
    try:
        return tiny()
    except OSError as e:
        pytest.skip(f"{TINY_MODEL} is not in the local HF cache, run `hf download {TINY_MODEL}` ({type(e).__name__})")

@pytest.fixture(scope="module")
def loras(plain, tmp_path_factory):
    """Two random LoRAs with different target sets: a on q_proj and v_proj, b on q_proj and o_proj. `plain` is taken for its skip."""
    d = tmp_path_factory.mktemp("loras")
    return {"a": save_lora(d / "a", ["q_proj", "v_proj"], 1), "b": save_lora(d / "b", ["q_proj", "o_proj"], 2)}

@pytest.fixture
def pm(loras):
    return load_adapters(tiny(), loras)

@pytest.fixture(scope="module")
def booted(loras):
    """hf -> load_adapters -> boot_bridge, the loading order the readouts assume."""
    hf = tiny()
    pm = load_adapters(hf, loras)
    return SimpleNamespace(hf=hf, pm=pm, bridge=boot_bridge(hf))

@pytest.fixture
def ids(tok):
    ids = tok.encode(PROMPT)
    assert len(ids) == 9  # POS and the comments above count on it
    return ids

# adapter_spec

def test_adapter_spec(tmp_path):
    assert adapter_spec(str(tmp_path)) == (str(tmp_path), {})
    assert adapter_spec("org/repo") == ("org/repo", {})
    assert adapter_spec("org/repo:sub/dir") == ("org/repo", {"subfolder": "sub/dir"})
    assert adapter_spec(str(tmp_path / "missing")) == (str(tmp_path / "missing"), {})  # not a directory, so a repo id for peft to resolve

# load_adapters

@pytest.mark.hf
def test_load_adapters_wraps_the_same_hf_model(loras):
    hf = tiny()
    pm = load_adapters(hf, loras)
    assert isinstance(pm, PeftModel) and pm.base_model.model is hf and set(pm.peft_config) == {"a", "b"}
    assert pm.active_adapters == ["a"] and not pm.training and not any(p.requires_grad for p in pm.parameters())
    lora = {name.rsplit(".", 1)[1]: m for name, m in pm.named_modules() if isinstance(m, LoraLayer)}
    assert set(lora) == {"q_proj", "v_proj", "o_proj"} and set(lora["q_proj"].lora_A) == {"a", "b"} and set(lora["v_proj"].lora_A) == {"a"} and set(lora["o_proj"].lora_A) == {"b"}

@pytest.mark.hf
def test_active_adapter_changes_logits_and_disable_adapter_restores_the_base(pm, plain, ids):
    ref = logits(plain, ids)
    pm.set_adapter("a")
    assert (logits(pm, ids) - ref).abs().max() > 0.1
    with pm.disable_adapter():
        t.testing.assert_close(logits(pm, ids), ref, rtol=0, atol=1e-5)
    assert (logits(pm, ids) - ref).abs().max() > 0.1  # enabled again after the block

@pytest.mark.hf
def test_stack_adds_the_deltas(pm, ids):
    pm.set_adapter("a")
    la = logits(pm, ids)
    pm.set_adapter("b")
    lb = logits(pm, ids)
    pm.base_model.set_adapter(["a", "b"])
    lab = logits(pm, ids)
    assert pm.active_adapters == ["a", "b"] and (lab - la).abs().max() > 0.1 and (lab - lb).abs().max() > 0.1
    q, o = decoder_layers(pm)[0].self_attn.q_proj, decoder_layers(pm)[0].self_attn.o_proj
    base_q, base_o = q.base_layer.weight.clone(), o.base_layer.weight.clone()
    def merged_delta(names):
        pm.base_model.set_adapter(names)
        pm.base_model.merge_adapter()
        dq, do = q.base_layer.weight - base_q, o.base_layer.weight - base_o
        pm.base_model.unmerge_adapter()
        return dq, do
    dq_a, do_a = merged_delta(["a"])
    dq_b, do_b = merged_delta(["b"])
    dq_ab, do_ab = merged_delta(["a", "b"])
    assert dq_a.abs().max() > 0.1 and dq_b.abs().max() > 0.1 and do_b.abs().max() > 0.1 and do_a.abs().max() == 0  # a has no o_proj
    t.testing.assert_close(dq_ab, dq_a + dq_b, rtol=0, atol=1e-5)
    t.testing.assert_close(do_ab, do_b, rtol=0, atol=1e-5)
    t.testing.assert_close(q.base_layer.weight, base_q, rtol=0, atol=1e-5)  # unmerge restores the base, one float rounding per cycle
    assert not q.merged and not o.merged

@pytest.mark.hf
def test_second_load_adds_a_name_and_a_duplicate_raises(pm, loras, tmp_path):
    c = save_lora(tmp_path / "c", ["v_proj"], 3)
    assert load_adapters(pm, {"c": c}) is pm and set(pm.peft_config) == {"a", "b", "c"} and pm.active_adapters == ["a"]  # a later load does not change which adapter is active
    with pytest.raises(ValueError, match="adapter 'a' is already loaded"):
        load_adapters(pm, {"a": loras["a"]})

@pytest.mark.hf
def test_load_adapters_raises_on_a_mismatch(plain, loras, tmp_path):
    sub = save_lora(tmp_path / "sub", ["q_proj", "v_proj"], 4)
    edit_targets(sub, ["q_proj"])  # the saved v_proj tensors land on no module
    with pytest.raises(ValueError, match=r"adapter 'd' .* 4 saved tensors with no module, 0 LoRA modules with no tensor, e\.g\. \['base_model.*v_proj\.lora_A\.d\.weight"):
        load_adapters(tiny(), {"d": sub})
    sup = save_lora(tmp_path / "sup", ["q_proj", "v_proj"], 5)
    edit_targets(sup, ["q_proj", "v_proj", "o_proj"])  # the o_proj LoRA modules get no tensor
    with pytest.raises(ValueError, match=r"adapter 'e' .* 0 saved tensors with no module, 4 LoRA modules with no tensor, e\.g\. \['base_model.*o_proj\.lora_A\.e\.weight"):
        load_adapters(tiny(), {"e": sup})
    with pytest.raises(ValueError, match=r"0 saved tensors with no module, 4 LoRA modules with no tensor"):  # as a second adapter, the first one's tensors are not counted as missing
        load_adapters(load_adapters(tiny(), {"a": loras["a"]}), {"e": sup})

@pytest.mark.hf
def test_load_adapters_on_a_booted_model_raises(booted, loras):
    for model in (booted.hf, booted.pm, booted.bridge):
        with pytest.raises(TypeError, match="already wrapped by a TransformerBridge"):
            load_adapters(model, {"z": loras["a"]})

@pytest.mark.hf
def test_autocast_adapter_dtype(loras):
    def dtypes(model):
        return {p.dtype for n, p in model.named_parameters() if "lora_" in n}, {p.dtype for n, p in model.named_parameters() if "lora_" not in n}
    assert dtypes(load_adapters(tiny(t.bfloat16), {"a": loras["a"]})) == ({t.float32}, {t.bfloat16})  # peft's default casts the adapter up
    assert dtypes(load_adapters(tiny(t.bfloat16), {"a": loras["a"]}, autocast_adapter_dtype=False)) == ({t.bfloat16}, {t.bfloat16})

# boot_bridge

@pytest.mark.hf
def test_bridge_runs_through_the_active_adapter(booted, plain, ids):
    assert booted.bridge.original_model is booted.hf
    q = decoder_layers(booted.bridge)[0].self_attn.q_proj
    assert type(q).__name__ == "LinearBridge" and isinstance(q.original_component, LoraLayer)  # the Bridge wraps the LoRA module, which stays toggleable
    booted.pm.set_adapter("a")
    out = booted.bridge(t.tensor([ids]))
    t.testing.assert_close(out, logits(booted.pm, ids), rtol=0, atol=1e-5)
    assert (out - logits(plain, ids)).abs().max() > 0.1
    booted.pm.set_adapter("b")
    t.testing.assert_close(booted.bridge(t.tensor([ids])), logits(booted.pm, ids), rtol=0, atol=1e-5)  # follows a switch

@pytest.mark.hf
def test_bridge_inside_disable_adapter_is_the_base_and_resid_post_L_is_hidden_states_L_plus_1(booted, plain, ids):
    hs = plain(input_ids=t.tensor([ids]), output_hidden_states=True).hidden_states
    with booted.pm.disable_adapter():
        out, cache = booted.bridge.run_with_cache(t.tensor([ids]))
    t.testing.assert_close(out, logits(plain, ids), rtol=0, atol=1e-5)
    t.testing.assert_close(cache["blocks.0.hook_resid_post"], hs[1], rtol=0, atol=1e-5)
    t.testing.assert_close(cache["blocks.1.hook_resid_pre"], hs[1], rtol=0, atol=1e-5)
    assert len(hs) == 3 and (cache["blocks.1.hook_resid_post"] - hs[2]).abs().max() > 0.1  # HF's last entry has the final norm applied

@pytest.mark.hf
def test_boot_bridge_dtype_defaults_to_the_models(plain, ids):
    bridge = boot_bridge(tiny(t.bfloat16))
    assert next(bridge.parameters()).dtype == t.bfloat16 and bridge(t.tensor([ids])).dtype == t.bfloat16
    assert not bridge.training and not any(p.requires_grad for p in bridge.parameters())

@pytest.mark.hf
def test_load_bridge_is_boot_bridge_of_load_hf_model(plain, ids):
    a = load_bridge(TINY_MODEL, dtype=t.float32, device_map="cpu")(t.tensor([ids]))
    b = boot_bridge(load_hf_model(TINY_MODEL, dtype=t.float32, device_map="cpu"))(t.tensor([ids]))
    assert t.equal(a, b)
    t.testing.assert_close(a, logits(plain, ids), rtol=0, atol=1e-5)

# check_active

@pytest.mark.hf
def test_check_active(pm):
    pm.set_adapter("a")
    check_active(pm, ["a"])
    check_active(pm, {"a"})
    check_active(pm, ("a",))
    for names in (["b"], ["a", "b"], []):
        with pytest.raises(RuntimeError, match=r"active adapters \['a'\] are not"):
            check_active(pm, names)
    pm.base_model.set_adapter(["a", "b"])
    check_active(pm, ["b", "a"])
    check_active(pm, {"a", "b"})
    with pytest.raises(RuntimeError, match=r"active adapters \['a', 'b'\] are not \['a'\]"):
        check_active(pm, ["a"])
    with pm.disable_adapter():
        with pytest.raises(RuntimeError, match=r"disabled .* \['a', 'b'\] must be active"):
            check_active(pm, ["a", "b"])
    check_active(pm, ["a", "b"])  # enabled again after the block
    merged = pm.merge_and_unload()
    assert not isinstance(merged, PeftModel)
    check_active(merged, ["a", "b"])

def test_check_active_passes_for_a_plain_module():
    check_active(t.nn.Linear(2, 2), ["a"])
    check_active(t.nn.Linear(2, 2), [])

# decoder_layers

@pytest.mark.hf
def test_decoder_layers_are_the_same_objects_through_peft_and_the_bridge(booted):
    layers = decoder_layers(booted.hf)
    assert len(layers) == 2 and all(x is y is z for x, y, z in zip(layers, decoder_layers(booted.pm), decoder_layers(booted.bridge), strict=True))
    assert all(layer is block for layer, block in zip(layers, booted.bridge.blocks, strict=True))

@pytest.mark.hf
def test_decoder_layer_L_outputs_hidden_states_L_plus_1(plain, ids):
    outs = []
    handles = [layer.register_forward_hook(lambda m, a, o: outs.append(o)) for layer in decoder_layers(plain)]
    hs = plain(input_ids=t.tensor([ids]), output_hidden_states=True).hidden_states
    for h in handles:
        h.remove()
    assert len(outs) == 2 and t.equal(outs[0], hs[1]) and (outs[1] - hs[2]).abs().max() > 0.1  # the last hidden_states entry has the final norm applied

@pytest.mark.hf
def test_boot_bridge_wraps_the_block_objects(plain):
    """boot_bridge puts a BlockBridge in the layer list with the old block as its original_component, which still runs, so a handle taken before booting keeps firing. `plain` is taken for its skip."""
    hf = tiny()
    before = list(decoder_layers(hf))
    fired = []
    handle = before[0].register_forward_hook(lambda m, a, o: fired.append(1))
    bridge = boot_bridge(hf)
    bridge(t.tensor([[1, 450, 4996]]))
    handle.remove()
    after = decoder_layers(hf)
    assert after[0] is not before[0] and after[0].original_component is before[0] and fired == [1]

# inject_generate

@pytest.mark.hf
def test_embed_replace_matches_a_hand_built_inputs_embeds_generate(plain, tok, ids):
    embed = plain.get_input_embeddings()
    texts = inject_generate(plain, tok, ids, embed, POS, VECS, lambda o, v: v, **GREEDY)
    prompt = t.tensor([ids] * 3)
    with t.no_grad():
        e = embed(prompt)
        e[:, POS] = VECS
        out = plain.generate(inputs_embeds=e, attention_mask=t.ones_like(prompt), eos_token_id=[1, 2], pad_token_id=0, **GREEDY)
        ref = plain.generate(input_ids=prompt, attention_mask=t.ones_like(prompt), eos_token_id=[1, 2], pad_token_id=0, **GREEDY)
    assert len(texts) == 3 and all(isinstance(s, str) for s in texts)
    assert texts == [cut(tok, row) for row in out.tolist()]  # with inputs_embeds, generate returns the new tokens only
    assert texts != inject_generate(plain, tok, ids, embed, POS, VECS, lambda o, v: o, **GREEDY) == [cut(tok, row) for row in ref[:, len(ids):].tolist()]  # the identity write is plain generation

@pytest.mark.hf
def test_inject_generate_runs_on_a_peft_model(pm, tok, ids):
    pm.set_adapter("a")
    prompt = t.tensor([ids] * 3)
    ref = pm.generate(input_ids=prompt, attention_mask=t.ones_like(prompt), eos_token_id=[1, 2], pad_token_id=0, **GREEDY)
    assert inject_generate(pm, tok, ids, pm.get_input_embeddings(), POS, VECS, lambda o, v: o, **GREEDY) == [cut(tok, row) for row in ref[:, len(ids):].tolist()]

@pytest.mark.hf
def test_norm_matched_add_at_layer_1(plain, tok, ids):
    norm_in = []
    handle = plain.get_decoder().norm.register_forward_pre_hook(lambda m, args: norm_in.append(args[0].clone()) if args[0].shape[1] == len(ids) else None)  # the module after the last block, prefill only
    with t.no_grad():
        plain(input_ids=t.tensor([ids] * 3))
    texts = inject_generate(plain, tok, ids, decoder_layers(plain)[1], POS, VECS, lambda o, v: o + o.norm(dim=-1, keepdim=True) * normed(v), **GREEDY)
    handle.remove()
    orig, got = norm_in
    expected = orig.clone()
    expected[:, POS] = orig[:, POS] + orig[:, POS].norm(dim=-1, keepdim=True) * normed(VECS)
    t.testing.assert_close(got, expected, rtol=0, atol=1e-6)
    others = [p for p in range(len(ids)) if p not in POS]
    assert t.equal(got[:, others], orig[:, others])
    assert texts != inject_generate(plain, tok, ids, decoder_layers(plain)[1], POS, VECS, lambda o, v: o, **GREEDY)

@pytest.mark.hf
@pytest.mark.parametrize("use_cache", [True, False])
def test_write_runs_on_the_prefill_only_with_a_cache(plain, tok, ids, use_cache):
    layer = decoder_layers(plain)[1]
    calls, seen = [], []
    before = set(layer._forward_hooks)  # transformers leaves its own output-capturing hook on a layer after a forward with output_hidden_states=True
    handle = layer.register_forward_hook(lambda m, a, o: seen.append(tuple(o.shape)))
    def write(o, v):
        calls.append((tuple(o.shape), o.dtype, tuple(v.shape), v.dtype))
        return o + v
    inject_generate(plain, tok, ids, layer, POS, VECS, write, use_cache=use_cache, **GREEDY)
    assert set(layer._forward_hooks) == before | {handle.id}  # inject_generate's hook is gone, the test's own is left
    handle.remove()
    assert set(calls) == {((3, 2, 16), t.float32, (3, 2, 16), t.float32)} and len(seen) > 1
    if use_cache:
        assert len(calls) == 1 and seen == [(3, len(ids), 16)] + [(3, 1, 16)] * (len(seen) - 1)  # decode steps see one position and are left alone
    else:
        assert len(calls) == len(seen) and seen == [(3, len(ids) + i, 16) for i in range(len(seen))]  # every step holds the whole sequence and is written

@pytest.mark.hf
def test_hook_is_removed_after_the_call_and_after_an_exception_in_write(plain, tok, ids):
    embed = plain.get_input_embeddings()
    inject_generate(plain, tok, ids, embed, POS, VECS, lambda o, v: v, **GREEDY)
    assert not embed._forward_hooks
    def boom(o, v):
        raise RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        inject_generate(plain, tok, ids, embed, POS, VECS, boom, **GREEDY)
    assert not embed._forward_hooks

@pytest.mark.hf
def test_seed_controls_sampling(plain, tok, ids):
    sample = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=8)
    def run(vecs=VECS, **kw):
        return inject_generate(plain, tok, ids, plain.get_input_embeddings(), POS, vecs, lambda o, v: v, **{**sample, **kw})
    assert run(seed=1) == run(seed=1) != run(seed=2)
    same = VECS[:1].expand(3, 2, 16)
    assert len(set(run(same, seed=1))) == 3 and len(set(run(same, do_sample=False))) == 1  # identical rows are sampled independently; greedy decoding shows they are identical
    t.manual_seed(1)
    assert run(seed=None) == run(seed=1)  # None leaves the RNG alone

@pytest.mark.hf
def test_vecs_shapes(plain, tok, ids):
    embed = plain.get_input_embeddings()
    write = lambda o, v: v
    assert inject_generate(plain, tok, ids, embed, [8], VECS[:, 0], write, **GREEDY) == inject_generate(plain, tok, ids, embed, [8], VECS[:, :1], write, **GREEDY)  # [n, d] is [n, 1, d]
    with pytest.raises(ValueError, match=r"vecs must be \[n, 1, d\] for 1 positions, got \(3, 2, 16\)"):
        inject_generate(plain, tok, ids, embed, [3], VECS, write, **GREEDY)
    with pytest.raises(ValueError, match=r"vecs must be \[n, 3, d\] for 3 positions, got \(3, 2, 16\)"):
        inject_generate(plain, tok, ids, embed, [3, 4, 5], VECS, write, **GREEDY)
    with pytest.raises(ValueError, match=r"got \(16,\)"):  # a bare vector, reported as passed
        inject_generate(plain, tok, ids, embed, [3], VECS[0, 0], write, **GREEDY)
    with pytest.raises(ValueError, match=r"vecs must be \[n, 2, d\] for 2 positions, got \(3, 1, 16\)"):  # [n, d] is [n, 1, d] whatever the positions
        inject_generate(plain, tok, ids, embed, POS, VECS[:, 0], write, **GREEDY)

@pytest.mark.hf
def test_hook_rewrites_the_first_element_of_a_tuple_output_in_the_activations_dtype(plain, tok, ids, monkeypatch):
    """A bf16 activation and bf16 vecs reach `write` as float32, and its float64 result goes back into the activation as bf16."""
    class Block(t.nn.Module):
        def forward(self, x):
            return x, "cache"
    block, seen, dtypes = Block(), [], []
    def generate(**kw):
        seen.append(block(t.zeros(3, len(ids), 16, dtype=t.bfloat16)))
        seen.append(block(t.zeros(3, 1, 16, dtype=t.bfloat16)))  # a decode step
        return kw["input_ids"]
    monkeypatch.setattr(plain, "generate", generate)
    def write(o, v):
        dtypes.append((o.dtype, v.dtype))
        return v.double()
    assert inject_generate(plain, tok, ids, block, [4], VECS[:, 0].bfloat16(), write, **GREEDY) == ["", "", ""]
    (new, extra), (step, _) = seen
    assert extra == "cache" and new.dtype == t.bfloat16 and dtypes == [(t.float32, t.float32)]
    assert t.equal(new[:, 4], VECS[:, 0].bfloat16()) and new[:, [p for p in range(len(ids)) if p != 4]].abs().sum() == 0
    assert t.equal(step, t.zeros(3, 1, 16, dtype=t.bfloat16))

@pytest.mark.hf
def test_generate_gets_both_eos_ids_the_pad_id_and_the_kwargs_and_rows_are_cut_at_eos(plain, tok, ids, monkeypatch):
    assert plain.generation_config.eos_token_id == 1 and tok.eos_token_id == 2 and tok.pad_token_id == 0
    words = tok.encode("one two three four", add_special_tokens=False)
    assert len(words) == 4
    tails = t.tensor([words[:2] + [2, words[2]], [1] + words[:3], [words[0], 0, words[1], words[2]]])
    seen = {}
    def generate(**kw):
        seen.update(kw)
        return t.cat([kw["input_ids"], tails], 1)
    monkeypatch.setattr(plain, "generate", generate)
    texts = inject_generate(plain, tok, ids, plain.get_input_embeddings(), POS, VECS, lambda o, v: v, max_new_tokens=7, do_sample=False, top_k=3)
    assert texts == [tok.decode(words[:2]), "", tok.decode(words[:3])]  # cut at the first eos id; special tokens (the pad) dropped
    assert t.equal(seen["input_ids"], t.tensor([ids] * 3)) and t.equal(seen["attention_mask"], t.ones(3, len(ids), dtype=t.long))
    assert seen["eos_token_id"] == [1, 2] and seen["pad_token_id"] == 0  # the generation config's eos and the tokenizer's
    assert seen["max_new_tokens"] == 7 and seen["do_sample"] is False and seen["top_k"] == 3
    monkeypatch.setattr(tok, "pad_token_id", None)
    inject_generate(plain, tok, ids, plain.get_input_embeddings(), POS, VECS, lambda o, v: v, **GREEDY)
    assert seen["pad_token_id"] == 1  # an eos id when the tokenizer has no pad

# single_token_marker

def test_single_token_marker_tiny(tok):
    render = olens_render(tok, 1)
    with pytest.raises(ValueError, match="no character in the range"):
        single_token_marker(tok, render)  # no U+3200..U+33FF character is one token on the Llama-2 tokenizer
    (z,), rendered = tok.encode("Z", add_special_tokens=False), render("Z")
    assert tok.convert_ids_to_tokens(z) == "▁Z" and z not in rendered and [tok.decode([x]) for x in rendered].count("Z") == 1  # sentencepiece encodes a lone 'Z' word-initially; the render holds the bare 'Z', another id with the same decode
    with pytest.raises(ValueError, match="no character in the range"):
        single_token_marker(tok, render, chars="ZQXJ")  # ids are compared, not decoded strings
    spaced = lambda char: to_ids([{"role": "user", "content": OLENS_PROMPT.format(layer=1, char=f" {char}")}], tok, add_generation_prompt=True)  # a space before the marker puts the word-initial token in the render
    char, tid, rendered, slot = single_token_marker(tok, spaced, chars="ZQXJ")
    assert (char, tid) == ("Z", z) and rendered == spaced("Z") and rendered[slot] == tid and rendered.count(tid) == 1 and rendered[0] == tok.bos_token_id
    assert [tok.decode([rendered[slot - 1]]), tok.decode([rendered[slot + 1]])] == [">", "</"]

class MarkerTok:
    """A, B and C are one token each (their code points) and D is two; the render below merges A away, repeats B and keeps C once."""
    def encode(self, s, add_special_tokens=True):
        return [1, 2] if s == "D" else [ord(s)]

RENDERS = {"A": [10, 11, 12], "B": [10, ord("B"), 11, ord("B")], "C": [10, ord("C"), 11], "D": [10, 1, 2, 11]}

def test_single_token_marker_skips_merged_repeated_and_multi_token_chars():
    assert single_token_marker(MarkerTok(), RENDERS.__getitem__, chars="DABC") == ("C", ord("C"), [10, ord("C"), 11], 1)
    assert single_token_marker(MarkerTok(), RENDERS.__getitem__, chars=iter("CB")) == ("C", ord("C"), [10, ord("C"), 11], 1)  # any iterable, first survivor wins
    with pytest.raises(ValueError, match="no character in the range"):
        single_token_marker(MarkerTok(), RENDERS.__getitem__, chars="DAB")

def test_single_token_marker_default_range_is_the_enclosed_cjk_letters():
    asked = []
    with pytest.raises(ValueError):
        single_token_marker(MarkerTok(), lambda c: asked.append(c) or [10, 11], None)
    assert asked == [chr(c) for c in range(0x3200, 0x3400)]

@pytest.mark.hf
@pytest.mark.parametrize("name, expected, neighbors", [("Qwen/Qwen3-0.6B", ("㈎", 149705), (29, 522)), ("Qwen/Qwen3.6-27B", ("㈜", 158983), (29, 510))])
def test_single_token_marker_oracle_lens_contract(name, expected, neighbors):
    """The marker the oracle lens cards name, inside the carrier prompt for layer 44 rendered as the lens renders it."""
    tok = load_tokenizer(name)
    char, tid, rendered, slot = single_token_marker(tok, olens_render(tok, 44))
    assert (char, tid) == expected and rendered[slot] == tid and (rendered[slot - 1], rendered[slot + 1]) == neighbors and len(rendered) == 56
    assert tok.decode(list(neighbors)) == "></"
