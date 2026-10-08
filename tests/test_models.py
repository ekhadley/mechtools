import functools
from types import SimpleNamespace

import httpx
import pytest
import torch as t
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError, RemoteEntryNotFoundError
from transformers import AutoModelForCausalLM, DeepseekV4Config, DeepseekV4ForCausalLM

import mechtools.models as mm
from conftest import TINY_MODEL, load_tokenizer
from mechtools.hooks import add_bias_hook
from mechtools.models import is_adapter_repo, load_bridge, load_hf_model

def hub_error(cls, status: int, msg: str):
    return cls(msg, response=httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/x")))

def test_is_adapter_repo_local(tmp_path):
    assert not is_adapter_repo(str(tmp_path))
    (tmp_path / "adapter_config.json").write_text("{}")
    assert is_adapter_repo(str(tmp_path))

def test_is_adapter_repo_hub_only_treats_a_missing_file_as_no(monkeypatch):
    def download(repo_id, filename):
        assert filename == "adapter_config.json"
        if repo_id == "org/adapter": return "/cache/adapter_config.json"
        if repo_id == "org/base": raise hub_error(RemoteEntryNotFoundError, 404, "entry not found")
        if repo_id == "org/offline-base": raise LocalEntryNotFoundError("not in the cache and outgoing traffic is disabled")
        if repo_id == "org/gated": raise hub_error(GatedRepoError, 403, "gated repo")
        raise httpx.ConnectError("no network")
    monkeypatch.setattr(mm, "hf_hub_download", download)
    assert is_adapter_repo("org/adapter") and not is_adapter_repo("org/base") and not is_adapter_repo("org/offline-base")
    with pytest.raises(GatedRepoError):  # peft would have folded this into "can't find adapter_config.json"
        is_adapter_repo("org/gated")
    with pytest.raises(httpx.ConnectError):
        is_adapter_repo("org/nowhere")

def test_load_hf_model_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(mm, "is_adapter_repo", lambda mid: mid.endswith("-lora"))
    monkeypatch.setattr(mm.AutoModelForCausalLM, "from_pretrained", lambda mid, **kw: calls.append(("auto", mid)) or SimpleNamespace(id=mid))
    monkeypatch.setattr(mm.PeftConfig, "from_pretrained", lambda mid: calls.append(("cfg", mid)) or SimpleNamespace(base_model_name_or_path="org/base"))
    monkeypatch.setattr(mm.PeftModel, "from_pretrained", lambda base, mid: calls.append(("peft", base.id, mid)) or SimpleNamespace(merge_and_unload=lambda: SimpleNamespace(id=f"{base.id}+{mid}")))
    assert load_hf_model("org/base").id == "org/base" and calls == [("auto", "org/base")]
    calls.clear()
    assert load_hf_model("org/x-lora").id == "org/base+org/x-lora" and calls == [("cfg", "org/x-lora"), ("auto", "org/base"), ("peft", "org/base", "org/x-lora")]
    calls.clear()
    assert load_hf_model("org/x-lora", parent_model_id="org/other").id == "org/other+org/x-lora" and calls[0] == ("auto", "org/other")

def test_loaders_pass_from_pretrained_kwargs(monkeypatch):
    """Extra kwargs reach AutoModelForCausalLM.from_pretrained, the base model's on an adapter repo, through load_hf_model and load_bridge; peft gets none of them."""
    seen = []
    def from_pretrained(mid, **kw):
        seen.append(kw)
        model = t.nn.Linear(1, 1)
        model.config = SimpleNamespace(name_or_path=mid)
        return model
    monkeypatch.setattr(mm, "is_adapter_repo", lambda mid: mid.endswith("-lora"))
    monkeypatch.setattr(mm.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(mm.PeftConfig, "from_pretrained", lambda mid: SimpleNamespace(base_model_name_or_path="org/base"))
    monkeypatch.setattr(mm.PeftModel, "from_pretrained", lambda base, mid: SimpleNamespace(merge_and_unload=lambda: base))
    monkeypatch.setattr(mm.TransformerBridge, "boot_transformers", lambda name, **kw: t.nn.Linear(1, 1))
    extra = {"quantization_config": "q", "max_memory": {0: "1GiB"}}
    load_hf_model("org/base", **extra)
    load_hf_model("org/x-lora", **extra)
    load_bridge("org/base", device_map="cpu", **extra)
    assert seen == [{"dtype": t.bfloat16, "device_map": "auto", **extra}] * 2 + [{"dtype": t.bfloat16, "device_map": "cpu", **extra}]

@pytest.mark.hf
def test_load_bridge(tiny_bridge):
    model = tiny_bridge
    assert not model.training and not any(p.requires_grad for p in model.parameters()) and model.tokenizer is not None
    assert next(model.parameters()).dtype == t.float32 and model.generation_config.eos_token_id is not None
    ids = t.tensor([model.tokenizer.encode("Hello there")])
    logits = model(ids)
    assert logits.shape == (1, ids.shape[1], model.cfg.d_vocab)

@pytest.fixture(scope="module")
def v4_dir(tmp_path_factory):
    """A saved random DeepSeek-V4 with Flash's 4 residual streams, one layer of each attention type, a hash-routed first MoE layer with a random expert table, and TINY_MODEL's tokenizer. Weights are drawn wide (std 0.3) so that layers and edits move the logits; index_topk covers every compressed entry, since tied indexer scores would otherwise make a cached forward pick other entries than a full one."""
    tok = load_tokenizer(TINY_MODEL)
    t.manual_seed(0)
    cfg = DeepseekV4Config(vocab_size=len(tok), hidden_size=16, moe_intermediate_size=8, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=8, q_lora_rank=8, num_experts_per_tok=2, n_routed_experts=4, n_shared_experts=1, sliding_window=8, o_groups=2, o_lora_rank=4, index_n_heads=2, index_head_dim=4, index_topk=64, hc_mult=4, compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4}, layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"], mlp_layer_types=["hash_moe", "moe", "moe"], max_position_embeddings=128, num_nextn_predict_layers=0)
    hf = DeepseekV4ForCausalLM(cfg)
    for p in hf.parameters():
        p.data.normal_(0, 0.3)
    for name, buf in hf.named_buffers():
        if name.endswith("tid2eid"):
            buf.copy_(t.randint(0, cfg.n_routed_experts, buf.shape))
    path = tmp_path_factory.mktemp("v4")
    hf.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)

@pytest.mark.hf
def test_load_bridge_deepseek_v4(v4_dir):
    """load_bridge on a saved DeepSeek-V4 picks the V4 adapter from config.architectures and matches an independently loaded HF model: logits on an unpadded batch, a KV-cached decode across the sliding window and compression boundaries, and a bias added at blocks.{l}.hook_in (the [batch, pos, streams, d] decoder-layer input) against the same bias in a pre-hook on the HF decoder layer."""
    ref = AutoModelForCausalLM.from_pretrained(v4_dir, dtype=t.float32).eval()
    model = load_bridge(v4_dir, dtype=t.float32, device_map="cpu")
    assert type(model.adapter).__name__ == "DeepSeekV4ArchitectureAdapter"
    toks = t.randint(3, ref.config.vocab_size, (2, 21))
    full = ref(toks).logits
    assert t.equal(model(toks), full)
    logits, past = model(toks[:, :9], return_type="logits_and_cache", use_cache=True)
    steps = [logits[:, -1]]
    for i in range(9, 20):
        logits, past = model(toks[:, i:i + 1], return_type="logits_and_cache", past_key_values=past, use_cache=True)
        steps.append(logits[:, -1])
    t.testing.assert_close(t.stack(steps, 1), full[:, 8:20], rtol=0, atol=1e-5)
    bias = t.randn(16)
    def pre_hook(module, args, kwargs):
        if args:
            return (args[0] + bias, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": kwargs["hidden_states"] + bias}
    handle = ref.model.layers[1].register_forward_pre_hook(pre_hook, with_kwargs=True)
    edited = ref(toks).logits
    handle.remove()
    assert (edited - full).abs().max() > 0.1
    assert t.equal(model.run_with_hooks(toks, fwd_hooks=[("blocks.1.hook_in", functools.partial(add_bias_hook, bias=bias))]), edited)

@pytest.mark.hf
def test_load_bridge_deepseek_v4_bf16(v4_dir):
    """load_bridge's default bf16 runs a DeepSeek-V4 forward (a raw HF V4 loaded in bf16 keeps its norms in fp32, and their fp32 outputs crash its bf16 attention projections) with the float32 model's top tokens."""
    model = load_bridge(v4_dir, device_map="cpu")
    toks = t.randint(3, model.cfg.d_vocab, (2, 21))
    logits = model(toks)
    assert next(model.parameters()).dtype == t.bfloat16 and logits.dtype == t.bfloat16 and logits.isfinite().all()
    ref = AutoModelForCausalLM.from_pretrained(v4_dir, dtype=t.float32).eval()
    assert (logits.argmax(-1) == ref(toks).logits.argmax(-1)).float().mean() >= 0.9
