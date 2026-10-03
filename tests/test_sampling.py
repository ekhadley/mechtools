from types import SimpleNamespace

import pytest
import torch as t
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM, Gemma3ForCausalLM, Gemma3TextConfig, GlmMoeDsaConfig, GlmMoeDsaForCausalLM, Qwen3_5ForCausalLM, Qwen3_5TextConfig, ZayaConfig, ZayaForCausalLM
from transformers.cache_utils import DynamicCache, DynamicIndexedLayer, DynamicLayer, DynamicSlidingWindowLayer, LinearAttentionAndFullAttentionLayer, LinearAttentionAndSlidingWindowAttentionLayer, LinearAttentionLayer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACache, DeepseekV4HCACache

from mechtools import set_seed
from mechtools.sampling import *

V, EOS, EOS2, H, HD = 64, 7, 5, 2, 4

class Tok:
    eos_token_id = EOS

class GenCfg:
    eos_token_id = [EOS, EOS2]

class FakeModel:
    """A causal LM whose next token is a deterministic function of the position and the previous token, so any bookkeeping error in a sampler shows up as a wrong token. It keeps a real DynamicCache, so the samplers' cache manipulation runs against transformers' cache API.
    Each step ends the row with probability p_eos (alternating EOS and EOS2, the generation_config eos), never before position 4. Key and value entries record their position, so `violations` collects every step at which a row's attention mask was not a suffix of ones of length pos+1 or the unmasked cache did not hold positions 0..pos, and every call whose mask was not exactly as wide as the cache plus the new tokens and as the longest row."""
    tokenizer = Tok()
    generation_config = GenCfg()

    def __init__(self, p_eos: float = 0.25, seed: int = 0):
        self.g, self.p_eos, self.calls, self.violations, self.n_eos = t.Generator().manual_seed(seed), p_eos, [], [], 0

    @staticmethod
    def rule(pos, prev):
        return (pos * 3 + prev) % 50 + 8  # never an eos id, never below 8

    def __call__(self, toks, return_type=None, past_key_values=None, use_cache=True, attention_mask=None, position_ids=None, logits_to_keep=0):
        B, S_new = toks.shape
        cache = past_key_values if past_key_values is not None else DynamicCache()
        S_old = cache.get_seq_length()
        pos = position_ids if position_ids is not None else t.arange(S_old, S_old + S_new)[None].expand(B, -1)
        k = t.zeros(B, H, S_new, HD)
        k[:, :, :, 0] = pos[:, None, :].float()
        cache.update(k, k.clone(), 0)
        if attention_mask is not None:
            if not attention_mask.shape[1] == S_old + S_new == int(pos.max()) + 1:
                self.violations.append(("width", S_old, attention_mask.shape[1], int(pos.max())))
            for b in range(B):
                m, p = attention_mask[b], int(pos[b, -1])
                n = int(m.sum())
                if n != p + 1 or not t.equal(m[-n:], t.ones(n, dtype=m.dtype)):
                    self.violations.append(("mask", S_old, b, m.tolist(), p))
                if not all(t.equal(kv[b, 0, -n:, 0], t.arange(n).float()) for kv in (cache.layers[0].keys, cache.layers[0].values)):
                    self.violations.append(("cache", S_old, b))
        logits = t.full((B, S_new, V), -1e4)
        for b in range(B):
            p, prev = int(pos[b, -1]), int(toks[b, -1])
            if p > 3 and t.rand(1, generator=self.g).item() < self.p_eos:
                self.n_eos += 1
                logits[b, -1, EOS if self.n_eos % 2 else EOS2] = 0.0
            else:
                logits[b, -1, self.rule(p, prev)] = 0.0
        self.calls.append((B, S_old))
        return logits[:, -logits_to_keep:], cache  # as the HF head does: the last logits_to_keep positions, or all of them at 0

class FakeHF(FakeModel):
    """FakeModel in the raw HF call shape."""
    def __call__(self, toks, past_key_values=None, use_cache=True):
        logits, cache = super().__call__(toks, past_key_values=past_key_values)
        return SimpleNamespace(logits=logits, past_key_values=cache)

PROMPT = t.tensor([[9, 10, 11, 12]])

def expected(prompt: t.Tensor, n: int = 64) -> list[int]:
    """The tokens FakeModel produces from the prompt when it never ends the row."""
    toks = prompt[0].tolist()
    for _ in range(n):
        toks.append(FakeModel.rule(len(toks) - 1, toks[-1]))
    return toks[prompt.shape[1]:]

EXP = expected(PROMPT)

def check_samples(out: list[list[int]], n: int, new_toks: int, exp: list[int] = EXP):
    assert len(out) == n
    for row in out:
        assert row == exp[:len(row)] and EOS not in row and EOS2 not in row and len(row) <= new_toks

class AsBridge:
    """A raw HF causal LM in the Bridge call shape."""
    tokenizer = Tok()

    def __init__(self, hf): self.hf = hf

    def __call__(self, toks, return_type=None, **kw):
        out = self.hf(toks, **kw)
        return (out.logits, out.past_key_values) if return_type == "logits_and_cache" else out.logits

class Checked:
    """Wraps a model in the Bridge call shape and checks every call stream_rolling makes. The mask must be exactly as wide as the longest row, and the cache one entry narrower, both in the length it reports and in its widest key tensor. `gap` is the largest difference between a row's logits and those of an uncached forward over the prompt and that row's own tokens; rows are followed by batch index, through the sampler's reorder_cache when it drops some. `widths` is the mask width of each call. `eos`, when given, replaces the model's eos ids, so that rows end at varied lengths."""
    def __init__(self, model, prompt, eos=None):
        self.model, self.tokenizer, self.prompt, self.gap, self.widths = model, model.tokenizer, prompt[0, :-1].tolist(), 0.0, []
        self.generation_config = model.generation_config if eos is None else SimpleNamespace(eos_token_id=list(eos))

    def __call__(self, toks, return_type=None, **kw):
        if "attention_mask" not in kw:  # the prefill
            logits, self.cache = self.model(toks, return_type=return_type, **kw)
            assert logits.shape[1] == 1  # the head ran on one position, not on every prompt token of every row
            def reorder(keep, hf_reorder=self.cache.reorder_cache):
                hf_reorder(keep)
                self.rows = [self.rows[i] for i in keep.tolist()]
            self.cache.reorder_cache, self.rows = reorder, [[]] * len(toks)
            return logits, self.cache
        S = kw["attention_mask"].shape[1] - 1
        assert S == int(kw["position_ids"].max()) == self.cache.get_seq_length() == max(l.keys.shape[2] for l in self.cache.layers if isinstance(l, DynamicLayer))
        self.widths.append(S + 1)
        logits, _ = self.model(toks, return_type=return_type, **kw)
        for b, (pos, tok) in enumerate(zip(kw["position_ids"][:, 0].tolist(), toks[:, 0].tolist())):
            self.rows[b] = [tok] if pos == len(self.prompt) else self.rows[b] + [tok]
            ref = self.model(t.tensor([self.prompt + self.rows[b]]), return_type="logits")[0, -1]
            self.gap = max(self.gap, (logits[b, -1] - ref).abs().max().item())
        return logits, self.cache

def check_rolling(model, prompt: t.Tensor, eos, n: int, new_toks: int) -> tuple[Checked, list[list[int]]]:
    """sample_rolling at batch 4 through Checked: n samples, every call's logits equal to an uncached forward, the mask never wider than the prompt less its last token plus new_toks."""
    m = Checked(model, prompt, eos)
    out = sample_rolling(m, prompt, n=n, batch_size=4, new_toks=new_toks, quiet=True)
    assert len(out) == n and m.gap < 1e-5 and max(m.widths) <= prompt.shape[1] - 1 + new_toks
    return m, out

TINY = dict(vocab_size=V, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
RANDOM_MODELS = {  # built from a config, so nothing is downloaded. The outer layers' cache layers: a sliding window of 8, which keeps 7 entries, and linear attention, which holds a conv and a recurrent state
    DynamicSlidingWindowLayer: lambda: Gemma3ForCausalLM(Gemma3TextConfig(**TINY, sliding_window=8, layer_types=["sliding_attention", "full_attention", "sliding_attention"])),
    LinearAttentionLayer: lambda: Qwen3_5ForCausalLM(Qwen3_5TextConfig(**TINY, linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2, linear_num_value_heads=4, layer_types=["linear_attention", "full_attention", "linear_attention"])),
}
UNSUPPORTED = [  # also built from configs: cache layers that pass the isinstance checks of a restart but hold per-row state beyond keys, values, conv and recurrent states. The expected layer types, and the model: sparse attention with indexer keys in every layer; hybrid layers holding keys and values beside a conv and a recurrent state; DeepSeek-V4's compressed-attention caches with compressor buffers and a batch-wide entry count
    ([DynamicIndexedLayer] * 3, lambda: GlmMoeDsaForCausalLM(GlmMoeDsaConfig(vocab_size=V, hidden_size=16, intermediate_size=32, moe_intermediate_size=8, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2, n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1, kv_lora_rank=8, q_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4, index_topk=4, index_head_dim=4, index_n_heads=2, first_k_dense_replace=1, max_position_embeddings=128))),
    ([LinearAttentionAndFullAttentionLayer, LinearAttentionAndSlidingWindowAttentionLayer, LinearAttentionAndFullAttentionLayer], lambda: ZayaForCausalLM(ZayaConfig(vocab_size=V, hidden_size=16, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=8, moe_intermediate_size=8, num_experts=4, router_hidden_size=8, max_position_embeddings=128, sliding_window=8, eos_token_id=1, layer_types=["hybrid", "hybrid_sliding", "hybrid"]))),
    ([DeepseekV4HCACache, DeepseekV4CSACache, DeepseekV4HCACache], lambda: DeepseekV4ForCausalLM(DeepseekV4Config(vocab_size=V, hidden_size=16, moe_intermediate_size=8, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=8, q_lora_rank=8, num_experts_per_tok=2, n_routed_experts=4, n_shared_experts=1, sliding_window=8, o_groups=2, o_lora_rank=4, index_n_heads=2, index_head_dim=4, index_topk=4, hc_mult=2, compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4}, layer_types=["heavily_compressed_attention", "compressed_sparse_attention", "heavily_compressed_attention"], max_position_embeddings=128, num_nextn_predict_layers=0))),
]

def test_eos_ids():
    assert eos_ids(FakeModel()) == {EOS, EOS2}
    class Int: generation_config = type("G", (), {"eos_token_id": 3})(); tokenizer = Tok()
    assert eos_ids(Int()) == {EOS, 3}
    class NoCfg: tokenizer = Tok()
    assert eos_ids(NoCfg()) == {EOS} and eos_ids(NoCfg(), Tok()) == {EOS} and eos_ids(NoCfg(), type("T", (), {"eos_token_id": 2})()) == {2}
    class Bridge:  # TransformerBridge.__getattr__ falls through to the HF model it wraps
        tokenizer, original_model = Tok(), FakeModel()
        def __getattr__(self, name): return getattr(self.original_model, name)
    assert eos_ids(Bridge()) == {EOS, EOS2}
    class NoEos: tokenizer = type("T", (), {"eos_token_id": None})()
    with pytest.raises(ValueError, match="never stop"):
        eos_ids(NoEos())

def test_stream_toks():
    assert list(stream_toks(FakeModel(p_eos=0.0), PROMPT, new_toks=6)) == EXP[:6]
    m = FakeModel(p_eos=1.0)  # ends at the first position past 3: the prompt's last position is 3, so the first sample is the rule and the second is EOS
    assert list(stream_toks(m, PROMPT, new_toks=6)) == EXP[:1]
    m = FakeModel(p_eos=1.0); m.n_eos = 1  # the next eos drawn is EOS2, the generation_config one
    assert list(stream_toks(m, PROMPT, new_toks=6)) == EXP[:1]
    assert list(stream_toks_hf(FakeHF(p_eos=0.0), Tok(), PROMPT, new_toks=6)) == EXP[:6]
    assert list(stream_toks_hf(FakeHF(p_eos=1.0), Tok(), PROMPT, new_toks=6)) == EXP[:1]

@pytest.mark.parametrize("p_eos", [0.0, 0.3])
def test_sample_batch(p_eos, capsys):
    out = sample_batch(FakeModel(p_eos=p_eos), PROMPT, n=6, new_toks=12, quiet=True)
    check_samples(out, 6, 12)
    assert capsys.readouterr().err == ""  # quiet hides the bar
    if p_eos == 0.0:
        assert [len(r) for r in out] == [12] * 6
    else:
        assert len({len(r) for r in out}) > 1 and min(len(r) for r in out) < 12

def test_sample_batch_stops_once_every_row_has_ended(capsys):
    m = FakeModel(p_eos=1.0)
    assert sample_batch(m, PROMPT, n=4, new_toks=50) == [EXP[:1]] * 4 and len(m.calls) == 2  # the step that produced the last terminator is the last one
    assert "sampling" in capsys.readouterr().err

@pytest.mark.parametrize("n,bs,new_toks,p_eos,quiet", [(7, 3, 10, 0.25, False), (2, 5, 10, 0.25, True), (5, 2, 6, 0.0, False), (1, 1, 6, 0.25, True), (20, 4, 9, 0.3, False), (9, 3, 1, 0.25, True)])
def test_sample_rolling(n, bs, new_toks, p_eos, quiet, capsys):
    m = FakeModel(p_eos=p_eos, seed=n)
    out = sample_rolling(m, PROMPT, n=n, batch_size=bs, new_toks=new_toks, quiet=quiet)
    check_samples(out, n, new_toks)
    assert m.violations == [], m.violations[:3]
    assert max(B for B, _ in m.calls) == min(bs, n)
    assert ("sampling" in capsys.readouterr().err) != quiet
    if p_eos == 0.0:
        assert all(len(r) == new_toks for r in out)

@pytest.mark.parametrize("n,bs,new_toks,p_eos", [(300, 4, 10, 0.25), (1000, 16, 40, 0.1), (160, 4, 12, 0.0)])  # the last: every row runs to new_toks, so all restart together
def test_rolling_cache_is_as_wide_as_the_longest_row(n, bs, new_toks, p_eos):
    m = FakeModel(p_eos=p_eos)
    check_samples(sample_rolling(m, PROMPT, n=n, batch_size=bs, new_toks=new_toks, quiet=True), n, new_toks)
    assert m.violations == [], m.violations[:3]
    assert len(m.calls) > 300 and max(S for _, S in m.calls) == PROMPT.shape[1] - 1 + new_toks - 1  # over a long run the widest cache is the prompt less its last token plus the new_toks - 1 tokens of a row about to reach the cap

def test_sample_rolling_two_token_prompt():
    prompt = t.tensor([[9, 10]])  # the prefill, and so each restart, is a single cache entry
    m = FakeModel(p_eos=0.25)
    check_samples(sample_rolling(m, prompt, n=40, batch_size=4, new_toks=8, quiet=True), 40, 8, expected(prompt))
    assert m.violations == [], m.violations[:3]

def test_sample_rolling_random_configs():
    g, empty = t.Generator().manual_seed(0), 0
    for _ in range(60):
        n, bs, new_toks, prompt_len = (int(t.randint(lo, hi, (1,), generator=g)) for lo, hi in ((1, 60), (1, 9), (1, 14), (2, 10)))
        prompt, m = t.arange(9, 9 + prompt_len)[None], FakeModel(p_eos=float(t.rand(1, generator=g)) / 2, seed=n)
        out = sample_rolling(m, prompt, n=n, batch_size=bs, new_toks=new_toks, quiet=True)
        check_samples(out, n, new_toks, expected(prompt))
        assert m.violations == [], (n, bs, new_toks, prompt_len, m.violations[:3])
        empty += sum(not row for row in out)
    assert empty > 20  # a prompt of five tokens or more can end a row at its first sampled token, which restarts it a step after it started

def test_stream_rolling_yields_as_samples_finish():
    full = FakeModel(p_eos=0.25, seed=3)
    out = sample_rolling(full, PROMPT, n=7, batch_size=3, new_toks=10, quiet=True)
    m = FakeModel(p_eos=0.25, seed=3)
    g = stream_rolling(m, PROMPT, n=7, batch_size=3, new_toks=10, quiet=True)
    assert m.calls == []  # nothing runs until consumed
    first = next(g)
    assert 0 < len(m.calls) < len(full.calls) and first == out[0]
    assert [first, *g] == out and m.calls == full.calls

def test_stream_rolling_closes_the_bar_when_abandoned(capsys):
    for _ in stream_rolling(FakeModel(), PROMPT, n=7, batch_size=3, new_toks=10): break
    err = capsys.readouterr().err
    assert "sampling" in err and err.endswith("\n")  # the bar's close, run when the dropped generator exits its with block, ends the line

@pytest.mark.hf
def test_bridge_sampling(tiny_bridge):
    ids = t.tensor([tiny_bridge.tokenizer.encode("Hello there")])
    set_seed(0)
    streamed = list(stream_toks(tiny_bridge, ids, new_toks=5))
    assert len(streamed) <= 5 and all(isinstance(i, int) for i in streamed)
    set_seed(0)
    a = sample_batch(tiny_bridge, ids, n=3, new_toks=6, quiet=True)
    set_seed(0)
    b = sample_batch(tiny_bridge, ids, n=3, new_toks=6, quiet=True)
    assert a == b and len(a) == 3 and all(len(r) <= 6 for r in a)
    rolled = sample_rolling(tiny_bridge, ids, n=3, batch_size=2, new_toks=6)
    assert len(rolled) == 3 and all(len(r) <= 6 and not (set(r) & eos_ids(tiny_bridge)) for r in rolled)
    set_seed(0)
    assert sample_rolling(tiny_bridge, ids, n=3, batch_size=3, new_toks=6, quiet=True) == a  # no row ends before new_toks, so the rolling batch draws the same random numbers as sample_batch
    set_seed(1)
    c = sample_rolling(tiny_bridge, ids, n=9, batch_size=2, new_toks=6, quiet=True)
    set_seed(1)
    assert sample_rolling(tiny_bridge, ids, n=9, batch_size=2, new_toks=6, quiet=True) == c

@pytest.mark.hf
@pytest.mark.parametrize("eos", [None, range(0, 32000, 8)])  # the model's own eos, which is all but never sampled, so every row runs to new_toks and all four restart together; and an eighth of the vocab, so rows end at varied lengths
def test_rolling_matches_uncached_forward_on_the_bridge(tiny_bridge, eos):
    set_seed(0)
    ids = t.tensor([tiny_bridge.tokenizer.encode("Hello there, how are")])
    m, out = check_rolling(tiny_bridge, ids, eos, n=160, new_toks=12)
    assert len(m.widths) > 200 and max(m.widths) == ids.shape[1] - 1 + 12 and (eos is None or len({len(r) for r in out}) > 4)  # 17 wide over 480 and 237 calls

@pytest.mark.parametrize("layer,prompt_len", [(DynamicSlidingWindowLayer, 4), (DynamicSlidingWindowLayer, 8), (DynamicSlidingWindowLayer, 9), (DynamicSlidingWindowLayer, 12), (LinearAttentionLayer, 4)])  # against the window of 8: rows that cross it while generating, a prefill one short of it, a prefill that fills it, a prefill past it
def test_rolling_matches_uncached_forward_on_other_cache_layers(layer, prompt_len):
    set_seed(0)
    model = AsBridge(RANDOM_MODELS[layer]().eval().requires_grad_(False))
    m, out = check_rolling(model, t.randint(8, V, (1, prompt_len)), range(1, V, 6), n=40, new_toks=16)
    assert [type(l) for l in m.cache.layers] == [layer, DynamicLayer, layer] and len({len(r) for r in out}) > 4

@pytest.mark.parametrize("types,build", UNSUPPORTED, ids=["glm_moe_dsa", "zaya", "deepseek_v4"])
def test_stream_rolling_refuses_cache_layers_it_cannot_restart(types, build):
    """The TypeError names the layer and comes right after the prefill, before any row is sampled."""
    prompt = t.randint(8, V, (1, 6))
    m = Checked(AsBridge(build().eval().requires_grad_(False)), prompt, [1])
    with pytest.raises(TypeError, match=f"{types[0].__name__} cache"):
        next(stream_rolling(m, prompt, n=8, batch_size=4, new_toks=8, quiet=True))
    assert [type(l) for l in m.cache.layers] == types and m.widths == []

def check_rollouts(out: dict[int, list[int]], toks: t.Tensor, cuts: list[int], max_len: int):
    """Every cut came back once, and each sample continues its cut's prefix as FakeModel's rule dictates, holds no eos, and stays within the cap."""
    assert sorted(out) == list(range(len(cuts)))
    for i, row in out.items():
        assert row == expected(toks[:, :cuts[i]], max_len)[:len(row)] and EOS not in row and EOS2 not in row and len(row) <= max_len - cuts[i]

@pytest.mark.parametrize("cuts,bs,max_len,p_eos,quiet", [([4, 1, 3, 4, 2, 3, 1], 3, 12, 0.25, False), ([4, 4, 4], 5, 10, 0.0, True), ([2, 3], 1, 4, 0.0, False), ([1], 4, 5, 0.0, True), (list(range(1, 5)) * 5, 4, 14, 0.3, True)])
def test_stream_rollouts(cuts, bs, max_len, p_eos, quiet, capsys):
    m = FakeModel(p_eos=p_eos, seed=len(cuts))
    out = dict(stream_rollouts(m, PROMPT, cuts, bs, max_len, quiet=quiet))
    check_rollouts(out, PROMPT, cuts, max_len)
    assert m.violations == [], m.violations[:3]
    assert m.calls[0] == (1, 0) and max(B for B, _ in m.calls[1:]) == min(bs, len(cuts))  # one forward over the prompt less its last token, then batches
    assert ("rollouts" in capsys.readouterr().err) != quiet
    if p_eos == 0.0:
        assert all(len(out[i]) == max_len - cuts[i] for i in out)  # every row runs to the cap

def test_stream_rollouts_order():
    """Batches go from the longest cut down, and within a batch the rows that finish first come first."""
    m = FakeModel(p_eos=0.0)
    assert [i for i, _ in stream_rollouts(m, PROMPT, [1, 4, 2, 4, 3], 2, 8, quiet=True)] == [1, 3, 4, 2, 0]  # the two cuts at 4 (four tokens each), then the cut at 3 before the one at 2, then the cut at 1 alone
    assert [B for B, _ in m.calls] == [1] + [2] * 4 + [2] * 5 + [1] + [1] * 7  # the second batch loses its cut-3 row after five steps and runs one more; the last batch is one row for seven steps
    assert m.violations == []

def test_stream_rollouts_random_configs():
    g = t.Generator().manual_seed(0)
    for _ in range(60):
        prompt_len, n, bs, extra = (int(t.randint(lo, hi, (1,), generator=g)) for lo, hi in ((2, 10), (1, 40), (1, 9), (1, 13)))
        prompt, cuts = t.arange(9, 9 + prompt_len)[None], (t.randint(1, prompt_len + 1, (n,), generator=g)).tolist()
        m = FakeModel(p_eos=float(t.rand(1, generator=g)) / 2, seed=n)
        out = dict(stream_rollouts(m, prompt, cuts, bs, prompt_len + extra, quiet=True))
        check_rollouts(out, prompt, cuts, prompt_len + extra)
        assert m.violations == [], (prompt_len, cuts, bs, extra, m.violations[:3])

def test_stream_rollouts_rejects_bad_cuts():
    for cuts, max_len in (([0, 2], 10), ([5], 10), ([2, 4], 4)):  # below 1, past the end, not below max_len
        with pytest.raises(ValueError, match="cuts"):
            next(stream_rollouts(FakeModel(), PROMPT, cuts, 2, max_len, quiet=True))

@pytest.mark.parametrize("layer", [DynamicSlidingWindowLayer, LinearAttentionLayer])
def test_stream_rollouts_refuses_other_cache_layers(layer):
    model = AsBridge(RANDOM_MODELS[layer]().eval().requires_grad_(False))
    with pytest.raises(TypeError, match="cannot be cut back"):
        next(stream_rollouts(model, t.randint(8, V, (1, 6)), [3, 6], 2, 20, quiet=True))

class CheckedRollouts:
    """Wraps a model in the Bridge call shape and checks every step of stream_rollouts: the mask as wide as the cache plus one and as the longest row's position plus one, and each row's logits against an uncached forward over the tokens it holds, which are its cut's prefix (the position of the first token it feeds, plus one, tokens of toks) and the tokens fed since. gap is the largest difference seen. eos, when given, replaces the model's eos ids."""
    def __init__(self, model, toks, eos=None):
        self.model, self.tokenizer, self.toks, self.gap, self.cache = model, model.tokenizer, toks[0].tolist(), 0.0, None
        self.generation_config = model.generation_config if eos is None else SimpleNamespace(eos_token_id=list(eos))

    def __call__(self, toks, return_type=None, past_key_values=None, **kw):
        if past_key_values is None:  # the one forward over the text
            return self.model(toks, return_type=return_type, **kw)
        if past_key_values is not self.cache:  # a new batch: each row holds its cut's prefix
            self.cache, self.rows = past_key_values, [self.toks[:c] for c in kw["position_ids"][:, 0].tolist()]
            def reorder(keep, hf_reorder=self.cache.reorder_cache):
                hf_reorder(keep)
                self.rows = [self.rows[i] for i in keep.tolist()]
            self.cache.reorder_cache = reorder
        S = kw["attention_mask"].shape[1] - 1
        assert S == int(kw["position_ids"].max()) == self.cache.get_seq_length() == max(l.keys.shape[2] for l in self.cache.layers)
        logits, cache = self.model(toks, return_type=return_type, past_key_values=past_key_values, **kw)
        assert cache is self.cache
        for b, tok in enumerate(toks[:, 0].tolist()):
            self.rows[b].append(tok)
            ref = self.model(t.tensor([self.rows[b]]), return_type="logits")[0, -1]
            self.gap = max(self.gap, (logits[b, -1] - ref).abs().max().item())
        return logits, cache

@pytest.mark.hf
def test_rollouts_match_uncached_forward_on_the_bridge(tiny_bridge):
    set_seed(0)
    ids = t.tensor([tiny_bridge.tokenizer.encode("Hello there, how are you")])
    cuts = list(range(1, ids.shape[1] + 1)) * 3  # every cut three times, including the first token alone and the whole text
    m = CheckedRollouts(tiny_bridge, ids, range(0, 32000, 8))  # an eighth of the vocab ends a row, so rows end at varied lengths
    out = dict(stream_rollouts(m, ids, cuts, 4, ids.shape[1] + 10, quiet=True))
    assert sorted(out) == list(range(len(cuts))) and m.gap < 1e-5 and len({len(r) for r in out.values()}) > 4
    assert all(len(out[i]) <= ids.shape[1] + 10 - cuts[i] and not (set(out[i]) & set(range(0, 32000, 8))) for i in out)
    set_seed(0)
    a = sample_batch(tiny_bridge, ids, n=3, new_toks=6, quiet=True)
    set_seed(0)
    b = dict(stream_rollouts(tiny_bridge, ids, [ids.shape[1]] * 3, 3, ids.shape[1] + 6, quiet=True))
    assert [b[i] for i in range(3)] == a  # three rollouts cut at the end are three samples of the prompt, drawing the same random numbers
