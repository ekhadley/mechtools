from types import SimpleNamespace

import pytest
import torch as t
from transformers.cache_utils import DynamicCache

from mechtools import set_seed
from mechtools.sampling import *

V, EOS, EOS2, H, HD = 64, 7, 5, 2, 4

class Tok:
    eos_token_id = EOS

class GenCfg:
    eos_token_id = [EOS, EOS2]

class FakeModel:
    """A causal LM whose next token is a deterministic function of the position and the previous token, so any bookkeeping error in a sampler shows up as a wrong token. It keeps a real DynamicCache, so the samplers' cache manipulation runs against transformers' cache API.
    Each step ends the row with probability p_eos (alternating EOS and EOS2, the generation_config eos), never before position 4. Key entries record their position, so `violations` collects every step at which a row's attention mask was not a suffix of ones of length pos+1 or the unmasked cache did not hold positions 0..pos."""
    tokenizer = Tok()
    generation_config = GenCfg()

    def __init__(self, p_eos: float = 0.25, seed: int = 0):
        self.g, self.p_eos, self.calls, self.violations, self.n_eos = t.Generator().manual_seed(seed), p_eos, [], [], 0

    @staticmethod
    def rule(pos, prev):
        return (pos * 3 + prev) % 50 + 8  # never an eos id, never below 8

    def __call__(self, toks, return_type=None, past_key_values=None, use_cache=True, attention_mask=None, position_ids=None):
        B, S_new = toks.shape
        cache = past_key_values if past_key_values is not None else DynamicCache()
        S_old = cache.get_seq_length()
        pos = position_ids if position_ids is not None else t.arange(S_old, S_old + S_new)[None].expand(B, -1)
        k = t.zeros(B, H, S_new, HD)
        k[:, :, :, 0] = pos[:, None, :].float()
        cache.update(k, k.clone(), 0)
        if attention_mask is not None:
            for b in range(B):
                m, p = attention_mask[b], int(pos[b, -1])
                n = int(m.sum())
                if n != p + 1 or not t.equal(m[-n:], t.ones(n, dtype=m.dtype)):
                    self.violations.append(("mask", S_old, b, m.tolist(), p))
                if not t.equal(cache.layers[0].keys[b, 0, -n:, 0], t.arange(n).float()):
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
        return logits, cache

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

def check_samples(out: list[list[int]], n: int, new_toks: int):
    assert len(out) == n
    for row in out:
        assert row == EXP[:len(row)] and EOS not in row and EOS2 not in row and len(row) <= new_toks

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

@pytest.mark.parametrize("n,bs,new_toks,p_eos,quiet", [(7, 3, 10, 0.25, False), (2, 5, 10, 0.25, True), (5, 2, 6, 0.0, False), (1, 1, 6, 0.25, True), (20, 4, 9, 0.3, False)])
def test_sample_rolling(n, bs, new_toks, p_eos, quiet, capsys):
    m = FakeModel(p_eos=p_eos, seed=n)
    out = sample_rolling(m, PROMPT, n=n, batch_size=bs, new_toks=new_toks, quiet=quiet)
    check_samples(out, n, new_toks)
    assert m.violations == [], m.violations[:3]
    assert max(B for B, _ in m.calls) == min(bs, n)
    assert ("sampling" in capsys.readouterr().err) != quiet
    if p_eos == 0.0:
        assert all(len(r) == new_toks for r in out)

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
