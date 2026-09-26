from types import SimpleNamespace

import pytest
import torch as t
from transformers.cache_utils import DynamicCache

from mechtools.sampling import eos_ids, sample_batch, sample_rolling, stream_toks, stream_toks_hf

V, EOS, END_OF_TURN, FIRST = 512, 1, 106, 200  # gemma-3-it's shape: the tokenizer's eos, and a turn terminator that only generation_config lists
PROMPT = t.tensor([[7, 8, 9]])
P = PROMPT.shape[1]

class Fake:
    """A scripted model in the TransformerBridge call shape, or the raw HF one with hf=True, over a real DynamicCache. Batch slot b emits FIRST, FIRST + 1, ... and, if ends(b) is not None, terminator(b) as its ends(b)th generated token. Every prob is 0 or 1, so multinomial is deterministic."""
    def __init__(self, gen_eos=(EOS, END_OF_TURN), tok_eos=EOS, hf=False, ends=lambda slot: 2 + slot, terminator=lambda slot: EOS if slot % 2 == 0 else END_OF_TURN):
        gen_cfg = SimpleNamespace(eos_token_id=list(gen_eos) if isinstance(gen_eos, tuple) else gen_eos)
        self.tokenizer = SimpleNamespace(eos_token_id=tok_eos)
        if hf: self.generation_config = gen_cfg  # a raw HF model carries its own
        else: self.original_model = SimpleNamespace(generation_config=gen_cfg)  # a Bridge delegates the attribute to the HF model it wraps
        self.hf, self.ends, self.terminator, self.calls = hf, ends, terminator, 0

    def __getattr__(self, name):  # TransformerBridge.__getattr__ falls through to original_model
        if name != "original_model" and "original_model" in self.__dict__: return getattr(self.original_model, name)
        raise AttributeError(name)

    def __call__(self, toks, past_key_values=None, use_cache=True, return_type=None, attention_mask=None, position_ids=None):
        self.calls += 1
        cache = DynamicCache() if past_key_values is None else past_key_values
        B, T = toks.shape
        for layer in range(2):
            cache.update(t.zeros(B, 1, T, 4), t.zeros(B, 1, T, 4), layer)
        pos = position_ids[:, -1] if position_ids is not None else t.full((B,), cache.get_seq_length() - 1)  # position of the last token fed
        logits = t.full((B, T, V), -1e4)
        for b, g in enumerate((pos + 1 - P).tolist()):  # g: index of the token being generated, 0 for the first after the prompt
            logits[b, -1, self.terminator(b) if g == self.ends(b) else FIRST + g] = 1e4
        return SimpleNamespace(logits=logits, past_key_values=cache) if self.hf else (logits, cache)

def test_eos_ids():
    assert eos_ids(Fake()) == [EOS, END_OF_TURN]
    assert eos_ids(Fake(gen_eos=EOS)) == [EOS]  # an int, the tokenizer's already in it
    assert eos_ids(Fake(gen_eos=[3], tok_eos=7)) == [3, 7]
    assert eos_ids(Fake(gen_eos=None, tok_eos=5)) == [5]
    assert eos_ids(SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=5))) == [5]  # no generation_config at all
    hf = Fake(hf=True)
    assert eos_ids(hf, hf.tokenizer) == [EOS, END_OF_TURN]
    with pytest.raises(ValueError, match="never stop"):
        eos_ids(Fake(gen_eos=None, tok_eos=None))

def test_sample_batch_cuts_each_row_at_its_own_terminator(capsys):
    fake = Fake(ends=lambda slot: 2 + slot if slot < 4 else None)  # rows 0-3 end with EOS, END_OF_TURN, EOS, END_OF_TURN after 2, 3, 4, 5 tokens; row 4 never ends
    rows = sample_batch(fake, PROMPT, n=5, new_toks=8)
    assert rows == [[200, 201], [200, 201, 202], [200, 201, 202, 203], [200, 201, 202, 203, 204], [200, 201, 202, 203, 204, 205, 206, 207]]
    assert fake.calls == 8  # one live row keeps the batch stepping to new_toks
    assert "sampling" in capsys.readouterr().err

def test_sample_batch_stops_stepping_once_every_row_has_ended(capsys):
    fake = Fake()
    assert sample_batch(fake, PROMPT, n=3, new_toks=50, quiet=True) == [[200, 201], [200, 201, 202], [200, 201, 202, 203]]
    assert fake.calls == 5  # the longest row's 4 tokens and the step that produced its terminator
    assert capsys.readouterr().err == ""  # quiet hides the bar

def test_sample_rolling_restarts_rows_and_returns_n_samples(capsys):
    fake = Fake(ends=lambda slot: 2 + slot if slot < 2 else None)  # slot 0 ends after 2 tokens, slot 1 after 3, slot 2 runs to new_toks
    out = sample_rolling(fake, PROMPT, n=7, batch_size=3, new_toks=5)
    assert len(out) == 7 and all(row == list(range(FIRST, FIRST + len(row))) for row in out)  # restarted rows get the prompt's position ids again
    lens = sorted(len(row) for row in out)
    assert lens[0] == 2 and lens[-1] == 5 and {2, 3} <= set(lens)  # cut at EOS, at END_OF_TURN, and at new_toks
    assert "sampling" in capsys.readouterr().err
    assert sample_rolling(fake, PROMPT, n=4, batch_size=2, new_toks=3, quiet=True) == [[200, 201], [200, 201, 202], [200, 201], [200, 201, 202]]
    assert capsys.readouterr().err == ""

def test_stream_toks_stop_at_either_terminator():
    assert list(stream_toks(Fake(), PROMPT)) == [200, 201]
    assert list(stream_toks(Fake(terminator=lambda slot: END_OF_TURN), PROMPT)) == [200, 201]
    assert list(stream_toks(Fake(ends=lambda slot: None), PROMPT, new_toks=4)) == [200, 201, 202, 203]
    hf = Fake(hf=True, terminator=lambda slot: END_OF_TURN)
    assert list(stream_toks_hf(hf, hf.tokenizer, PROMPT)) == [200, 201]
