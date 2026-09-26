from types import SimpleNamespace

import pytest
import torch as t

from mechtools.sampling import sample_batch, stop_ids_of, stream_toks, stream_toks_hf

class Scripted:
    """A model whose next token for row b at step s is script[b][s], as one-hot logits over a vocab of 16. The tokenizer's eos is 1 and generation_config also lists 5, like a chat model whose end-of-turn token is not the tokenizer's eos."""
    def __init__(self, script: list[list[int]]):
        self.script, self.step = script, 0
        self.tokenizer = SimpleNamespace(eos_token_id=1)
        self.generation_config = SimpleNamespace(eos_token_id=[1, 5])

    def logits(self, toks):
        logits = t.full((toks.shape[0], toks.shape[1], 16), -1e4)
        for b in range(toks.shape[0]):
            logits[b, -1, self.script[b][self.step]] = 1e4
        self.step += 1
        return logits

    def __call__(self, toks, return_type=None, past_key_values=None, use_cache=True):
        return self.logits(toks), "cache"

class ScriptedHF(Scripted):
    def __call__(self, toks, past_key_values=None, use_cache=True):
        return SimpleNamespace(logits=self.logits(toks), past_key_values="cache")

PROMPT = t.tensor([[3, 4]])

def test_stop_ids_of():
    m = Scripted([[7]])
    assert stop_ids_of(m) == {1, 5}
    m.generation_config = SimpleNamespace(eos_token_id=9)
    assert stop_ids_of(m) == {1, 9}
    m.generation_config = None
    assert stop_ids_of(m) == {1}
    bridge = SimpleNamespace(original_model=Scripted([[7]]), tokenizer=SimpleNamespace(eos_token_id=1))  # a TransformerBridge holds the HF model as original_model
    assert stop_ids_of(bridge) == {1, 5}
    assert stop_ids_of(Scripted([[7]]), SimpleNamespace(eos_token_id=2)) == {1, 2, 5}
    with pytest.raises(ValueError, match="pass stop_ids"):
        stop_ids_of(SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=None)))

def test_stream_toks_stops_on_any_stop_id():
    assert list(stream_toks(Scripted([[7, 8, 5, 9]]), PROMPT, new_toks=10)) == [7, 8]  # 5 ends the turn although the tokenizer's eos is 1
    assert list(stream_toks(Scripted([[7, 8, 5, 9, 1, 2]]), PROMPT, new_toks=10, stop_ids={1})) == [7, 8, 5, 9]
    assert list(stream_toks(Scripted([[7, 8, 9, 9, 9]]), PROMPT, new_toks=3)) == [7, 8, 9]
    assert list(stream_toks_hf(ScriptedHF([[7, 5, 9]]), SimpleNamespace(eos_token_id=1), PROMPT, new_toks=10)) == [7]

def test_sample_batch_cuts_each_row_before_its_stop():
    out = sample_batch(Scripted([[7, 8, 5, 9, 9], [7, 1, 6, 6, 6], [7, 8, 9, 9, 9]]), PROMPT, n=3, new_toks=5, quiet=True)
    assert out == [[7, 8], [7], [7, 8, 9, 9, 9]]  # the last row hit new_toks without stopping, which its length says
    out = sample_batch(Scripted([[7, 8, 5, 9, 9], [7, 1, 6, 6, 6]]), PROMPT, n=2, new_toks=5, quiet=True, stop_ids={6})
    assert out == [[7, 8, 5, 9, 9], [7, 1]]
