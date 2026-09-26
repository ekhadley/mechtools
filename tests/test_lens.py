import re

import pytest
import torch as t

from mechtools.lens import get_jlens_token_vec, tabbed, token_strip

TOKS = [f"t{i}" for i in range(20)]

def test_token_strip_marks_positions():
    h = token_strip(TOKS, pos=[5, 12], ctx=3)
    assert h.count("<span data-p=") == 2 and "<span data-p=0 title='5" in h and "<span data-p=1 title='12" in h
    assert "title='2 " in h and "title='15 " in h and "title='1 " not in h and "title='16 " not in h  # window is min(pos)-ctx to max(pos)+ctx
    h = token_strip(TOKS, pos=7, ctx=3)
    assert "<span data-p" not in h and h.count("border-bottom:2px solid #e66") == 1

def test_tabbed_stacks_panes_and_wires_head():
    h = tabbed({"L0": {"p1": "a", "p2": "b"}, "L1": {"p1": "c", "p2": "d"}}, head=token_strip(TOKS, pos=[1, 2]))
    assert h.count("class='pane'") == 4 and h.count("class='pn'") == 1
    assert h.index("data-p=0") < h.index("class='tb'")  # head goes above the tab bars
    assert "marks.forEach(x=>x.onclick" in h and "x.classList.toggle('on',x.dataset.k==cur.join(','))" in h

def test_show_logits_marks_every_position_and_the_actual_next_token(monkeypatch):
    from mechtools import lens
    out = []
    monkeypatch.setattr(lens, "display", lambda x: out.append(x.data))
    tok = type("T", (), {"decode": lambda self, i: f"<{int(i)}>"})()
    logits = t.zeros(3, 8)
    logits[0, 1] = 9.0  # pos 0 predicts the actual next id, 1
    logits[1] = t.tensor([9., 8., 7., 6., 0., 5., 4., 3.])  # pos 1's actual next id, 4, is ranked last
    lens.show_logits([3, 1, 4], logits=logits, tokenizer=tok, k=3)
    h = out[0]
    assert h.count("class='pane'") == 3 and h.count("data-p=") == 3 and h.count("class='tb' hidden") == 2  # a pane and a clickable token per position, no tab bars
    p0, p1, p2 = re.findall(r"<table>.*?</table>", h)
    assert p0.count("<tr") == 4 and p0.count("color:#e88") == 1 and "#1</span></td><td>&#x27;&lt;1&gt;" in p0  # actual next token colored in place at rank 1
    assert p1.count("<tr") == 5 and p1.split("<tr")[-1].count("#8") == 1 and "&lt;4&gt;" in p1.split("<tr")[-1]  # outside the top-k, appended with its rank
    assert p2.count("color:#e88") == 0  # last position has no next token

def test_get_jlens_token_vec_uses_the_token_not_bos():
    class Tok:
        def encode(self, s, add_special_tokens=True):
            return ([0] if add_special_tokens else []) + [10 + i for i in range(len(s.split()))]  # 0 is BOS, as on Llama or Gemma
    model = type("M", (), {"tokenizer": Tok(), "W_U": t.randn(4, 16)})()
    jlens = {"J": t.randn(2, 4, 4)}
    v = get_jlens_token_vec("hello", 1, model, jlens)
    assert t.allclose(v, jlens["J"][1].T @ model.W_U[:, 10]) and t.equal(v, get_jlens_token_vec(10, 1, model, jlens))
    with pytest.raises(ValueError, match="2 tokens"):
        get_jlens_token_vec("hello world", 1, model, jlens)
