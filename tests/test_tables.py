import numpy as np
import pytest
import torch as t
from tabulate import tabulate

from mechtools.colors import bold, endc
from mechtools.tables import *

class FakeTok:
    def decode(self, i): return f"<{i[0] if isinstance(i, list) else i}>"

def test_html_table():
    h = html_table(["a", "b<"], [(0.123456, "<x>"), (t.tensor(2.0), np.float64(1 / 3)), (True, None)], title="T&")
    assert "<caption" in h and "T&amp;" in h and "b&lt;" in h and "&lt;x&gt;" in h
    assert "0.1235" in h and ">2<" in h and "0.3333" in h and ">True<" in h and ">None<" in h
    assert "tensor(" not in h and h.count("<tr>") == 4
    assert "<caption" not in html_table(["a"], [(1,)])

def test_print_titled_table(capsys):
    boxed = tabulate([(1, 2)], headers=["a", "b"], tablefmt="rounded_outline")
    print_titled_table(boxed, "title")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("╭") and f"{bold}" in lines[1] and "title" in lines[1] and lines[2].startswith("├")
    assert "│   a │   b │" in lines[3] and lines[3:] == boxed.splitlines()[1:]  # the header row is kept
    print_titled_table(boxed, "a title much longer than the table is wide")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"{bold}a title much longer than the table is wide{endc}" and lines[1:] == boxed.splitlines()
    plain = tabulate([(1, 2)], headers=["a", "b"], tablefmt="github")
    print_titled_table(plain, "title")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"{bold}title{endc}" and lines[1:] == plain.splitlines()  # no box to put the title in: nothing is dropped
    print_titled_table(plain)
    assert capsys.readouterr().out.splitlines() == plain.splitlines()

def test_show_table_text_mode(capsys):
    show_table(["x", "y"], [(1, 2.5)], title="T")
    out = capsys.readouterr().out
    assert out.startswith("╭") and "T" in out and "│   x │   y │" in out and "2.5" in out

def test_top_toks_table(capsys):
    logits = t.tensor([0.1, 3.0, -2.0, 1.0])
    top = top_toks_table(logits, FakeTok(), k=2, show_negative=True, title="T", return_top=True)
    assert top == (["'<1>'", "'<3>'"], [3.0, 1.0], ["'<2>'", "'<0>'"], [-2.0, pytest.approx(0.1)])
    out = capsys.readouterr().out
    header = next(l for l in out.splitlines() if "Idx" in l)
    assert [h.strip() for h in header.strip("│").split("│")] == ["Idx", "Top Tok", "Top Logit", "Top Prob", "Bot Tok", "Bot Logit", "Bot Prob"]
    row0 = next(l for l in out.splitlines() if "'<1>'" in l)
    assert f"{logits.softmax(-1)[1].item():.4f}"[:5] in row0 and "'<2>'" in row0
    assert top_toks_table(logits[None, None], FakeTok(), k=1, show_probs=False, return_top=True) == (["'<1>'"], [3.0])  # leading size-1 dims are fine
    assert "Prob" not in capsys.readouterr().out
    assert top_toks_table(logits, FakeTok(), k=1) is None
    with pytest.raises(ValueError, match=r"got shape \(3, 4\)"):
        top_toks_table(t.randn(1, 3, 4), FakeTok())  # a whole sequence's logits used to be flattened silently
