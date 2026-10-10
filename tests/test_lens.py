import html
import json
import math
import re

import pytest
import torch as t

from conftest import first_pane, widget_data
from mechtools import lens
from mechtools.lens import *

TOKS = [f"t{i}" for i in range(20)]

class FakeTok:
    """Words tokenize to their letter counts; a BOS (id 0) is prepended unless add_special_tokens=False, like the Llama and gemma tokenizers."""
    bos_token_id = 0
    def encode(self, s, add_special_tokens=True): return ([0] if add_special_tokens else []) + [len(w) for w in s.split()]
    def decode(self, i): return f"<{i[0] if isinstance(i, list) else int(i)}>"

class FakeModel:
    def __init__(self, d=4, vocab=6, seed=0):
        g = t.Generator().manual_seed(seed)
        self.W_U, self.W_E, self.tokenizer = t.randn(d, vocab, generator=g), t.randn(vocab, d, generator=g), FakeTok()
    def ln_final(self, h): return h
    def unembed(self, h): return h @ self.W_U
    def __call__(self, ids): return t.randn(1, ids.shape[1], self.W_U.shape[1])

@pytest.fixture
def shown(monkeypatch):
    out = []
    monkeypatch.setattr(lens, "display", lambda x: out.append(x.data))
    return out

def rules(css: str) -> list[tuple[str, str]]:
    """(selector, declarations) of every rule in a <style> block."""
    return [(sel.strip(), decl) for sel, decl in re.findall(r"([^{}]+)\{([^}]*)\}", css.removeprefix("<style>").removesuffix("</style>"))]

def test_token_strip_marks_positions():
    h = token_strip(TOKS, pos=[5, 12], ctx=3)
    assert h.count("<span data-p=") == 2 and "<span data-p=0 class=on data-h='pos 5" in h and "<span data-p=1 data-h='pos 12" in h and h.count("class=on") == 1  # the first marked token is selected from the start
    assert "data-h='pos 2 " in h and "data-h='pos 15 " in h and "data-h='pos 1 " not in h and "data-h='pos 16 " not in h  # window is min(pos)-ctx to max(pos)+ctx
    h = token_strip(TOKS, pos=7, ctx=3)
    assert "<span data-p" not in h and "class=on" not in h and h.count("outline:1px solid #fc6") == 1
    with pytest.raises(ValueError, match=r"repeat once wrapped: \[19, 19\]"):
        token_strip(TOKS, pos=[-1, 19])  # two tabs cannot share one mark

def test_fmt3_and_helpers():
    assert [fmt3(x) for x in (0.5, 1.0, 12.345, 1e-5, 0.0001234, 0.0)] == ["0.500", "1.00", "12.3", "1.00e-05", "0.000123", "0.00"]
    assert per_pos(lambda p: p * 2, 3) == 6 and per_pos(lambda p: p * 2, [1, 2]) == {1: 2, 2: 4}
    assert cluster_color(0) != cluster_color(1) and cluster_color(3) == f"hsl({math.floor(3 * 0.618 % 1 * 360 + 0.5)},70%,65%)" and cluster_color(0) == "hsl(0,70%,65%)"
    assert "hsl(${Math.floor(c*0.618%1*360+0.5)},70%,65%)" in TABBED_JS  # the script maps a cluster id to its color by the same expression, rounding included
    assert css_color(3) == cluster_color(3) and css_color("#e88") == "#e88" and css_color(None) is None
    assert norm_table(("h", [["x"]])) == ("h", [["x"]], None, None) and norm_table(("h", [["x"]], 2)) == ("h", [["x"]], 2, None) and norm_table(("h", [["x"]], 2, ["d"])) == ("h", [["x"]], 2, ["d"])
    assert norm_table(([("c3", 3), "n=2"], [], None, None)) == ([("c3", 3), "n=2"], [], None, None)
    with pytest.raises(ValueError, match="2 row colors for 1 rows"):
        norm_table(("h", [["x"]], [1, 2]))
    with pytest.raises(TypeError, match=r"list of str cells.*got \(\['a', '1'\], '#e88'\)"):
        norm_table(("h", [(["a", "1"], "#e88")]))  # the html form of before: (cell htmls, color) rows
    with pytest.raises(TypeError, match="list of str cells"):
        norm_table(("h", [["a", 1.0]]))
    with pytest.raises(TypeError, match="header is a str or a list of segments"):
        norm_table((3, []))
    import numpy as np
    assert norm_table(("h", [["x"], ["y"]], [np.int64(3), t.tensor(4)])) == ("h", [["x"], ["y"]], [3, 4], None) and norm_table(("h", [["x"]], t.tensor(2))) == ("h", [["x"]], 2, None)  # a numpy or tensor cluster id is an int
    assert norm_table(([("c3", np.int32(3)), "n=2"], [], "#e88")) == ([("c3", 3), "n=2"], [], "#e88", None) and norm_table(("h", [["x"]], [None]))[2] == [None]
    assert norm_color(None) is None and norm_color("#e88") == "#e88" and norm_color(t.tensor(5)) == 5 and type(norm_color(np.int64(5))) is int
    for bad in (1.5, t.tensor(1.0), b"x"):
        with pytest.raises(TypeError, match="a color is a cluster id"):
            norm_table(("h", [["x"]], bad))
    with pytest.raises(TypeError, match="a color is a cluster id"):
        norm_table(("h", [["x"], ["y"]], [1, 2.0]))
    assert header_html("a<b") == "a&lt;b" and header_html([("c3", 3), "mass 0.5", "n=2<"]) == f'<span style="color:{cluster_color(3)}">c3</span> &middot; mass 0.5 &middot; n=2&lt;'

def test_readout_grid_renders_tables():
    assert readout_grid([("h", [])]) == "<div class='rg'><table><tr><th colspan=1>h</th></tr></table></div>"  # an empty table renders its header
    h = readout_grid([("a<b", [["x<", "'y&'"], ["p", "q"]])])
    assert "<th colspan=2>a&lt;b</th>" in h and "<tr><td>x&lt;</td><td>'y&amp;'</td></tr><tr><td>p</td><td>q</td></tr>" in h  # cells are text, escaped here; a quote stays one byte
    assert "<td>a</td><td class=d>b</td><td class=w><div>c</div></td><td class=v>d</td><td>e</td>" in readout_grid([("h", [["a", "b", "c", "d", "e"]], None, [None, "d", "w", "v"])])  # per-column classes, none past the list; a wrapping cell's text in a div
    h = readout_grid([("h", [["a", "b"], ["c", "d"]], 5), ("i", [["e"], ["f"]], [None, "#e88"]), ("j", [["g"]], "rgb(230, 80, 80)"), ("k", [["x"]], 'a"b')])
    assert h.count(f'<tr style="color:{cluster_color(5)}"><td>') == 2  # one cluster id colors every row
    assert "<tr><td>e</td></tr>" in h and '<tr style="color:#e88"><td>f</td></tr>' in h  # a list colors row by row
    assert '<tr style="color:rgb(230, 80, 80)"><td>g</td></tr>' in h and '<tr style="color:a&quot;b"><td>x</td></tr>' in h  # quoted and escaped, so a space or a quote cannot split the attribute
    with pytest.raises(ValueError, match="3 row colors for 2 rows"):
        readout_grid([("h", [["a"], ["b"]], [1, 2, 3])])
    with pytest.raises(TypeError, match="list of str cells"):
        readout_grid([("h", [(["a", "1"], None)])])

def test_cell_color_policy():
    """A row's color is on its tr and reaches its cells by inheritance; the only cell colors are the dim 'd' column and the value 'v' column, which stays the frame's color, so a cluster row's name and cluster cells are colored and its value is not, and a single-cell row (olens's shape) is colored."""
    assert readout_grid([("h", [["a", "b", "c"]], "#e88", [None, "d", "v"])]) == "<div class='rg'><table><tr><th colspan=3>h</th></tr>" + '<tr style="color:#e88"><td>a</td><td class=d>b</td><td class=v>c</td></tr></table></div>'
    assert f'<tr style="color:{cluster_color(7)}"><td class=w><div>only</div></td></tr>' in readout_grid([("h", [["only"]], 7, ["w"])])
    rs = rules(readout_css("fid"))
    for sel, decl in rs:
        bare = sel.removeprefix("#fid ")
        if re.fullmatch(r"(table|tr|th|td)(:[\w-]+)?", bare):  # a rule on bare cells, rows or the table: no color of its own, only the frame's or the row's
            assert set(re.findall(r"(?:^|;)color:([^;]+)", decl)) <= {"inherit", "var(--fg)"}, sel
    assert {sel for sel, decl in rs if re.search(r"(?:^|;)color:", decl) and " td" in sel} == {"#fid td", "#fid td.d", "#fid td.v"}  # td inherits; d and v are the column colors
    decls = dict(rs)
    assert "color:#999" in decls["#fid td.d"] and decls["#fid td.v"] == "text-align:right;color:var(--fg)" and "color:inherit" in decls["#fid td"] and "color:inherit" in decls["#fid th"]

def test_readout_css_is_scoped_to_the_frame():
    """Every rule is under the frame's id, so it outranks a host's class rules (JupyterLab right-aligns td, stripes tbody rows and centers fixed-layout tables at (0,2,x); VS Code right-aligns and stripes rows) and the global rules of readouts saved by earlier versions, and the table rule resets what a host sets on tables."""
    rs = rules(readout_css("fabc"))
    assert rs and all(sel.startswith("#fabc ") and "," not in sel for sel, _ in rs)
    decls = dict(rs)
    assert decls["#fabc table"] == "border-collapse:collapse;border-spacing:0;border:0;margin:0;table-layout:auto;align-self:start;font:inherit;color:var(--fg)"
    assert "text-align:left" in decls["#fabc td"] and "border:0" in decls["#fabc td"] and "white-space:nowrap" in decls["#fabc td"] and decls["#fabc tr"] == "background:none" and "border:0" in decls["#fabc th"] and "font-weight:normal" in decls["#fabc th"]
    assert decls["#fabc td.w"] == "white-space:pre-wrap" and decls["#fabc td.w>div"] == "max-width:100ch" and "repeat(var(--n),1fr)" in decls["#fabc .rg"] and decls["#fabc .pane"] == "visibility:visible"
    assert not any(":last-child" in sel for sel, _ in rs)  # the value column is a class, not a position, so a one-column table is not right-aligned
    h = readout_html("<b>body</b>", "T", TOKS, None, 3, 2, n_cols=3)
    uid = re.search(r"<div id='(\w+)' style='--n:3;--fg:#eee;background:#111;color:var\(--fg\);", h).group(1)
    assert uid[0].isalpha() and h.startswith(f"<style>#{uid} .rg{{") and h.count("#eee") == 1 and ".ro" not in h  # one source of the foreground; no unscoped class rule
    assert "T</h3>" in h and "outline:1px solid #fc6" in h and h.endswith("<b>body</b></div>")
    assert re.search(r"<div id='(\w+)'", readout_html("x")).group(1) != uid  # a fresh id per frame

def test_tabbed_payload_and_prerender():
    panes = {"L0": {"p1": [("a", [["x"]])], "p2": [("b", [])]}, "L1": {"p1": [("c", [["y", "z"]], 3, [None, "d"])], "p2": "<i>raw</i>"}}
    h = tabbed(panes, head=token_strip(TOKS, pos=[1, 2]))
    d = widget_data(h)
    assert d["tabs"] == ["L0", "L1"] and d["inner"] == ["p1", "p2"] and d["panes"] == [[None, [["b", [], None, None]]], [[["c", [["y", "z"]], 3, [None, "d"]]], "<i>raw</i>"]]  # the first pane's entry is null (it is the rendered one); every other table in its 4-tuple form, raw html panes as they are
    assert first_pane(h) == readout_grid([("a", [["x"]])]) and h.count("class='pane'") == 1 and h.count("<table>") == 1  # the first pane is the one rendered
    assert h.index("data-p=0") < h.index("class='tb'") < h.index("class='pn'")  # head, bars, pane area
    assert "<button class=on>L0</button><button>L1</button>" in h and "<button class=on>p1</button><button>p2</button>" in h and "<span data-p=0 class=on " in h  # the first tabs and the first marked token are selected without a script
    uid = re.search(r"<div id='(\w+)' tabindex=0", h).group(1)
    assert f"<script type=application/json id='{uid}d'>" in h and f"const r=document.getElementById('{uid}')" in h and "getElementById(r.id+'d')" in h and h.count("<script>") == 1 and h.index("application/json") < h.index("<script>(()=>")
    assert "box.replaceChildren(p==null?first:pane(p))" in h and "box.style.minHeight=floor+'px'" in h and "new ResizeObserver" in h  # the rendered first pane is shown as is, the others built on show; the height floor
    assert ").textContent=x" in TABBED_JS and "th.append(s)" in TABBED_JS and "innerHTML" not in TABBED_JS.split("const pane=")[0]  # tables are built from text; innerHTML only takes a raw html pane
    h = tabbed({"<b>x</b>": {"p": [("a", [["1"]])], "q": [("</script><b>", [["<", "↵"]])]}}, bars=False)
    raw = re.search(r"<script type=application/json id='\w+'>(.*?)</script>", h).group(1)
    assert "<" not in raw and "\\u003c" in raw and "↵" in raw and widget_data(h)["panes"][0][1] == [["</script><b>", [["<", "↵"]], None, None]]  # no "<" in the script data, non-ascii as is
    assert h.count("class='tb' hidden") == 2 and "<button class=on>&lt;b&gt;x&lt;/b&gt;</button>" in h and "<td>1</td>" in h
    one = tabbed({"only": [("h", [["x", "y"]], None, [None, "v"])]})
    assert "<script" not in one and "class='tb' hidden" in one and one.count("<table>") == 1 and "<td>x</td><td class=v>y</td>" in one and first_pane(one) == readout_grid([("h", [["x", "y"]], None, [None, "v"])])  # a single pane: no payload, no script

def test_tabbed_validation_and_escaping():
    h = tabbed({"<b>x</b>": [("p", [])], "L&1": [("p2", [])]})
    assert "<button class=on>&lt;b&gt;x&lt;/b&gt;</button><button>L&amp;1</button>" in h and "class='tb'>" in h and "<th colspan=1>p</th>" in h and json.loads(re.search(r"json id='\w+'>(.*?)</script>", h).group(1))["panes"] == [[None], [[["p2", [], None, None]]]]
    assert "class='tb' hidden" in tabbed({"only": "pane"}) and ">pane<" in tabbed({"only": "pane"})  # a single tab shows no bar; a raw html pane
    with pytest.raises(ValueError, match="no panes"):
        tabbed({})
    with pytest.raises(ValueError, match="no panes"):
        tabbed({"L0": {}})  # an empty inner dict is no pane either
    with pytest.raises(ValueError, match="same inner keys"):
        tabbed({"a": {"p1": "x"}, "b": {"p2": "y"}})
    with pytest.raises(ValueError, match="2 row colors for 1 rows"):
        tabbed({"a": [("h", [["x"]], [1, 2])]})
    with pytest.raises(TypeError, match="list of str cells"):
        tabbed({"a": [("h", [["x"]])], "b": [("h", [["x", 2]])]})  # every pane is checked, not only the rendered one

def test_get_toks():
    tok = FakeTok()
    assert get_toks(None) == (None, None) and get_toks(["a", "b"]) == (["a", "b"], None)
    assert get_toks([3, 5], tok) == (["<3>", "<5>"], [3, 5]) and get_toks(t.tensor([[3, 5]]), tok) == (["<3>", "<5>"], [3, 5])
    assert get_toks("one two", tok) == (["<0>", "<3>", "<3>"], [0, 3, 3]) and get_toks("one two", tok, add_special_tokens=False) == (["<3>", "<3>"], [3, 3])
    with pytest.raises(ValueError, match="empty"):
        get_toks([])
    with pytest.raises(AssertionError, match="pass tokenizer"):
        get_toks("x")

def test_show_logits_marks_every_position_and_the_actual_next_token(shown):
    tok = type("T", (), {"decode": lambda self, i: f"<{int(i)}>"})()
    logits = t.zeros(3, 8)
    logits[0, 1] = 9.0  # pos 0 predicts the actual next id, 1
    logits[1] = t.tensor([9., 8., 7., 6., 0., 5., 4., 3.])  # pos 1's actual next id, 4, is ranked last
    show_logits([3, 1, 4], logits=logits, tokenizer=tok, k=3)
    h = shown[0]
    d = widget_data(h)
    assert d["tabs"] == [""] and d["inner"] == ["p0", "p1", "p2"] and h.count("data-p=") == 3 and h.count("class='tb' hidden") == 2  # a pane and a clickable token per position, no tab bars
    none, (p1,), (p2,) = d["panes"][0]  # one table per pane; the first is rendered, not shipped
    probs = logits.softmax(-1)
    p0 = first_pane(h)
    assert none is None and p0.count("<tr") == 4 and "<th colspan=3>p0 &middot; '&lt;3&gt;' →</th>" in p0
    assert f'<tr style="color:{NEXT_COLOR}"><td class=d>#1</td><td>\'&lt;1&gt;\'</td><td class=v>{fmt3(probs[0, 1])}</td></tr>' in p0 and p0.count(f"<td class=v>{fmt3(probs[0, 0])}</td>") == 2  # actual next token colored in place at rank 1, then two of the tied rest
    assert len(p1[1]) == 4 and p1[1][-1] == ["#8", "'<4>'", fmt3(probs[1, 4])] and p1[2] == [None, None, None, NEXT_COLOR] and p1[3] == ["d", None, "v"]  # outside the top-k, appended with its rank
    assert len(p2[1]) == 3 and p2[2] is None  # last position has no next token

def test_show_logits_inputs(shown):
    tok = FakeTok()
    with pytest.raises(ValueError, match="str tokens"):
        show_logits(["a", "b"], model=FakeModel(), tokenizer=tok)
    with pytest.raises(ValueError, match=r"got \(2, 3, 8\)"):
        show_logits([3, 1, 4], logits=t.randn(2, 3, 8), tokenizer=tok)
    with pytest.raises(ValueError, match="3 input tokens"):
        show_logits([3, 1, 4], logits=t.randn(5, 8), tokenizer=tok)
    with pytest.raises(ValueError, match="needs input_src"):
        show_logits(None, logits=t.randn(3, 8), tokenizer=tok)
    with pytest.raises(ValueError, match=r"repeats a position once wrapped: \[2, 2\]"):
        show_logits([3, 1, 4], logits=t.randn(3, 8), tokenizer=tok, pos=[-1, 2])  # one mark per pane
    show_logits([3, 1, 4], logits=t.randn(1, 3, 8), tokenizer=tok, pos=[-1], k=2)
    assert "application/json" not in shown[-1] and first_pane(shown[-1]).count("<tr") == 3 and "data-p=0 class=on data-h='pos 2" in shown[-1] and "<h3" in shown[-1]  # one position: one rendered pane, no payload
    show_logits("one two", model=FakeModel(), k=2, title=None)  # a string is tokenized (with BOS) and run through the model
    assert len(widget_data(shown[-1])["inner"]) == 3 and "<h3" not in shown[-1]
    show_logits(["a", "b"], logits=t.randn(2, 8), tokenizer=tok)  # str tokens with logits: no ids, so no next-token coloring
    d = widget_data(shown[-1])
    assert len(d["inner"]) == 2 and all(tb[2] is None for pane in d["panes"][0] if pane for tb in pane) and NEXT_COLOR not in shown[-1]

def test_get_jlens_token_vec():
    m = FakeModel(d=8, vocab=8)
    m.W_U = t.eye(8)
    J = {"J": t.eye(8)[None]}
    assert get_jlens_token_vec(" cat", 0, m, J).argmax().item() == 3  # the token's id, not the BOS the tokenizer prepends to a string
    assert t.equal(get_jlens_token_vec(5, 0, m, J), t.eye(8)[5])
    with pytest.raises(ValueError, match="2 tokens"):
        get_jlens_token_vec("two words", 0, m, J)

def test_top_readout(shown):
    scores = {"a<": t.tensor([1.0, 3.0, 2.0]), "b": t.tensor([0.0, 0.0, 5.0])}
    top_readout(scores, ["x", "y", "z<"], k=2, input_src=["u", "v"], pos=1)
    h = shown[-1]
    assert h.count("<table>") == 2 and "<th colspan=2>a&lt;</th>" in h and "<td>'z&lt;'</td>" in h and "application/json" not in h  # one static grid, no payload
    assert h.count("<tr") == 6 and f"<td class=v>{fmt3(scores['a<'].softmax(-1)[1].item())}</td>" in h and h.count("outline:1px solid #fc6") == 1
    top_readout(scores, lambda i: f"n{i}", k=1, softmax=False)
    assert "<td>'n1'</td><td class=v>3.00</td>" in shown[-1] and "<td>'n2'</td><td class=v>5.00</td>" in shown[-1]

def test_cluster_tables_and_readout(shown):
    t.manual_seed(0)
    scores = t.randn(20)
    labels = t.tensor([0, 1, 2, 3, 4] * 4)
    names = [f"w{i}" for i in range(20)]
    tables, ids = cluster_tables(scores, labels, names.__getitem__, n_clusters=11, n_rows=3, softmax=True)
    assert len(tables) == 6 and len(ids) == 5  # more clusters asked for than exist: the overall table plus one per cluster
    probs = scores.softmax(-1)
    mass = t.zeros(5).index_add_(0, labels, probs)
    assert ids == mass.argsort(descending=True).tolist()
    header, rows, colors, cols = tables[0]
    top = probs.topk(3)
    assert header == "top-k overall" and rows == [[repr(names[i]), f"c{labels[i].item()}", fmt3(p)] for i, p in zip(top.indices.tolist(), top.values.tolist())] and colors == labels[top.indices].tolist() and cols == [None, None, "v"]  # each row colored by its own cluster, the value column neutral
    header, rows, colors, cols = tables[1]
    c = ids[0]
    members = (labels == c).nonzero().flatten()
    best = members[probs[members].argmax()]
    assert header[0] == (f"c{c}", c) and header[1] == f"mass {mass[c]:.3f}" and header[2] == f"n={len(members)}" and len(rows) == 3  # the cluster named in its color
    assert rows[0] == [repr(names[best]), f"#{(probs > probs[best]).sum().item() + 1}", fmt3(probs[best])] and colors == c and cols == [None, "d", "v"]  # the rank column is dim, every row the cluster's color
    tables_raw, ids_raw = cluster_tables(scores, labels, names.__getitem__, 2, 3, softmax=False)
    mx = t.full((5,), -t.inf).scatter_reduce_(0, labels, scores, "amax")
    assert ids_raw == mx.topk(2).indices.tolist() and tables_raw[1][0][1].startswith("max ") and tables_raw[0][1][0][2] == fmt3(scores.max().item())
    labels2 = labels.clone()
    labels2[labels2 == 3] = 4  # an empty cluster id inside the range, and more rows than items
    tables2, ids2 = cluster_tables(scores, labels2, names.__getitem__, 5, 30, softmax=True)
    assert 3 not in ids2 and len(tables2) == 5 and len(tables2[0][1]) == 20 and sorted(len(rows) for _, rows, _, _ in tables2[1:]) == [4, 4, 4, 8]
    with pytest.raises(ValueError, match="20 scores but 5 labels"):
        cluster_tables(scores, labels[:5], names.__getitem__, 2, 3, True)
    shown_ids = cluster_readout({"L0": scores, "L1": -scores}, labels, names, n_clusters=2, n_rows=3, input_src=TOKS, pos=5, title="T")
    assert shown_ids["L0"] == mass.topk(2).indices.tolist() and set(shown_ids) == {"L0", "L1"}
    h = shown[-1]
    d = widget_data(h)
    assert d["tabs"] == ["L0", "L1"] and d["inner"] == ["p5"] and "<button class=on>L0</button><button>L1</button>" in h and h.count("class='tb' hidden") == 1 and "T</h3>" in h
    assert d["panes"][0][0] is None and d["panes"][1][0] == json.loads(json.dumps([norm_table(tb) for tb in cluster_tables(-scores, labels, names.__getitem__, 2, 3, True)[0]]))  # the payload is cluster_tables' output for the panes not rendered
    assert first_pane(h) == readout_grid(cluster_tables(scores, labels, names.__getitem__, 2, 3, True)[0]) and h.count("<table>") == 3 and h.count(f'<tr style="color:{cluster_color(ids[0])}">') >= 3  # the first pane rendered, its rows colored
    nested = cluster_readout({"L0": {2: scores, 7: -scores}}, {"L0": labels}, names, n_clusters=2, n_rows=3, input_src=TOKS)
    assert list(nested["L0"]) == [2, 7] and nested["L0"][2] == shown_ids["L0"]
    h = shown[-1]
    d = widget_data(h)
    assert d["tabs"] == ["L0"] and d["inner"] == ["p2", "p7"] and h.count("data-p=") == 2 and "<button class=on>p2</button><button>p7</button>" in h and "<span data-p=1 data-h='pos 7" in h and h.count("<table>") == 3

def test_lens_readouts_with_fake_cache(shown):
    d = 4
    m = FakeModel(d=d, vocab=6)
    cache = {f"blocks.{l}.hook_resid_pre": t.randn(1, 5, d) for l in range(3)} | {"blocks.1.hook_resid_post": t.randn(1, 5, d)}
    jl = {"J": t.stack([t.eye(d) * (l + 1) for l in range(3)])}
    h = cache["blocks.1.hook_resid_pre"][0, -1]
    assert t.allclose(get_lens_logits(h, 1, m, jl), 2 * h @ m.W_U) and t.allclose(jlens_transport(h, 2, jl), 3 * h)
    jlens_readout(cache, [0, 2], -1, m, jl, k=2, input_src=[3, 1, 4, 1, 5])
    out = shown[-1]
    assert out.count("<table>") == 2 and "<th colspan=2>L0</th>" in out and "<th colspan=2>L2</th>" in out and out.count("<tr") == 6
    top = (3 * cache["blocks.2.hook_resid_pre"][0, -1] @ m.W_U).softmax(-1).topk(2)
    assert f"<td>'&lt;{top.indices[0].item()}&gt;'</td><td class=v>{fmt3(top.values[0].item())}</td>" in out and "outline:1px solid #fc6" in out
    jlens_readout(cache, [1], 2, m, jl, hook="hook_resid_post", title="T")
    assert "T</h3>" in shown[-1] and fmt3((2 * cache["blocks.1.hook_resid_post"][0, 2] @ m.W_U).softmax(-1).max().item()) in shown[-1]
    tl = {"templates": t.randn(3, 5, d), "words": [f"tpl{i}" for i in range(5)]}
    sc = get_tlens_scores(h, 1, tl)
    assert sc.shape == (5,) and t.allclose(sc, t.cosine_similarity(tl["templates"][1], h[None], dim=-1))
    assert get_template_idx("tpl3", tl) == 3 and t.equal(get_template_vec("tpl3", 1, tl), tl["templates"][1, 3]) and get_template_vecs(["tpl0", "tpl4"], 2, tl).shape == (2, d)
    tlens_readout(cache, [1], -1, tl, k=3, tokenizer=m.tokenizer, input_src="a b")
    out = shown[-1]
    assert out.count("<table>") == 1 and f"<td>'tpl{sc.argmax().item()}'</td><td class=v>{fmt3(sc.max().item())}" in out
    labels = t.tensor([0, 1, 0, 1, 0, 1])
    ids = jlens_cluster_readout(cache, [0, 1], [1, 3], m, jl, labels, n_clusters=2, n_rows=2, input_src=[3, 1, 4, 1, 5])
    data = widget_data(shown[-1])
    assert list(ids) == ["L0", "L1"] and list(ids["L0"]) == [1, 3] and data["tabs"] == ["L0", "L1"] and data["inner"] == ["p1", "p3"] and shown[-1].count("data-p=") == 2
    ids = tlens_cluster_readout(cache, [0], -1, tl, k=2, n_clusters=2, n_rows=2, tokenizer=m.tokenizer)
    assert list(ids) == ["L0"] and sorted(ids["L0"]) == [0, 1] and "application/json" not in shown[-1] and shown[-1].count("<table>") == 3  # one layer at one position: one rendered pane, no payload

def test_vocab_vecs_and_cluster_vocab():
    m = FakeModel(d=4, vocab=30)
    x = vocab_vecs(m)
    assert x.shape == (30, 4) and t.allclose(x.mean(0), t.zeros(4), atol=1e-6) and vocab_vecs(m, embed=True).shape == (30, 4)
    labels, cents = cluster_vocab(m, k=3, iters=5)
    assert labels.shape == (30,) and cents.shape == (3, 4)

def test_top_templates_table(capsys):
    top_templates_table(t.tensor([0.1, 0.9, 0.5]), ["a", "b", "c"], k=2, title="T")
    out = capsys.readouterr().out
    assert "'b'" in out and "0.9" in out and "'a'" not in out

@pytest.mark.hf
def test_bridge_readouts(tiny_bridge, shown):
    model = tiny_bridge
    show_logits("Hello there", model=model, k=3)
    n = len(model.tokenizer.encode("Hello there"))
    assert len(widget_data(shown[-1])["inner"]) == n
    ids = t.tensor([model.tokenizer.encode("Hello there")])
    _, cache = model.run_with_cache(ids)
    J = {"J": t.eye(model.cfg.d_model)[None].repeat(model.cfg.n_layers, 1, 1)}
    jlens_readout(cache, [0, 1], -1, model, J, k=3, input_src="Hello there")
    assert shown[-1].count("<table>") == 2
    tok_id = model.tokenizer.encode("Hello", add_special_tokens=False)[0]
    assert tok_id != model.tokenizer.bos_token_id and t.allclose(get_jlens_token_vec("Hello", 0, model, J), model.W_U[:, tok_id])
    assert vocab_vecs(model).shape == (model.cfg.d_vocab, model.cfg.d_model)
    labels, _ = cluster_vocab(model, k=8, iters=3)
    jlens_cluster_readout(cache, [1], [0, -1], model, J, labels, n_clusters=3, n_rows=4, input_src=ids)
    assert widget_data(shown[-1])["inner"] == ["p0", "p-1"]
