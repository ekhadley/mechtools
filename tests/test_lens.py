import html
import re

import pytest
import torch as t

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

def test_tabbed_validation_and_escaping():
    h = tabbed({"<b>x</b>": "pane", "L&1": "p2"})
    assert "<button>&lt;b&gt;x&lt;/b&gt;</button>" in h and "<button>L&amp;1</button>" in h and "class='tb'>" in h and ">pane<" in h
    assert "class='tb' hidden" in tabbed({"only": "pane"})  # a single tab shows no bar
    with pytest.raises(ValueError, match="no panes"):
        tabbed({})
    with pytest.raises(ValueError, match="same inner keys"):
        tabbed({"a": {"p1": "x"}, "b": {"p2": "y"}})

def test_get_toks():
    tok = FakeTok()
    assert get_toks(None) == (None, None) and get_toks(["a", "b"]) == (["a", "b"], None)
    assert get_toks([3, 5], tok) == (["<3>", "<5>"], [3, 5]) and get_toks(t.tensor([[3, 5]]), tok) == (["<3>", "<5>"], [3, 5])
    assert get_toks("one two", tok) == (["<0>", "<3>", "<3>"], [0, 3, 3])
    with pytest.raises(ValueError, match="empty"):
        get_toks([])
    with pytest.raises(AssertionError, match="pass tokenizer"):
        get_toks("x")

def test_fmt3_and_helpers():
    assert [fmt3(x) for x in (0.5, 1.0, 12.345, 1e-5, 0.0001234, 0.0)] == ["0.500", "1.00", "12.3", "1.00e-05", "0.000123", "0.00"]
    assert per_pos(lambda p: p * 2, 3) == 6 and per_pos(lambda p: p * 2, [1, 2]) == {1: 2, 2: 4}
    assert cluster_color(0) != cluster_color(1) and cluster_color(3).startswith("hsl(")
    assert readout_grid([("h", [])]).count("<th colspan=1>h</th>") == 1  # an empty table renders its header

def test_show_logits_marks_every_position_and_the_actual_next_token(shown):
    tok = type("T", (), {"decode": lambda self, i: f"<{int(i)}>"})()
    logits = t.zeros(3, 8)
    logits[0, 1] = 9.0  # pos 0 predicts the actual next id, 1
    logits[1] = t.tensor([9., 8., 7., 6., 0., 5., 4., 3.])  # pos 1's actual next id, 4, is ranked last
    show_logits([3, 1, 4], logits=logits, tokenizer=tok, k=3)
    h = shown[0]
    assert h.count("class='pane'") == 3 and h.count("data-p=") == 3 and h.count("class='tb' hidden") == 2  # a pane and a clickable token per position, no tab bars
    p0, p1, p2 = re.findall(r"<table>.*?</table>", h)
    assert p0.count("<tr") == 4 and p0.count("color:#e88") == 1 and "#1</span></td><td>&#x27;&lt;1&gt;" in p0  # actual next token colored in place at rank 1
    assert p1.count("<tr") == 5 and p1.split("<tr")[-1].count("#8") == 1 and "&lt;4&gt;" in p1.split("<tr")[-1]  # outside the top-k, appended with its rank
    assert p2.count("color:#e88") == 0  # last position has no next token

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
    show_logits([3, 1, 4], logits=t.randn(1, 3, 8), tokenizer=tok, pos=[-1], k=2)
    assert shown[-1].count("class='pane'") == 1 and "data-p=0 title='2" in shown[-1] and "<h3" in shown[-1]
    show_logits("one two", model=FakeModel(), k=2, title=None)  # a string is tokenized (with BOS) and run through the model
    assert shown[-1].count("class='pane'") == 3 and "<h3" not in shown[-1]
    show_logits(["a", "b"], logits=t.randn(2, 8), tokenizer=tok)  # str tokens with logits: no ids, so no next-token coloring
    assert shown[-1].count("class='pane'") == 2 and "color:#e88" not in shown[-1]

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
    assert h.count("<table>") == 2 and "<th colspan=2>a&lt;</th>" in h and "&#x27;z&lt;&#x27;" in h
    assert h.count("<tr") == 6 and fmt3(scores["a<"].softmax(-1)[1].item()) in h and h.count("border-bottom:2px solid #e66") == 1
    top_readout(scores, lambda i: f"n{i}", k=1, softmax=False)
    assert "&#x27;n1&#x27;</td><td>3.00</td>" in shown[-1] and "&#x27;n2&#x27;</td><td>5.00</td>" in shown[-1]

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
    header, rows = tables[0]
    top = probs.topk(3)
    assert header == "top-k overall" and [r[0][0] for r in rows] == [html.escape(repr(names[i])) for i in top.indices.tolist()] and [r[0][1] for r in rows] == [f"c{labels[i].item()}" for i in top.indices.tolist()]
    header, rows = tables[1]
    c = ids[0]
    members = (labels == c).nonzero().flatten()
    best = members[probs[members].argmax()]
    assert f"c{c}" in header and "mass" in header and f"n={len(members)}" in header and len(rows) == 3
    assert rows[0][0][0] == html.escape(repr(names[best])) and rows[0][0][1] == f"<span style='color:#999'>#{(probs > probs[best]).sum().item() + 1}</span>" and rows[0][1] == cluster_color(c)
    tables_raw, ids_raw = cluster_tables(scores, labels, names.__getitem__, 2, 3, softmax=False)
    mx = t.full((5,), -t.inf).scatter_reduce_(0, labels, scores, "amax")
    assert ids_raw == mx.topk(2).indices.tolist() and "max" in tables_raw[1][0] and tables_raw[0][1][0][0][2] == fmt3(scores.max().item())
    labels2 = labels.clone()
    labels2[labels2 == 3] = 4  # an empty cluster id inside the range, and more rows than items
    tables2, ids2 = cluster_tables(scores, labels2, names.__getitem__, 5, 30, softmax=True)
    assert 3 not in ids2 and len(tables2) == 5 and len(tables2[0][1]) == 20 and sorted(len(rows) for _, rows in tables2[1:]) == [4, 4, 4, 8]
    with pytest.raises(ValueError, match="20 scores but 5 labels"):
        cluster_tables(scores, labels[:5], names.__getitem__, 2, 3, True)
    shown_ids = cluster_readout({"L0": scores, "L1": -scores}, labels, names, n_clusters=2, n_rows=3, input_src=TOKS, pos=5, title="T")
    assert shown_ids["L0"] == mass.topk(2).indices.tolist() and set(shown_ids) == {"L0", "L1"}
    h = shown[-1]
    assert h.count("class='pane'") == 2 and h.count("<table>") == 6 and "<button>L0</button><button>L1</button>" in h and h.count("class='tb' hidden") == 1 and "T</h3>" in h
    nested = cluster_readout({"L0": {2: scores, 7: -scores}}, {"L0": labels}, names, n_clusters=2, n_rows=3, input_src=TOKS)
    assert list(nested["L0"]) == [2, 7] and nested["L0"][2] == shown_ids["L0"]
    h = shown[-1]
    assert h.count("class='pane'") == 2 and h.count("data-p=") == 2 and "<button>p2</button><button>p7</button>" in h and "<span data-p=1 title='7" in h

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
    assert f"<td>&#x27;&lt;{top.indices[0].item()}&gt;&#x27;</td><td>{fmt3(top.values[0].item())}</td>" in out and "border-bottom:2px solid #e66" in out
    jlens_readout(cache, [1], 2, m, jl, hook="hook_resid_post", title="T")
    assert "T</h3>" in shown[-1] and fmt3((2 * cache["blocks.1.hook_resid_post"][0, 2] @ m.W_U).softmax(-1).max().item()) in shown[-1]
    tl = {"templates": t.randn(3, 5, d), "words": [f"tpl{i}" for i in range(5)]}
    sc = get_tlens_scores(h, 1, tl)
    assert sc.shape == (5,) and t.allclose(sc, t.cosine_similarity(tl["templates"][1], h[None], dim=-1))
    assert get_template_idx("tpl3", tl) == 3 and t.equal(get_template_vec("tpl3", 1, tl), tl["templates"][1, 3]) and get_template_vecs(["tpl0", "tpl4"], 2, tl).shape == (2, d)
    tlens_readout(cache, [1], -1, tl, k=3, tokenizer=m.tokenizer, input_src="a b")
    out = shown[-1]
    assert out.count("<table>") == 1 and f"&#x27;tpl{sc.argmax().item()}&#x27;</td><td>{fmt3(sc.max().item())}" in out
    labels = t.tensor([0, 1, 0, 1, 0, 1])
    ids = jlens_cluster_readout(cache, [0, 1], [1, 3], m, jl, labels, n_clusters=2, n_rows=2, input_src=[3, 1, 4, 1, 5])
    assert list(ids) == ["L0", "L1"] and list(ids["L0"]) == [1, 3] and shown[-1].count("class='pane'") == 4 and shown[-1].count("data-p=") == 2
    ids = tlens_cluster_readout(cache, [0], -1, tl, k=2, n_clusters=2, n_rows=2, tokenizer=m.tokenizer)
    assert list(ids) == ["L0"] and sorted(ids["L0"]) == [0, 1] and shown[-1].count("<table>") == 3

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
    assert shown[-1].count("class='pane'") == n
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
    assert shown[-1].count("class='pane'") == 2
