from mechtools.lens import tabbed, token_strip

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
