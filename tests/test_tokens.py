import pytest
import torch as t
from conftest import load_tokenizer

from mechtools.colors import underline, endc
from mechtools.tokens import *

MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B", "Qwen/Qwen2.5-3B-Instruct", "google/gemma-3-1b-it", "google/gemma-3-4b-it", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "berkeley-nest/Starling-LM-7B-alpha", "thinkingmachines/Inkling"]
CONV = [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "It is 4."},
    {"role": "user", "content": "And 3+3?\nShow work."},
    {"role": "assistant", "content": "3+3 = 6, since 3+3 is 6."},
]

EDGE = [
    {"role": "user", "content": " leading space, trailing newline\n"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "x"},
    {"role": "assistant", "content": "Two\n\nparagraphs."},
]


@pytest.fixture(scope="module", params=MODELS)
def tok(request):
    tok = load_tokenizer(request.param)
    if tok.pad_token is None and tok.eos_token is None:  # Inkling ships neither; any token not used in text works as pad since the attention mask covers it
        tok.pad_token = "<|unused|>"
    return tok

@pytest.fixture(scope="module")
def qwen():
    return load_tokenizer("Qwen/Qwen3-0.6B")

class FakeTok:
    def encode(self, s): return [1, 2, 3]
    def decode(self, i): return f"<{i}>"

def runs(mask: Tensor) -> list[tuple[int, int]]:
    """(start, end) of each contiguous run of 1s."""
    edges = t.diff(t.cat([t.tensor([0]), mask, t.tensor([0])])).nonzero().flatten().tolist()
    return list(zip(edges[::2], edges[1::2]))

def test_to_ids_input_kinds(tok):
    ids = to_ids("Hello world.", tok)
    assert to_ids(t.tensor([ids]), tok) == ids == to_ids(ids, tok) == to_ids(t.tensor([[ids]]), tok)
    assert to_ids(CONV, tok) == tok.apply_chat_template(CONV, tokenize=True, return_dict=False)
    assert to_str_toks(CONV, tok, add_generation_prompt=True) == [tok.decode(i) for i in to_ids(CONV, tok, add_generation_prompt=True)]
    assert to_ids([], tok) == [] and to_ids(t.tensor(5), tok) == [5]

def test_to_ids_rejects_batches_and_str_tokens():
    with pytest.raises(ValueError, match=r"got shape \(2, 2\)"):
        to_ids(t.tensor([[1, 2], [3, 4]]), FakeTok())
    with pytest.raises(TypeError, match="not str tokens"):
        to_ids(["a", "b"], FakeTok())

def test_toks_html():
    h = toks_html(["a<b", "\n", "c"], [1, 2, 3], pos=-1)
    assert "a&lt;b" in h and "↵" in h and h.count("border-bottom:2px solid #e66") == 1 and "id 3 &middot; &#x27;c&#x27;" in h
    h = toks_html(list("abcdefg"), lo=2, hi=5)
    assert h.count("<span") == 3 and "… " in h and " …" in h
    h = toks_html(list("abcdefg"), pos=[3, -1])
    assert h.count("<span data-p=") == 2 and "<span data-p=0 title='3" in h and "<span data-p=1 title='6" in h and "border-bottom:2px solid #e66" not in h
    assert "<span" not in toks_html([], pos=-1) and "<span" not in toks_html([], pos=[0])  # no tokens: no crash

def test_underline_stoks():
    s = underline_stoks([1, 2, 3], FakeTok())
    assert s == f"{endc}<1>{underline}<2>{endc}<3>{endc}"

def test_turn_tok_idx(tok):
    ids = to_ids(CONV, tok)
    for i, msg in enumerate(CONV):
        start, end = get_turn_tok_idx(CONV, i, tok, "both")
        assert tok.decode(ids[start:end]) == msg["content"]
        assert get_turn_tok_idx(CONV, i, tok, "start") == start
    assert get_turn_tok_idx(CONV, 4, tok, "both", add_generation_prompt=True) == (start, end)
    ids = to_ids(EDGE, tok)
    for i, msg in enumerate(EDGE):  # templates may trim whitespace at the content edges, so compare stripped
        start, end = get_turn_tok_idx(EDGE, i, tok, "both")
        assert tok.decode(ids[start:end]).strip() == msg["content"].strip()

def test_turn_tok_idx_edge_cases(tok):
    dup = [{"role": "user", "content": "same"}, {"role": "assistant", "content": "same"}]  # the same content in two turns: each span is its own
    ids = to_ids(dup, tok)
    spans = [get_turn_tok_idx(dup, i, tok, "both") for i in range(2)]
    assert spans[0][1] <= spans[1][0] and all(tok.decode(ids[s:e]) == "same" for s, e in spans)
    empty = [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}]  # empty content: an empty span after the user turn
    ids = to_ids(empty, tok)
    s, e = get_turn_tok_idx(empty, 1, tok, "both")
    assert s == e and get_turn_tok_idx(empty, 0, tok, "end") <= s <= len(ids)
    sentinel = pick_sentinel(tok, set(ids))  # content holding the default sentinel: another one is picked and the span is still right
    coll = [{"role": "user", "content": f"x {sentinel} y"}, {"role": "assistant", "content": "ok"}]
    ids = to_ids(coll, tok)
    assert pick_sentinel(tok, set(ids)) != sentinel
    s, e = get_turn_tok_idx(coll, 0, tok, "both")
    assert tok.decode(ids[s:e]) == f"x {sentinel} y"

def test_pick_sentinel(tok):
    ids = to_ids(CONV, tok)
    s = pick_sentinel(tok, set(ids))
    (sid,), = [[i for i, a in tok.added_tokens_decoder.items() if a.content == s]]
    assert sid not in ids and tok.encode(f"a {s} b", add_special_tokens=False).count(sid) == 1  # an added token: one id, split off atomically
    added = tok.added_tokens_decoder[sid]
    assert not (added.lstrip or added.rstrip) or all(a.lstrip or a.rstrip for i, a in tok.added_tokens_decoder.items() if i not in ids)
    with pytest.raises(ValueError, match="pass sentinel="):
        pick_sentinel(tok, set(tok.added_tokens_decoder))

def test_qwen3_reasoning_spans(qwen):
    """Pins how the sentinel span treats Qwen3's reasoning rendering: reasoning_content and the last turn's empty think block are template text outside the span and the mask; a <think> block inside an earlier turn's content is stripped by the template; inside the last turn's content it stays in the span."""
    rc = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "It is 4.", "reasoning_content": "hmm"}]
    ids = to_ids(rc, qwen)
    assert "<think>\nhmm\n</think>" in qwen.decode(ids)
    s, e = get_turn_tok_idx(rc, 1, qwen, "both")
    assert qwen.decode(ids[s:e]) == "It is 4."
    ids_b, _, mask = get_assistant_mask(qwen, rc)
    assert qwen.decode(ids_b[0][mask[0].bool()]) == "It is 4.<|im_end|>"
    plain = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "It is 4."}]
    ids_b, _, mask = get_assistant_mask(qwen, plain)
    assert qwen.decode(ids_b[0]).endswith("<think>\n\n</think>\n\nIt is 4.<|im_end|>\n") and qwen.decode(ids_b[0][mask[0].bool()]) == "It is 4.<|im_end|>"
    think = "<think>\nhmm\n</think>\n\nIt is 4."
    mid = [{"role": "user", "content": "q"}, {"role": "assistant", "content": think}, {"role": "user", "content": "q2"}, {"role": "assistant", "content": "five"}]
    ids = to_ids(mid, qwen)
    s, e = get_turn_tok_idx(mid, 1, qwen, "both")
    assert "hmm" not in qwen.decode(ids) and qwen.decode(ids[s:e]) == "It is 4."
    last = mid[:2]
    ids = to_ids(last, qwen)
    s, e = get_turn_tok_idx(last, 1, qwen, "both")
    assert qwen.decode(ids[s:e]) == "\nhmm\n</think>\n\nIt is 4."

def test_apply_chat_template(tok):
    items = ["hi", CONV[:2], CONV]
    ids, attn = apply_chat_template(tok, items)
    assert ids.shape == attn.shape and ids.shape[0] == 3 and attn[:, -1].all() and not attn[0, 0]
    for b, item in enumerate(items):
        conv = [{"role": "user", "content": item}] if isinstance(item, str) else item
        assert ids[b][attn[b].bool()].tolist() == to_ids(conv, tok, add_generation_prompt=True)
    single_ids, single_attn = apply_chat_template(tok, "hi")
    assert single_ids.shape[0] == 1 and single_attn.all() and single_ids[0].tolist() == ids[0][attn[0].bool()].tolist()

def test_apply_chat_template_restores_padding_side(tok):
    side = tok.padding_side
    tok.padding_side = "right"
    try:
        apply_chat_template(tok, "hi")
        assert tok.padding_side == "right"
        with pytest.raises(Exception):
            apply_chat_template(tok, "hi", chat_template="{{ raise_exception('boom') }}")
        assert tok.padding_side == "right"
        with pytest.raises(ValueError):
            apply_chat_template(tok, [])
    finally:
        tok.padding_side = side

@pytest.mark.parametrize("include_eot", [True, False])
def test_assistant_mask(tok, include_eot):
    convs = [CONV, CONV[:3]]
    ids, attn, mask = get_assistant_mask(tok, convs, include_eot=include_eot)
    assert ids.shape == attn.shape == mask.shape and ids.shape[0] == 2 and not (mask & ~attn).any()
    for b, conv in enumerate(convs):
        assert ids[b][attn[b].bool()].tolist() == to_ids(conv, tok)
        contents = [msg["content"] for msg in conv if msg["role"] == "assistant"]
        segments = [ids[b, start:end].tolist() for start, end in runs(mask[b])]
        assert len(segments) == len(contents)
        for seg, content in zip(segments, contents):
            if include_eot:
                assert seg[-1] in tok.added_tokens_decoder and tok.decode(seg[:-1]) == content
            else:
                assert tok.decode(seg) == content
    single = get_assistant_mask(tok, CONV, include_eot=include_eot)
    assert single[2].shape[0] == 1 and single[2][0].tolist() == mask[0].tolist()
    assert not get_assistant_mask(tok, CONV[:2])[2].any()  # no assistant turn: nothing marked

def test_completion_loss():
    toks, mask, V = t.tensor([[1, 2, 3, 4, 5]]), t.tensor([[0, 0, 1, 1, 1]]), 7
    logits = t.zeros(1, 5, V)
    logits[0, t.arange(4), toks[0, 1:]] = 100.0
    assert completion_loss(logits, toks, mask).item() < 1e-3
    logits = t.randn(1, 5, V)
    lp = logits.log_softmax(-1)
    expected = -(lp[0, 1, 3] + lp[0, 2, 4] + lp[0, 3, 5]) / 3
    assert t.isclose(completion_loss(logits, toks, mask), expected)
    assert t.isclose(completion_loss(lp, toks, mask, is_logprobs=True), expected)
    full = -(lp[0, 0, 2] + lp[0, 1, 3] + lp[0, 2, 4] + lp[0, 3, 5]) / 4  # a 1 at position 0 has nothing to predict and must not count in the mean
    assert t.isclose(completion_loss(logits, toks, t.ones(1, 5, dtype=t.long)), full)
    two = t.cat([logits, logits])  # the mean is over every marked token of the batch
    assert t.isclose(completion_loss(two, t.cat([toks, toks]), t.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 0, 1]])), -(lp[0, 1, 3] + lp[0, 2, 4] + lp[0, 3, 5] + lp[0, 3, 5]) / 4)
