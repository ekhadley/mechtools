import pytest
import torch as t
from transformers import AutoTokenizer

from mechtools.colors import endc, underline
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
    tok = AutoTokenizer.from_pretrained(request.param)
    if tok.pad_token is None and tok.eos_token is None:  # Inkling ships neither; any token not used in text works as pad since the attention mask covers it
        tok.pad_token = "<|unused|>"
    return tok

def runs(mask: Tensor) -> list[tuple[int, int]]:
    """(start, end) of each contiguous run of 1s."""
    edges = t.diff(t.cat([t.tensor([0]), mask, t.tensor([0])])).nonzero().flatten().tolist()
    return list(zip(edges[::2], edges[1::2]))

def test_to_ids_input_kinds(tok):
    ids = to_ids("Hello world.", tok)
    assert to_ids(t.tensor([ids]), tok) == ids == to_ids(ids, tok)
    assert to_ids(CONV, tok) == tok.apply_chat_template(CONV, tokenize=True, return_dict=False)
    assert to_str_toks(CONV, tok, add_generation_prompt=True) == [tok.decode(i) for i in to_ids(CONV, tok, add_generation_prompt=True)]

def test_toks_html():
    h = toks_html(["a<b", "\n", "c"], [1, 2, 3], pos=-1)
    assert "a&lt;b" in h and "↵" in h and h.count("border-bottom:2px solid #e66") == 1 and "id 3 &middot; &#x27;c&#x27;" in h
    h = toks_html(list("abcdefg"), lo=2, hi=5)
    assert h.count("<span") == 3 and "… " in h and " …" in h
    h = toks_html(list("abcdefg"), pos=[3, -1])
    assert h.count("<span data-p=") == 2 and "<span data-p=0 title='3" in h and "<span data-p=1 title='6" in h and "border-bottom:2px solid #e66" not in h

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

def test_apply_chat_template(tok):
    items = ["hi", CONV[:2], CONV]
    ids, attn = apply_chat_template(tok, items)
    assert ids.shape == attn.shape and ids.shape[0] == 3 and attn[:, -1].all() and not attn[0, 0]
    for b, item in enumerate(items):
        conv = [{"role": "user", "content": item}] if isinstance(item, str) else item
        assert ids[b][attn[b].bool()].tolist() == to_ids(conv, tok, add_generation_prompt=True)
    single_ids, single_attn = apply_chat_template(tok, "hi")
    assert single_ids.shape[0] == 1 and single_attn.all() and single_ids[0].tolist() == ids[0][attn[0].bool()].tolist()

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

def test_to_ids_special_tokens_and_shape(tok):
    ids = to_ids("Hello world.", tok, add_special_tokens=False)
    assert ids == tok.encode("Hello world.", add_special_tokens=False) and to_str_toks("Hello world.", tok, add_special_tokens=False) == [tok.decode(i) for i in ids]
    assert to_ids("Hello world.", tok) == tok.encode("Hello world.")  # the default adds BOS where the tokenizer does
    assert to_ids(t.tensor([[[1, 2]]]), tok) == [1, 2] == to_ids([[1, 2]], tok)
    with pytest.raises(ValueError, match="one sequence"):
        to_ids(t.tensor([[1, 2], [3, 4]]), tok)

def test_underline_stoks(tok):
    s = underline_stoks("Hello world.", tok)
    assert s.endswith(endc) and "Hello" in s and s.count(underline) >= 1

def test_apply_chat_template_leaves_the_tokenizer_alone(tok):
    before = (tok.pad_token_id, tok.padding_side)
    apply_chat_template(tok, ["hi", CONV])
    assert (tok.pad_token_id, tok.padding_side) == before
