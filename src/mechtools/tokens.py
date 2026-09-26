import copy
import html

import torch as t
from torch import Tensor
from IPython.display import HTML, display

from mechtools.colors import underline, endc

def to_ids(inp: str | Tensor | list[int] | list[dict], tokenizer, **chat_kwargs) -> list[int]:
    """Token ids of a string (with the tokenizer's special tokens, e.g. a leading BOS), one sequence of ids as a tensor or list (a leading batch dim of 1 is fine), or a chat conversation (list of role/content dicts, tokenized with apply_chat_template and chat_kwargs)."""
    if isinstance(inp, str):
        return tokenizer.encode(inp)
    if isinstance(inp, list) and inp and isinstance(inp[0], dict):
        return tokenizer.apply_chat_template(inp, tokenize=True, return_dict=False, **chat_kwargs)
    if isinstance(inp, list) and inp and isinstance(inp[0], str):
        raise TypeError("to_ids takes a string, ids, or a conversation, not str tokens (tokenizer.convert_tokens_to_ids turns those into ids)")
    ids = t.as_tensor(inp)
    if ids.squeeze().ndim > 1:
        raise ValueError(f"to_ids takes one sequence of ids, got shape {tuple(ids.shape)}")
    return ids.flatten().tolist()

def to_str_toks(inp: str | Tensor | list[int] | list[dict], tokenizer, **chat_kwargs) -> list[str]:
    return [tokenizer.decode(tok) for tok in to_ids(inp, tokenizer, **chat_kwargs)]

TOKS_CSS = "<style>.tk{cursor:default} .tk span:hover{outline:1px solid #e66} .tk span[data-p]{cursor:pointer;border-bottom:2px solid #666} .tk span[data-p].on{border-bottom-color:#e66;background:#503a3a !important}</style>"

def toks_html(strs: list[str], ids: list[int] | None = None, pos: int | list[int] | None = None, lo: int = 0, hi: int | None = None) -> str:
    """strs[lo:hi] as spans with alternating backgrounds, hover showing index, id (if given) and repr of the string. An int `pos` underlines that token; a list of positions marks each one as a clickable tab target (data-p is its index in the list; the enclosing widget gives the selected one class 'on'). Put it inside a dark monospace container."""
    hi = len(strs) if hi is None else hi
    sel = [p % len(strs) for p in ([] if pos is None else [pos] if isinstance(pos, int) else pos)] if strs else []
    tabs = {p: j for j, p in enumerate(sel)} if isinstance(pos, list) else {}
    spans = "".join(f"<span {f'data-p={tabs[i]} ' if i in tabs else ''}title='{i}{f' &middot; id {ids[i]}' if ids else ''} &middot; {html.escape(repr(s))}' style='background:{'#3c3c3c' if i % 2 else '#262626'};{'border-bottom:2px solid #e66' if i in sel and not tabs else ''}'>{html.escape(s).replace(chr(10), '↵\n')}</span>" for i, s in enumerate(strs[lo:hi], lo))
    return f"{TOKS_CSS}<div class='tk' style='white-space:pre-wrap;line-height:1.8'>{'… ' if lo > 0 else ''}{spans}{' …' if hi < len(strs) else ''}</div>"

def show_toks(inp: str | Tensor | list[int] | list[dict], tokenizer, pos: int | None = None, add_generation_prompt: bool = False, continue_final_message: bool = False, tools: list | None = None, chat_template: str | None = None, **template_kwargs):
    """Rich HTML display of a prompt's tokens, hover shows index, token id and repr, the token at pos (if given) underlined.
    A conversation (list of role/content dicts) goes through apply_chat_template; the chat kwargs and any template_kwargs (e.g. enable_thinking=False for Qwen3) are forwarded to it."""
    ids = to_ids(inp, tokenizer, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message, tools=tools, chat_template=chat_template, **template_kwargs)
    display(HTML(f"<div style='background:#111;color:#ddd;font:12px monospace;padding:8px'>{toks_html([tokenizer.decode(i) for i in ids], ids, pos)}</div>"))

def underline_stoks(toks: str | Tensor | list[int], tokenizer) -> str:
    """The tokens of `toks` as one terminal string with every other token underlined, to see the boundaries."""
    return "".join(f"{underline if i % 2 else endc}{s}" for i, s in enumerate(to_str_toks(toks, tokenizer))) + endc

def pick_sentinel(tokenizer, used_ids: set[int]) -> str:
    """The lowest-id added token not in used_ids, preferring one whose lstrip and rstrip flags are off (a token that eats the whitespace around it would shift the span it marks). Added tokens are split off atomically, so one can mark a spot in text without changing the surrounding tokenization."""
    free = [tok for i, tok in sorted(tokenizer.added_tokens_decoder.items()) if i not in used_ids]
    if not free:
        raise ValueError("every added token of the tokenizer occurs in the conversation; pass sentinel=")
    return next((tok.content for tok in free if not (tok.lstrip or tok.rstrip)), free[0].content)

def get_turn_tok_idx(conversation: list[dict], turn: int, tokenizer, idx_point: str = "start", sentinel: str | None = None, **chat_kwargs) -> int | tuple[int, int]:
    """Index of the first token holding conversation[turn]'s content ("start"), the index after the last ("end"), or both, in to_ids(conversation, tokenizer, **chat_kwargs).
    Compares against a rendering with the content replaced by a sentinel token, so tokens that merge template text with content count as content, and the span is what the template rendered from the content field: on Qwen3, a reasoning_content field renders as template text outside the span, an earlier turn's <think> block inside content is stripped by the template, and the last turn's is kept in the span."""
    ids = to_ids(conversation, tokenizer, **chat_kwargs)
    conv = copy.deepcopy(conversation)
    conv[turn]["content"] = sentinel or pick_sentinel(tokenizer, set(ids))
    marked = to_ids(conv, tokenizer, **chat_kwargs)
    start = next(i for i, (a, b) in enumerate(zip(ids, marked)) if a != b)
    end = len(ids) - next(i for i, (a, b) in enumerate(zip(ids[::-1], marked[::-1])) if a != b)
    return {"start": start, "end": end, "both": (start, end)}[idx_point]

def apply_chat_template(tokenizer, convs: str | list[str] | list[dict] | list[list[dict]], add_generation_prompt: bool = True, **chat_kwargs) -> tuple[Tensor, Tensor]:
    """(input_ids, attention_mask) [batch, seq], left-padded. Each item of convs is a conversation (list of role/content dicts) or a user prompt string; a single item gives a batch of 1.
    A tokenizer without a pad token gets its eos as pad (permanently); its padding_side is restored afterwards, also when the template raises."""
    if not convs:
        raise ValueError("no conversations to render")
    convs = [convs] if isinstance(convs, str) or isinstance(convs[0], dict) else convs
    convs = [[{"role": "user", "content": c}] if isinstance(c, str) else c for c in convs]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    side, tokenizer.padding_side = tokenizer.padding_side, "left"
    try:
        out = tokenizer.apply_chat_template(convs, return_tensors="pt", return_dict=True, padding=True, add_generation_prompt=add_generation_prompt, **chat_kwargs)
    finally:
        tokenizer.padding_side = side
    return out["input_ids"], out["attention_mask"]

def get_assistant_mask(tokenizer, convs: list[dict] | list[list[dict]], include_eot: bool = True, **chat_kwargs) -> tuple[Tensor, Tensor, Tensor]:
    """(input_ids, attention_mask, assistant_mask), all [batch, seq] left-padded. assistant_mask is 1 on the content tokens of every assistant turn (the span get_turn_tok_idx finds, which excludes a reasoning_content field and the empty think block Qwen3 renders on the last turn), plus the end-of-turn token after each if include_eot.
    The mask marks the tokens to be predicted, as completion_loss expects."""
    convs = [convs] if isinstance(convs[0], dict) else convs
    input_ids, attn = apply_chat_template(tokenizer, convs, add_generation_prompt=False, **chat_kwargs)
    mask = t.zeros_like(input_ids)
    for b, conv in enumerate(convs):
        offset = input_ids.shape[1] - attn[b].sum().item()
        for i, msg in enumerate(conv):
            if msg["role"] == "assistant":
                start, end = get_turn_tok_idx(conv, i, tokenizer, "both", **chat_kwargs)
                mask[b, offset + start:offset + end + include_eot] = 1
    return input_ids, attn, mask

def completion_loss(logits: Tensor, conv_toks: Tensor, comp_mask: Tensor, is_logprobs: bool = False) -> Tensor:
    """Mean cross-entropy over the tokens where comp_mask is 1 (position 0 has no prediction, so a 1 there is ignored). Pass logits, or already log-softmaxed logprobs with is_logprobs=True."""
    comp_mask = comp_mask.to(logits.device)[:, 1:]
    logprobs = logits if is_logprobs else logits.log_softmax(dim=-1)
    tok_losses = -logprobs[:, :-1].gather(-1, conv_toks[:, 1:, None].to(logits.device)).squeeze(-1)
    return (tok_losses * comp_mask).sum() / comp_mask.count_nonzero()
