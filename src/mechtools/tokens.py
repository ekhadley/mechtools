import copy
import html
from collections.abc import Callable, Iterable

import torch as t
from torch import Tensor
from IPython.display import HTML, display

from mechtools.colors import underline, endc

def to_ids(inp: str | Tensor | list[int] | list[dict], tokenizer, add_special_tokens: bool = True, **chat_kwargs) -> list[int]:
    """Token ids of a string, one sequence of ids as a tensor or list (shape [seq] or [1, seq]), or a chat conversation (list of role/content dicts, tokenized with apply_chat_template and chat_kwargs).
    add_special_tokens applies to a string: BOS on tokenizers that add one, so pass False for a self-rendered template string. A conversation's special tokens come from the template."""
    if isinstance(inp, str):
        return tokenizer.encode(inp, add_special_tokens=add_special_tokens)
    if isinstance(inp, list) and inp and isinstance(inp[0], dict):
        return tokenizer.apply_chat_template(inp, tokenize=True, return_dict=False, **chat_kwargs)
    if isinstance(inp, list) and inp and isinstance(inp[0], str):
        raise TypeError("to_ids takes a string, ids, or a conversation, not str tokens (tokenizer.convert_tokens_to_ids turns those into ids)")
    ids = t.as_tensor(inp)
    if ids.ndim > 1 and any(d != 1 for d in ids.shape[:-1]):
        raise ValueError(f"to_ids takes one sequence of ids, got shape {tuple(ids.shape)}")
    return ids.flatten().tolist()

def to_str_toks(inp: str | Tensor | list[int] | list[dict], tokenizer, add_special_tokens: bool = True, **chat_kwargs) -> list[str]:
    return [tokenizer.decode(tok) for tok in to_ids(inp, tokenizer, add_special_tokens, **chat_kwargs)]

TOKS_CSS = "<style>.tk{cursor:default;position:relative} .tk span{background:linear-gradient(var(--v,#0000),var(--v,#0000)),#1c1c1c} .tk span:hover{background:#3d4a5c;anchor-name:--tk} .tk span:hover::after{content:attr(data-h);position:absolute;position-anchor:--tk;position-area:bottom span-right;align-self:unsafe start;position-try-fallbacks:flip-inline,flip-block,flip-block flip-inline;z-index:9;background:#000;color:#ddd;border:1px solid #888;padding:1px 6px;font:11px/1.5 monospace;white-space:pre;pointer-events:none} .tk span[data-p]{cursor:pointer;outline:1px solid #666} .tk span[data-p].on{outline-color:#fc6}</style>"

def toks_html(strs: list[str], ids: list[int] | None = None, pos: int | list[int] | None = None, lo: int = 0, hi: int | None = None, vals: list[float] | Tensor | None = None, val_name: str = "value") -> str:
    """strs[lo:hi] as spans on one flat background, the hovered token highlighted, hover showing position, id (if given) and repr of the string. The hover is a CSS tooltip under the token (CSS anchor positioning), flipping left or above at the strip's edges. An int `pos` puts an amber outline on that token; a list of positions outlines each one in gray as a clickable tab target (data-p is its index in the list; the enclosing widget gives the selected one class 'on', which turns its outline amber, and the first one carries it from the start, so it reads as selected where no script runs).
    `vals`, one scalar per token, fills each token by its value: red for positive, blue for negative, opacity 0.85 |v| / max|v|; the hover then also shows `val_name` and the value. Put it inside a dark monospace container."""
    hi = len(strs) if hi is None else hi
    sel = [p % len(strs) for p in ([] if pos is None else [pos] if isinstance(pos, int) else pos)] if strs else []
    tabs = {p: j for j, p in enumerate(sel)} if isinstance(pos, list) else {}
    vals = None if vals is None else t.as_tensor(vals).flatten().float().tolist()
    if vals is not None and len(vals) != len(strs):
        raise ValueError(f"{len(vals)} values for {len(strs)} tokens")
    scale = max((abs(v) for v in vals), default=0) or 1 if vals else 1
    shade = lambda i: f"--v:rgba({'230,80,80' if vals[i] >= 0 else '80,130,230'},{0.85 * abs(vals[i]) / scale:.3f});" if vals else ""
    spans = "".join(f"<span {f'data-p={tabs[i]} ' if i in tabs else ''}{'class=on ' if tabs.get(i) == 0 else ''}data-h='pos {i}{f' &middot; id {ids[i]}' if ids else ''} &middot; {html.escape(repr(s))}{f' &middot; {html.escape(val_name)} {vals[i]:.4g}' if vals else ''}' style='{shade(i)}{'outline:1px solid #fc6' if i in sel and not tabs else ''}'>{html.escape(s).replace(chr(10), '↵\n')}</span>" for i, s in enumerate(strs[lo:hi], lo))
    return f"{TOKS_CSS}<div class='tk' style='white-space:pre-wrap;line-height:1.8'>{'… ' if lo > 0 else ''}{spans}{' …' if hi < len(strs) else ''}</div>"

def show_toks(inp: str | Tensor | list[int] | list[dict], tokenizer, pos: int | None = None, vals: list[float] | Tensor | None = None, val_name: str = "value", title: str | None = None, add_special_tokens: bool = True, add_generation_prompt: bool = False, continue_final_message: bool = False, tools: list | None = None, chat_template: str | None = None, **template_kwargs):
    """Rich HTML display of a prompt's tokens, hover shows position, token id and repr, the token at pos (if given) outlined in amber.
    `vals`, one scalar per token (list or tensor of shape [seq] or [1, seq]), shades each token by its value (red positive, blue negative, opacity relative to max |v|), names it `val_name` in the hover, and puts the value range in the header line with `title`.
    add_special_tokens applies to a string (see to_ids). A conversation (list of role/content dicts) goes through apply_chat_template; the chat kwargs and any template_kwargs (e.g. enable_thinking=False for Qwen3) are forwarded to it."""
    ids = to_ids(inp, tokenizer, add_special_tokens, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message, tools=tools, chat_template=chat_template, **template_kwargs)
    strip = toks_html([tokenizer.decode(i) for i in ids], ids, pos, vals=vals, val_name=val_name)
    rng = f"<span style='color:#888;font-weight:normal'>{html.escape(val_name)} &isin; [{t.as_tensor(vals).min():.3g}, {t.as_tensor(vals).max():.3g}]</span>" if vals is not None else ""
    head = f"<div style='margin-bottom:6px;font-weight:bold'>{html.escape(title) if title else ''}{' &middot; ' if title and rng else ''}{rng}</div>" if title or rng else ""
    display(HTML(f"<div style='background:#111;color:#ddd;font:12px monospace;padding:8px'>{head}{strip}</div>"))

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
    Compares against a rendering with the content replaced by a sentinel token, so tokens that merge template text with content count as content, and the span is what the template rendered from the content field: on Qwen3, a reasoning_content field renders as template text outside the span, an earlier turn's <think> block inside content is stripped by the template, and the last turn's is in the span from after its opening <think> tag, which stays template text."""
    ids = to_ids(conversation, tokenizer, **chat_kwargs)
    conv = copy.deepcopy(conversation)
    conv[turn]["content"] = sentinel or pick_sentinel(tokenizer, set(ids))
    marked = to_ids(conv, tokenizer, **chat_kwargs)
    start = next(i for i, (a, b) in enumerate(zip(ids, marked)) if a != b)
    end = len(ids) - next(i for i, (a, b) in enumerate(zip(ids[::-1], marked[::-1])) if a != b)
    return {"start": start, "end": end, "both": (start, end)}[idx_point]

def apply_chat_template(tokenizer, convs: str | list[str] | list[dict] | list[list[dict]], add_generation_prompt: bool = True, **chat_kwargs) -> tuple[Tensor, Tensor]:
    """(input_ids, attention_mask) [batch, seq], left-padded. Each item of convs is a conversation (list of role/content dicts) or a user prompt string; a single item gives a batch of 1.
    Each row is to_ids(conv, tokenizer, add_generation_prompt=..., **chat_kwargs), padded with the tokenizer's pad token or, when it has none, its eos (any id works: the mask covers it). The tokenizer is not modified."""
    if not convs:
        raise ValueError("no conversations to render")
    convs = [convs] if isinstance(convs, str) or isinstance(convs[0], dict) else convs
    convs = [[{"role": "user", "content": c}] if isinstance(c, str) else c for c in convs]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("the tokenizer has neither a pad nor an eos token; set tokenizer.pad_token")
    rows = [to_ids(c, tokenizer, add_generation_prompt=add_generation_prompt, **chat_kwargs) for c in convs]
    width = max(len(r) for r in rows)
    input_ids = t.tensor([[pad] * (width - len(r)) + r for r in rows])
    attn = t.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])
    return input_ids, attn

def get_assistant_mask(tokenizer, convs: list[dict] | list[list[dict]], include_eot: bool = True, **chat_kwargs) -> tuple[Tensor, Tensor, Tensor]:
    """(input_ids, attention_mask, assistant_mask), all [batch, seq] left-padded. assistant_mask is 1 on the content tokens of every assistant turn (the span get_turn_tok_idx finds, which excludes a reasoning_content field and the empty think block Qwen3 renders on the last turn), plus the end-of-turn token after each if include_eot.
    The mask marks the tokens to be predicted, as completion_loss expects. With include_eot the token after each assistant turn's content must be a special token, else a ValueError says so."""
    convs = [convs] if isinstance(convs[0], dict) else convs
    input_ids, attn = apply_chat_template(tokenizer, convs, add_generation_prompt=False, **chat_kwargs)
    mask = t.zeros_like(input_ids)
    for b, conv in enumerate(convs):
        offset = input_ids.shape[1] - attn[b].sum().item()
        for i, msg in enumerate(conv):
            if msg["role"] == "assistant":
                start, end = get_turn_tok_idx(conv, i, tokenizer, "both", **chat_kwargs)
                if include_eot:
                    eot = input_ids[b, offset + end].item() if offset + end < input_ids.shape[1] else None
                    if eot not in tokenizer.added_tokens_decoder:
                        raise ValueError(f"the token after assistant turn {i}'s content is {None if eot is None else tokenizer.decode([eot])!r} (id {eot}), not a special token: this template does not end a turn with one token; pass include_eot=False")
                mask[b, offset + start:offset + end + include_eot] = 1
    return input_ids, attn, mask

def completion_loss(logits: Tensor, conv_toks: Tensor, comp_mask: Tensor, is_logprobs: bool = False) -> Tensor:
    """Mean cross-entropy over the tokens where comp_mask is 1 (position 0 has no prediction, so a 1 there is ignored). Pass logits, or already log-softmaxed logprobs with is_logprobs=True."""
    comp_mask = comp_mask.to(logits.device)[:, 1:]
    logprobs = logits if is_logprobs else logits.log_softmax(dim=-1)
    tok_losses = -logprobs[:, :-1].gather(-1, conv_toks[:, 1:, None].to(logits.device)).squeeze(-1)
    return (tok_losses * comp_mask).sum() / comp_mask.count_nonzero()

def single_token_marker(tokenizer, render: Callable[[str], list[int]], chars: Iterable[str] | None = None) -> tuple[str, int, list[int], int]:
    """A character that is one token on its own and occurs as exactly one token in the rendered prompt: (char, its id, the rendered ids, the index of that token). `render(char)` returns the ids of the whole prompt with the character in place, chat template included, since BPE merges depend on the neighbors: a character that is one token alone can merge with a tag once inside the template (the oracle lens card's first candidate did).
    `chars` defaults to the enclosed CJK letters U+3200 to U+33FF, the range the oracle lens and NLA checkpoints were trained with; on a Qwen3 tokenizer the scan lands on ㈎ (149705) and on Qwen3.6's on ㈜ (158983), the ids their cards name. Raises when no character survives."""
    for char in map(chr, range(0x3200, 0x3400)) if chars is None else chars:
        ids = tokenizer.encode(char, add_special_tokens=False)
        if len(ids) != 1:
            continue
        rendered = render(char)
        slots = [i for i, x in enumerate(rendered) if x == ids[0]]
        if len(slots) == 1:
            return char, ids[0], rendered, slots[0]
    raise ValueError("no character in the range is a single token that occurs once in the rendered prompt")
