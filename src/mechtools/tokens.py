import copy

import torch as t
from torch import Tensor

from mechtools.colors import underline, endc

NO_SYS_PROMPT_SUPPORT = ["gemma"]

def to_str_toks(inp: str | Tensor | list[int], tokenizer) -> list[str]:
    ids = tokenizer.encode(inp) if isinstance(inp, str) else t.as_tensor(inp).flatten().tolist()
    return [tokenizer.decode(tok) for tok in ids]

def underline_stoks(toks: str | Tensor | list[int], tokenizer) -> str:
    return "".join(f"{underline if i % 2 else endc}{s}" for i, s in enumerate(to_str_toks(toks, tokenizer))) + endc

def find_first_idx(toks: Tensor, str_tok: str, tokenizer) -> int | list[int]:
    if toks.squeeze().ndim == 2:
        return [find_first_idx(row, str_tok, tokenizer) for row in toks]
    return to_str_toks(toks, tokenizer).index(str_tok)

def get_turn_tok_idx(conversation: list[dict], turn: int, tokenizer, idx_point: str = "start", sentinel: str = "<unused77>") -> int | tuple[int, int]:
    """Token index where `conversation[turn]`'s content starts (or ends, or both). `sentinel` must be a single token in the tokenizer's vocab."""
    if idx_point == "both":
        return get_turn_tok_idx(conversation, turn, tokenizer, "start", sentinel), get_turn_tok_idx(conversation, turn, tokenizer, "end", sentinel)
    conv = copy.deepcopy(conversation)
    conv[turn]["content"] = sentinel + conv[turn]["content"] if idx_point == "start" else conv[turn]["content"] + sentinel
    return tokenizer.apply_chat_template(conv, tokenize=True, return_dict=False, add_generation_prompt=True).index(tokenizer.vocab[sentinel])

def add_system_prompt_to_messages(tokenizer, messages: list[dict], system_prompt: str | None) -> list[dict]:
    """Prepend a system message, or for tokenizers in NO_SYS_PROMPT_SUPPORT prepend the prompt to the first user message instead."""
    if not system_prompt or not system_prompt.strip():
        return list(messages)
    if not any(name in tokenizer.name_or_path for name in NO_SYS_PROMPT_SUPPORT):
        return [{"role": "system", "content": system_prompt.strip()}] + list(messages)
    out = list(messages)
    i = next(i for i, m in enumerate(out) if m["role"] == "user")
    out[i] = {"role": "user", "content": f"{system_prompt.strip()}\n\n{out[i]['content']}"}
    return out

def apply_chat_template(tokenizer, user_prompt: str | list[str], system_prompt: str | None = None, add_generation_prompt: bool = True) -> tuple[Tensor, Tensor]:
    """(input_ids, attention_mask) for one user prompt, or a left-padded batch of them."""
    if isinstance(user_prompt, str):
        out = tokenizer.apply_chat_template(add_system_prompt_to_messages(tokenizer, [{"role": "user", "content": user_prompt}], system_prompt), return_tensors="pt", return_dict=True, add_generation_prompt=add_generation_prompt)
        return out["input_ids"], out["attention_mask"]
    convs = [add_system_prompt_to_messages(tokenizer, [{"role": "user", "content": up}], system_prompt) for up in user_prompt]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    side, tokenizer.padding_side = tokenizer.padding_side, "left"
    out = tokenizer.apply_chat_template(convs, return_tensors="pt", return_dict=True, add_generation_prompt=add_generation_prompt, padding=True)
    tokenizer.padding_side = side
    return out["input_ids"], out["attention_mask"]

def get_assistant_mask(tokenizer, conversations: list[dict] | list[list[dict]], pad: bool = False, device: str = "cuda") -> tuple[list[int] | list[list[int]], Tensor]:
    """Token ids of each (user, assistant) conversation and a mask that is 1 on the assistant's content tokens only."""
    single = isinstance(conversations[0], dict)
    if single:
        conversations = [conversations]
    all_tok_ids, masks = [], []
    for conv in conversations:
        prefix_ids = tokenizer.apply_chat_template([conv[0]], return_dict=False, add_generation_prompt=True)
        empty_asst_ids = tokenizer.apply_chat_template([conv[0], {"role": "assistant", "content": ""}], return_dict=False)
        full_ids = tokenizer.apply_chat_template(conv, return_dict=False)
        turn_end_len = len(empty_asst_ids) - len(prefix_ids)
        all_tok_ids.append(full_ids)
        masks.append([0] * len(prefix_ids) + [1] * (len(full_ids) - len(prefix_ids) - turn_end_len - 1) + [0] * (turn_end_len + 1))
    if pad and not single:
        max_len = max(len(ids) for ids in all_tok_ids)
        for i in range(len(all_tok_ids)):
            pad_ids, pad_mask = [tokenizer.pad_token_id] * (max_len - len(all_tok_ids[i])), [0] * (max_len - len(all_tok_ids[i]))
            all_tok_ids[i] = pad_ids + all_tok_ids[i] if tokenizer.padding_side == "left" else all_tok_ids[i] + pad_ids
            masks[i] = pad_mask + masks[i] if tokenizer.padding_side == "left" else masks[i] + pad_mask
    masks = t.tensor(masks, device=device)
    return (all_tok_ids[0], masks[0]) if single else (all_tok_ids, masks)

def completion_loss(logits: Tensor, conv_toks: Tensor, comp_mask: Tensor, is_logprobs: bool = False) -> Tensor:
    """Mean cross-entropy over the tokens where comp_mask is 1. Pass logits, or already log-softmaxed logprobs with is_logprobs=True."""
    comp_mask = comp_mask.to(logits.device)
    logprobs = logits if is_logprobs else logits.log_softmax(dim=-1)
    tok_losses = -logprobs[:, :-1].gather(-1, conv_toks[:, 1:, None].to(logits.device)).squeeze(-1)
    return (tok_losses * comp_mask[:, 1:]).sum() / comp_mask.count_nonzero()
