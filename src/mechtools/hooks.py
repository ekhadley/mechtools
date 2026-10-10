import functools
from collections.abc import Callable

import torch as t
from torch import Tensor

from mechtools.sampling import eos_ids

def add_bias_hook(act: Tensor, hook, bias: Tensor, scale: float = 1.0, seq_pos: int | slice | list[int] | None = None, target_norm: float | None = None) -> Tensor:
    """act + bias * scale at every position, or only at `seq_pos` (an int, a slice, or a list of ints). `target_norm` rescales the bias to that norm first.
    The bias is moved to the activation's device and cast to its dtype, so the hook returns the activation's dtype on every path; the activation itself is not modified."""
    delta = bias.to(act.device).float()
    if target_norm is not None:
        delta = delta / delta.norm() * target_norm
    delta = (delta * scale).to(act.dtype)
    if seq_pos is None:
        return act + delta
    out = act.clone()
    out[:, seq_pos] += delta
    return out

def make_add_bias_hook(bias: Tensor, **kw) -> functools.partial:
    return functools.partial(add_bias_hook, bias=bias, **kw)

def replace_act_hook(act: Tensor, hook, new: Tensor, seq_pos: int | slice | list[int] | None = None) -> Tensor:
    """`new` (cast to the activation's dtype and device) in place of the whole activation, or written into it at `seq_pos`; the activation itself is not modified."""
    new = new.to(act.device, act.dtype)
    if seq_pos is None:
        return new
    out = act.clone()
    out[:, seq_pos] = new
    return out

def make_sae_feat_steer_hook(sae, feat_idx: int, feat_act: float, normalize: bool = False, **kw) -> tuple[str, functools.partial]:
    """(hook_name, hook) adding `feat_act` times the feature's decoder direction. Extra kwargs go to add_bias_hook."""
    feat_dec = sae.W_dec[feat_idx].clone()
    if normalize:
        feat_dec /= feat_dec.norm()
    return sae.cfg.metadata.hook_name, make_add_bias_hook(feat_act * feat_dec, **kw)

def proj_out(A: Tensor, B: Tensor, norm: bool = False) -> Tensor:
    """A with its component along B removed: one [d] vector, or a stack [..., d] handled row by row. `norm` rescales each result to unit norm."""
    B = B.to(A.device, A.dtype)
    out = A - (A @ B)[..., None] * B / (B @ B)
    return out / out.norm(dim=-1, keepdim=True) if norm else out

def orthonormal_basis(dirs: Tensor) -> Tensor:
    """[d, r] orthonormal columns spanning the rows of `dirs` ([r, d], or a single [d] direction), by SVD with the matrix_rank cutoff, so duplicated or linearly dependent directions do not add spurious dimensions the way QR would."""
    D = t.atleast_2d(dirs.float())
    _, S, Vh = t.linalg.svd(D, full_matrices=False)
    return Vh[S > S.max() * max(D.shape) * t.finfo(S.dtype).eps].T

def scale_hook(resid: Tensor, hook, Q: Tensor, factor: float) -> Tensor:
    Q = Q.to(resid.device, resid.dtype)
    return resid + (factor - 1) * ((resid @ Q) @ Q.T)

def scale_hooks(dirs_by_layer: dict[int, Tensor], factor: float, hook_fmt: str = "blocks.{}.hook_resid_pre") -> list[tuple[str, Callable]]:
    """fwd_hooks that rescale the residual's projection onto span(dirs) by `factor` at each layer (0 ablates). `dirs` is [r, d] or a single [d] direction; the span is orthonormalized, so correlated or repeated directions aren't double counted."""
    return [(hook_fmt.format(layer), functools.partial(scale_hook, Q=orthonormal_basis(dirs), factor=factor)) for layer, dirs in dirs_by_layer.items()]

def set_hook(resid: Tensor, hook, Q: Tensor, v: Tensor) -> Tensor:
    Q, v = Q.to(resid.device, resid.dtype), v.to(resid.device, resid.dtype)
    return resid - (resid @ Q) @ Q.T + v

def set_hooks(dirs_by_layer: dict[int, Tensor], target: float, hook_fmt: str = "blocks.{}.hook_resid_pre") -> list[tuple[str, Callable]]:
    """fwd_hooks that replace the residual's projection onto span(dirs) with the in-span vector whose dot product with each unit direction is `target` (the least-squares vector, by pinv, when no vector satisfies all of them, e.g. for opposite directions). `dirs` is [r, d] or a single [d] direction."""
    hooks = []
    for layer, dirs in dirs_by_layer.items():
        D = t.atleast_2d(dirs.float())
        D = D / D.norm(dim=-1, keepdim=True)
        v = t.linalg.pinv(D) @ t.full((D.shape[0],), float(target), device=D.device)
        hooks.append((hook_fmt.format(layer), functools.partial(set_hook, Q=orthonormal_basis(D), v=v)))
    return hooks

def decoder_layers(model) -> list:
    """The decoder blocks of an HF causal LM, a PeftModel around one, or a Bridge (through its original_model). Block L's output is the residual stream after layer L: `blocks.L.hook_resid_post` in the Bridge, HF's hidden_states[L + 1], and what the oracle lens, NLA and activation oracle checkpoints call "layer L"."""
    return getattr(model, "original_model", model).get_decoder().layers

@t.no_grad()
def inject_generate(model, tokenizer, ids: list[int], module, positions: list[int], vecs: Tensor, write: Callable[[Tensor, Tensor], Tensor], max_new_tokens: int = 256, seed: int | None = 0, **generate_kwargs) -> list[str]:
    """Generate from the prompt `ids` with vectors written into `module`'s output at `positions` on the prefill, one row of the batch per row of `vecs` ([n, len(positions), d], or [n, d] for a single position), and return each row's continuation decoded without special tokens, cut at the first id in eos_ids(model, tokenizer).
    `write(orig, vecs)` gets the original rows and the vectors, both [n, k, d] float32 on the activation's device, and returns what replaces the rows, e.g. `lambda o, v: 16000 * normed(v)` or `lambda o, v: o + o.norm(dim=-1, keepdim=True) * normed(v)`. A forward with fewer positions than the prompt is a decode step, left alone, so sampled tokens are never written over; without a KV cache every step holds the whole sequence and is written at the same positions. The hook is removed when generation ends, also on an exception. `seed` seeds torch first, so the same vectors give the same samples (None leaves the RNG alone). `generate_kwargs` (do_sample, temperature, top_p, top_k, ...) go to generate, whose own defaults come from the model's generation config, so pass them all. `model` is any HF causal LM or PeftModel, with the adapter you want already active."""
    vecs = vecs[:, None] if vecs.ndim == 2 else vecs
    if vecs.ndim != 3 or vecs.shape[1] != len(positions):
        raise ValueError(f"vecs must be [n, {len(positions)}, d] for {len(positions)} positions, got {tuple(vecs.shape)}")
    prompt = t.tensor([ids] * vecs.shape[0], device=next(model.parameters()).device)
    def hook(mod, args, out):
        resid = out[0] if isinstance(out, tuple) else out
        if resid.shape[1] < len(ids):
            return out
        new = resid.clone()
        orig = new[:, positions].float()
        new[:, positions] = write(orig, vecs.to(orig.device).float()).to(new.dtype)
        return (new, *out[1:]) if isinstance(out, tuple) else new
    handle = module.register_forward_hook(hook)
    try:
        if seed is not None:
            t.manual_seed(seed)
        eos = sorted(eos_ids(model, tokenizer))
        out = model.generate(input_ids=prompt, attention_mask=t.ones_like(prompt), max_new_tokens=max_new_tokens, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0], **generate_kwargs)
    finally:
        handle.remove()
    rows = out[:, len(ids):].tolist()
    return [tokenizer.decode(row[:next((i for i, x in enumerate(row) if x in eos), len(row))], skip_special_tokens=True) for row in rows]
