import functools
from collections.abc import Callable

import torch as t
from torch import Tensor

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
    """fwd_hooks that replace the residual's projection onto span(dirs) with the in-span vector whose dot product with each unit direction is `target`. `dirs` is [r, d] or a single [d] direction."""
    hooks = []
    for layer, dirs in dirs_by_layer.items():
        D = t.atleast_2d(dirs.float())
        D = D / D.norm(dim=-1, keepdim=True)
        v = t.linalg.pinv(D) @ t.full((D.shape[0],), float(target), device=D.device)
        hooks.append((hook_fmt.format(layer), functools.partial(set_hook, Q=orthonormal_basis(D), v=v)))
    return hooks
