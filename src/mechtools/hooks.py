import functools
from typing import Callable

import torch as t
from torch import Tensor

def add_bias_hook(act: Tensor, hook, bias: Tensor, scale: float = 1.0, seq_pos: int | slice | list[int] | None = None, target_norm: float | None = None) -> Tensor:
    """Add `bias * scale` to the activation, at every position or only at `seq_pos`. `target_norm` rescales the bias to that norm first."""
    if target_norm is not None:
        bias = bias / bias.norm() * target_norm
    if seq_pos is None:
        return act + bias * scale
    act[:, seq_pos] += bias * scale
    return act

def make_add_bias_hook(bias: Tensor, **kw) -> functools.partial:
    return functools.partial(add_bias_hook, bias=bias, **kw)

def replace_act_hook(act: Tensor, hook, new: Tensor, seq_pos: int | slice | list[int] | None = None) -> Tensor:
    if seq_pos is None:
        return new
    act[:, seq_pos] = new
    return act

def make_sae_feat_steer_hook(sae, feat_idx: int, feat_act: float, normalize: bool = False, **kw) -> tuple[str, functools.partial]:
    """(hook_name, hook) adding `feat_act` times the feature's decoder direction. Extra kwargs go to add_bias_hook."""
    feat_dec = sae.W_dec[feat_idx].clone()
    if normalize:
        feat_dec /= feat_dec.norm()
    return sae.cfg.metadata.hook_name, make_add_bias_hook(feat_act * feat_dec, **kw)

def proj_out(A: Tensor, B: Tensor, norm: bool = False) -> Tensor:
    """A with its component along B removed."""
    out = A - B * (A @ B) / (B @ B)
    return out / out.norm() if norm else out

def scale_hook(resid: Tensor, hook, Q: Tensor, factor: float) -> Tensor:
    return resid + (factor - 1) * ((resid @ Q) @ Q.T)

def scale_hooks(dirs_by_layer: dict[int, Tensor], factor: float, hook_fmt: str = "blocks.{}.hook_resid_pre") -> list[tuple[str, Callable]]:
    """fwd_hooks that rescale the residual's projection onto span(dirs) by `factor` at each layer (0 ablates). Directions are orthonormalized so correlated ones aren't double counted."""
    return [(hook_fmt.format(layer), functools.partial(scale_hook, Q=t.linalg.qr(dirs.T.float())[0].to(dirs.dtype), factor=factor)) for layer, dirs in dirs_by_layer.items()]

def set_hook(resid: Tensor, hook, Q: Tensor, v: Tensor) -> Tensor:
    return resid - (resid @ Q) @ Q.T + v

def set_hooks(dirs_by_layer: dict[int, Tensor], target: float, hook_fmt: str = "blocks.{}.hook_resid_pre") -> list[tuple[str, Callable]]:
    """fwd_hooks that replace the residual's projection onto span(dirs) with the in-span vector whose dot product with each unit direction is `target`."""
    hooks = []
    for layer, dirs in dirs_by_layer.items():
        D = (dirs / dirs.norm(dim=-1, keepdim=True)).float()
        Q = t.linalg.qr(D.T)[0]
        v = t.linalg.pinv(D) @ t.full((D.shape[0],), target, device=D.device)
        hooks.append((hook_fmt.format(layer), functools.partial(set_hook, Q=Q.to(dirs.dtype), v=v.to(dirs.dtype))))
    return hooks
