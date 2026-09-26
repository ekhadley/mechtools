import pytest
import torch as t

from mechtools.hooks import *

D = 8

@pytest.fixture
def dirs():
    t.manual_seed(0)
    v = t.randn(D); v /= v.norm()
    w = t.randn(D); w -= (w @ v) * v; w /= w.norm()  # orthogonal to v
    return v, w

def test_proj_out_vector_and_stack(dirs):
    v, w = dirs
    a = 3 * v + 2 * w
    assert t.allclose(proj_out(a, v), 2 * w, atol=1e-6) and t.allclose(proj_out(a, v, norm=True), w, atol=1e-6)
    for n in (D, 3):  # n == d used to be silently wrong, n != d used to raise
        A = t.randn(n, D)
        out = proj_out(A, v)
        assert out.shape == A.shape and t.allclose(out, t.stack([proj_out(row, v) for row in A]), atol=1e-6)
        assert (out @ v).abs().max() < 1e-5
        assert t.allclose(proj_out(A, v, norm=True).norm(dim=-1), t.ones(n), atol=1e-5)
    assert proj_out(t.randn(2, 3, D), v).shape == (2, 3, D)

def test_orthonormal_basis(dirs):
    v, w = dirs
    assert orthonormal_basis(v).shape == (D, 1) and orthonormal_basis(t.stack([v, v, 2 * v])).shape == (D, 1)  # dependent rows collapse
    Q = orthonormal_basis(t.stack([v, v + 0.5 * w]))
    assert Q.shape == (D, 2) and t.allclose(Q.T @ Q, t.eye(2), atol=1e-5)
    assert t.allclose(Q @ (Q.T @ w), w, atol=1e-5) and (Q.T @ t.randn(D)).shape == (2,)

def test_scale_hooks_factor_and_dependent_dirs(dirs):
    v, w = dirs
    resid = (3 * v + 2 * w)[None, None]
    for factor in (0.0, 1.0, 2.0):
        (name, hook), = scale_hooks({4: v}, factor)
        out = hook(resid, None)
        assert name == "blocks.4.hook_resid_pre" and t.allclose(out @ v, t.tensor([[3 * factor]]), atol=1e-5) and t.allclose(out @ w, t.tensor([[2.0]]), atol=1e-5)
    (_, hook), = scale_hooks({0: t.stack([v, v])}, 0.0)  # a repeated direction must not ablate anything else
    out = hook(resid, None)
    assert t.allclose(out, 2 * w[None, None], atol=1e-5)
    (_, hook), = scale_hooks({0: t.stack([v, w])}, 0.0)
    assert t.allclose(hook(resid, None), t.zeros(1, 1, D), atol=1e-5)
    assert [n for n, _ in scale_hooks({1: v, 3: w}, 0.5, "blocks.{}.hook_resid_post")] == ["blocks.1.hook_resid_post", "blocks.3.hook_resid_post"]

def test_set_hooks(dirs):
    v, w = dirs
    resid = (3 * v + 2 * w)[None, None]
    for stack in (v, t.stack([v, v]), t.stack([2 * v, v + 0.5 * w])):  # one, repeated, correlated directions
        (_, hook), = set_hooks({0: stack}, 0.5)
        out = hook(resid, None)
        units = t.atleast_2d(stack) / t.atleast_2d(stack).norm(dim=-1, keepdim=True)
        assert t.allclose(out[0, 0] @ units.T, t.full((units.shape[0],), 0.5), atol=1e-5)
    (_, hook), = set_hooks({0: t.stack([v, v])}, 0.5)
    assert t.allclose(hook(resid, None) @ w, t.tensor([[2.0]]), atol=1e-5)  # the orthogonal complement is untouched
    u = t.randn(D); u -= (u @ v) * v; u -= (u @ w) * w
    (_, hook), = set_hooks({0: t.stack([v, w])}, 0.0)
    assert t.allclose(hook((v + w + u)[None, None], None), u[None, None], atol=1e-5)

def test_hooks_keep_resid_dtype(dirs):
    v, _ = dirs
    resid = t.randn(1, 2, D, dtype=t.bfloat16)
    (_, sh), = scale_hooks({0: v}, 0.0)
    (_, st), = set_hooks({0: v}, 0.5)
    assert sh(resid, None).dtype == t.bfloat16 and st(resid, None).dtype == t.bfloat16

@pytest.mark.parametrize("seq_pos", [None, 1, slice(0, 2), [0, 2]])
def test_add_bias_hook(seq_pos):
    t.manual_seed(0)
    act = t.randn(2, 3, D, dtype=t.bfloat16)
    before = act.clone()
    bias = t.randn(D)
    out = add_bias_hook(act, None, bias, scale=2.0, seq_pos=seq_pos)
    assert out.dtype == t.bfloat16 and out.shape == act.shape and t.equal(act, before)  # same dtype on every path, input untouched
    positions = list(range(3)) if seq_pos is None else [seq_pos] if isinstance(seq_pos, int) else list(range(3))[seq_pos] if isinstance(seq_pos, slice) else seq_pos
    for p in range(3):
        expected = act[:, p] + (2 * bias).bfloat16() if p in positions else act[:, p]  # the delta is cast to the activation's dtype, then added
        assert t.equal(out[:, p], expected)
    hook = make_add_bias_hook(bias, seq_pos=seq_pos, target_norm=3.0, scale=0.5)
    delta = (hook(t.zeros(1, 3, D), None)[0, positions[0]])
    assert t.isclose(delta.norm(), t.tensor(1.5), atol=1e-5) and t.allclose(delta / delta.norm(), bias / bias.norm(), atol=1e-5)

def test_replace_act_hook():
    act = t.randn(1, 3, D, dtype=t.bfloat16)
    before = act.clone()
    new = t.ones(D)
    assert replace_act_hook(act, None, new).dtype == t.bfloat16
    out = replace_act_hook(act, None, new, seq_pos=[0, 2])
    assert t.equal(act, before) and t.equal(out[0, 0], t.ones(D, dtype=t.bfloat16)) and t.equal(out[0, 1], act[0, 1]) and t.equal(out[0, 2], t.ones(D, dtype=t.bfloat16))

def test_make_sae_feat_steer_hook():
    class Meta: hook_name = "blocks.3.hook_resid_post"
    class Cfg: metadata = Meta()
    class Sae: W_dec = t.randn(5, D); cfg = Cfg()
    sae = Sae()
    before = sae.W_dec.clone()
    name, hook = make_sae_feat_steer_hook(sae, 2, 4.0, normalize=True, seq_pos=1)
    out = hook(t.zeros(1, 2, D), None)
    assert name == "blocks.3.hook_resid_post" and t.equal(sae.W_dec, before)  # normalize must not modify the SAE's weights
    assert t.isclose(out[0, 1].norm(), t.tensor(4.0), atol=1e-5) and t.allclose(out[0, 1] / 4, sae.W_dec[2] / sae.W_dec[2].norm(), atol=1e-5) and t.equal(out[0, 0], t.zeros(D))
    _, hook = make_sae_feat_steer_hook(sae, 2, 2.0)
    assert t.allclose(hook(t.zeros(1, 1, D), None)[0, 0], 2 * sae.W_dec[2], atol=1e-5)

@pytest.mark.hf
def test_hooks_on_bridge(tiny_bridge):
    model = tiny_bridge
    ids = t.tensor([model.tokenizer.encode("Hello there")])
    base = model(ids)
    bias = t.randn(model.cfg.d_model)
    assert not t.allclose(model.run_with_hooks(ids, fwd_hooks=[("blocks.0.hook_resid_pre", make_add_bias_hook(bias))]), base)
    assert not t.allclose(model.run_with_hooks(ids, fwd_hooks=[("blocks.0.hook_resid_pre", make_add_bias_hook(bias, seq_pos=-1))]), base)
    assert t.allclose(model.run_with_hooks(ids, fwd_hooks=scale_hooks({0: bias}, 1.0)), base, atol=1e-5)
    assert not t.allclose(model.run_with_hooks(ids, fwd_hooks=scale_hooks({0: bias}, 0.0)), base)
    assert not t.allclose(model.run_with_hooks(ids, fwd_hooks=set_hooks({1: bias}, 5.0)), base)
