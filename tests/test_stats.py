import math

import numpy as np
import pytest
import torch as t
from scipy.stats import binomtest

from mechtools.stats import *

def test_small_stats():
    t.manual_seed(0)
    a, b = t.randn(16), t.randn(16)
    assert t.allclose(normed(t.stack([a, b])).norm(dim=-1), t.ones(2))
    assert math.isclose(cosine_sim(a, b), (a @ b / (a.norm() * b.norm())).item(), rel_tol=1e-5) and math.isclose(cosine_sim(a, 3 * a), 1.0, abs_tol=1e-6)
    assert math.isclose(pearson(a, b), np.corrcoef(a.numpy(), b.numpy())[0, 1], rel_tol=1e-5) and math.isclose(pearson(a, 2 * a + 1), 1.0, abs_tol=1e-6)
    with pytest.raises(AssertionError):
        pearson(a, b[:8])
    vecs = t.randn(4, 16)
    G = vecs @ vecs.T
    assert math.isclose(mean_self_sim(vecs), ((G.sum() - G.trace()) / 12).item(), rel_tol=1e-5)
    assert math.isclose(mean_self_sim(normed(t.stack([a, a, a]))), 1.0, abs_tol=1e-5)

def test_topk_vector_matches():
    t.manual_seed(0)
    vectors, test = t.randn(20, 8), t.randn(8)
    m = topk_vector_matches(vectors, test, k=3)
    dots = vectors @ test
    assert m["pos_indices"].tolist() == dots.topk(3).indices.tolist() and m["neg_indices"].tolist() == dots.topk(3, largest=False).indices.tolist()
    assert t.allclose(m["pos_sims"], dots.topk(3).values)
    c = topk_vector_matches(vectors, test, k=3, cosine=True)
    cos = normed(vectors) @ normed(test)
    assert c["pos_indices"].tolist() == cos.topk(3).indices.tolist() and t.allclose(c["pos_sims"], cos.topk(3).values)
    assert topk_vector_matches(vectors, test, k=50)["pos_indices"].shape == (20,)  # k clipped to n
    assert t.allclose(topk_vector_matches(vectors, test, k=3, normalize_test=True)["pos_sims"], (vectors @ normed(test)).topk(3).values)

def test_wilson_matches_scipy():
    for n in (1, 2, 5, 10, 50, 500):
        for k in range(n + 1):
            lo, hi = wilson(k, n)
            ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
            assert abs(lo - ci.low) < 1e-5 and abs(hi - ci.high) < 1e-5, (k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0
    assert wilson(0, 0) == (0.0, 1.0) and wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0 and wilson(3, 10, z=0) == (0.3, 0.3)
    assert wilson(50, 100)[1] - wilson(50, 100)[0] > wilson(500, 1000)[1] - wilson(500, 1000)[0]

def clusters(n_clusters: int, per: int, d: int = 8, noise: float = 0.05, seed: int = 0) -> tuple[t.Tensor, t.Tensor]:
    g = t.Generator().manual_seed(seed)
    centers = normed(t.randn(n_clusters, d, generator=g))
    x = t.cat([c + noise * t.randn(per, d, generator=g) for c in centers])
    return x, t.arange(n_clusters).repeat_interleave(per)

def objective(x: t.Tensor, labels: t.Tensor, centroids: t.Tensor) -> float:
    return (normed(x) * centroids[labels]).sum(-1).mean().item()

def test_kmeans_properties():
    x, _ = clusters(4, 50)
    labels, cents = kmeans(x, 4, iters=30)
    assert labels.shape == (200,) and cents.shape == (4, 8) and labels.min() >= 0 and labels.max() < 4
    assert t.allclose(cents.norm(dim=-1), t.ones(4), atol=1e-5)
    assert t.equal(labels, (normed(x) @ cents.T).argmax(-1))  # converged: the labels are the assignment to the returned centroids
    l2, c2 = kmeans(x, 4, iters=30)
    assert t.equal(labels, l2) and t.equal(cents, c2)  # deterministic per seed
    objs = [objective(x, *kmeans(x, 4, iters=i)) for i in (1, 2, 4, 8, 30)]
    assert all(b >= a - 1e-6 for a, b in zip(objs, objs[1:])), objs  # Lloyd's never decreases the objective
    assert kmeans(x.bfloat16(), 4, iters=2)[1].dtype == t.float32
    with pytest.raises(ValueError):
        kmeans(x[:3], 5)

def test_hierarchical_kmeans():
    # 8 tight groups: four near +e0 and four near -e0, so the fine clusters must agglomerate into those two halves
    g = t.Generator().manual_seed(0)
    base = t.zeros(8); base[0] = 1
    subs = 0.3 * normed(t.randn(8, 8, generator=g))
    x = t.cat([base * (1 if i < 4 else -1) + subs[i] + 0.02 * t.randn(30, 8, generator=g) for i in range(8)])
    labels, cents = hierarchical_kmeans(x, k_fine=8, k_coarse=2, iters=20)
    assert labels.shape == (240,) and cents.shape == (2, 8) and set(labels.tolist()) == {0, 1}
    assert len(set(labels[:120].tolist())) == 1 and len(set(labels[120:].tolist())) == 1 and labels[0] != labels[120]
    assert t.allclose(cents.norm(dim=-1), t.ones(2), atol=1e-5) and cents[labels[0], 0] > 0.9 and cents[labels[120], 0] < -0.9
