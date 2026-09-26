import math

import torch as t

from mechtools.stats import hierarchical_kmeans, kmeans, wilson

def test_wilson():
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.35
    lo, hi = wilson(10, 10)
    assert 0.65 < lo < 0.8 and hi == 1.0
    lo, hi = wilson(5, 10)
    assert math.isclose(lo, 0.2366, abs_tol=1e-3) and math.isclose(hi, 0.7634, abs_tol=1e-3)
    assert wilson(50, 100)[1] - wilson(50, 100)[0] < hi - lo

def three_clusters(n: int = 50) -> t.Tensor:
    g = t.Generator().manual_seed(0)
    return t.cat([c + 0.05 * t.randn(n, 3, generator=g) for c in t.eye(3)])

def test_kmeans_converges_and_labels_the_clusters(capsys):
    labels, centroids = kmeans(three_clusters(), 3, iters=50)
    assert labels.shape == (150,) and centroids.shape == (3, 3) and t.allclose(centroids.norm(dim=-1), t.ones(3), atol=1e-5)
    assert [len(set(labels[i * 50:(i + 1) * 50].tolist())) for i in range(3)] == [1, 1, 1] and len(set(labels.tolist())) == 3
    assert "kmeans:" not in capsys.readouterr().out  # converged with no empty cluster: nothing to note

def test_kmeans_notes_non_convergence_and_empty_clusters(capsys):
    kmeans(three_clusters(), 3, iters=1)
    assert "not converged after 1 iterations" in capsys.readouterr().out
    labels, centroids = kmeans(t.tensor([[1.0, 0.0]] * 4), 3, iters=5)  # one direction only, so two clusters must stay empty
    assert "2 of 3 clusters are empty" in capsys.readouterr().out and (centroids.norm(dim=-1) == 0).sum() == 2 and len(set(labels.tolist())) == 1

def test_hierarchical_kmeans():
    labels, centroids = hierarchical_kmeans(three_clusters(), k_fine=6, k_coarse=3, iters=20)
    assert labels.shape == (150,) and centroids.shape == (3, 3) and labels.max() <= 2
    assert [len(set(labels[i * 50:(i + 1) * 50].tolist())) for i in range(3)] == [1, 1, 1] and len(set(labels.tolist())) == 3
