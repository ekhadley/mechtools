import torch as t

from mechtools.stats import kmeans

def test_kmeans_shapes_and_quiet(capsys):
    x = t.randn(30, 6)
    labels, centroids = kmeans(x, 4, iters=3, quiet=True)
    assert labels.shape == (30,) and centroids.shape == (4, 6) and capsys.readouterr().err == ""
    kmeans(x, 4, iters=3)
    assert "kmeans" in capsys.readouterr().err
