import math

import torch as t
from torch import Tensor
from tqdm import trange
from scipy.cluster.hierarchy import linkage, fcluster

def normed(x: Tensor) -> Tensor:
    return t.nn.functional.normalize(x, dim=-1)

def cosine_sim(a: Tensor, b: Tensor) -> float:
    return (normed(a.float()) @ normed(b.float())).item()

def pearson(a: Tensor, b: Tensor) -> float:
    assert a.shape == b.shape, f"a and b should have same shape, got {a.shape} and {b.shape}"
    a_c, b_c = a.float() - a.float().mean(), b.float() - b.float().mean()
    return ((a_c * b_c).sum() / t.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())).item()

def mean_self_sim(vecs: Tensor) -> float:
    """Mean off-diagonal dot product of a [n, d] stack of vectors."""
    G = vecs.float() @ vecs.float().T
    return ((G.sum() - G.trace()) / (G.numel() - G.shape[0])).item()

def topk_vector_matches(vectors: Tensor, test_vector: Tensor, k: int = 10, normalize_test: bool = False, cosine: bool = False) -> dict:
    """Top-k most positive and most negative matches of a [n, d] stack against a [d] vector, by dot product or cosine."""
    vectors, test_vector = vectors.float(), test_vector.float().to(vectors.device)
    sims = (normed(vectors) if cosine else vectors) @ (normed(test_vector) if cosine or normalize_test else test_vector)
    pos, neg = sims.topk(min(k, len(sims))), sims.topk(min(k, len(sims)), largest=False)
    return {"pos_indices": pos.indices, "pos_sims": pos.values, "neg_indices": neg.indices, "neg_sims": neg.values}

def kmeans(x: Tensor, k: int, iters: int = 50, seed: int = 0) -> tuple[Tensor, Tensor]:
    """Spherical k-means from k random rows of x (Lloyd's iterations, no restarts, so a fixed seed is a fixed local optimum). x: [n, d]. Returns (labels [n], centroids [k, d]); a cluster that empties keeps a zero centroid."""
    if k > len(x):
        raise ValueError(f"k={k} clusters for {len(x)} points")
    x = normed(x.float())
    centroids = x[t.randperm(len(x), generator=t.Generator(device=x.device).manual_seed(seed), device=x.device)[:k]]
    for _ in trange(iters, desc="kmeans"):
        labels = (x @ centroids.T).argmax(-1)
        centroids = normed(t.zeros_like(centroids).index_add_(0, labels, x))
    return labels, centroids

def hierarchical_kmeans(x: Tensor, k_fine: int = 4096, k_coarse: int = 512, iters: int = 50, seed: int = 0) -> tuple[Tensor, Tensor]:
    """Spherical k-means with k_fine clusters, then average-linkage cosine agglomeration of the non-empty fine centroids into k_coarse groups. Returns (labels [n], centroids [k_coarse, d])."""
    fine_labels, fine_centroids = kmeans(x, k_fine, iters, seed)
    alive = t.bincount(fine_labels, minlength=k_fine) > 0
    merge = t.full((k_fine,), -1, device=x.device)
    merge[alive] = t.as_tensor(fcluster(linkage(fine_centroids[alive].cpu().numpy(), method="average", metric="cosine"), k_coarse, criterion="maxclust") - 1, device=x.device, dtype=t.long)
    labels = merge[fine_labels]
    centroids = normed(t.zeros(k_coarse, x.shape[1], device=x.device).index_add_(0, labels, normed(x.float())))
    return labels, centroids

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p, d = k / n, 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (min(max(c - h, 0.0), p), max(min(c + h, 1.0), p))  # clamped: at k=0 or k=n rounding can put a bound a float epsilon outside [0, 1] or on the wrong side of p
