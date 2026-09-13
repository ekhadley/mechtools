import html

import torch as t
from torch import Tensor
from huggingface_hub import hf_hub_download
from IPython.display import HTML, display
from safetensors import safe_open

from mechtools.stats import kmeans
from mechtools.tables import show_table
from mechtools.tokens import toks_html

LENS_REPO = "camilablank/workspace-lenses"

# ============================= loading and scoring ============================= #

def load_jlens(path: str, device: str = "cpu") -> dict:
    """A j-lens .pt from the workspace-lenses repo: {"J": [n_layers, d_model, d_model], "provenance": ...}."""
    return t.load(hf_hub_download(repo_id=LENS_REPO, filename=path), map_location=device, weights_only=False)

def load_tlens(path: str, device: str = "cpu") -> dict:
    """A template lens safetensors from the workspace-lenses repo plus its row -> text map: {"meta", "templates": [n_layers, n_templates, d_model], "word_ids", "words"}."""
    local_path = hf_hub_download(repo_id=LENS_REPO, filename=path)
    words_path = hf_hub_download(repo_id=LENS_REPO, filename=path.replace("templates", "template_words").replace(".safetensors", ".txt"))
    with safe_open(local_path, framework="pt", device=device) as f:
        tlens = {"meta": f.metadata(), "templates": f.get_tensor("templates"), "word_ids": f.get_tensor("word_ids")}
    tlens["words"] = [line.split("\t", 1)[1] for line in open(words_path).read().splitlines()]
    return tlens

def jlens_transport(h: Tensor, layer: int, jlens: dict) -> Tensor:
    return jlens["J"][layer].to(h.device, h.dtype) @ h

def get_lens_logits(h: Tensor, layer: int, model, jlens: dict) -> Tensor:
    return model.unembed(model.ln_final(jlens_transport(h, layer, jlens)))

def get_jlens_token_vec(token: str | int, layer: int, model, jlens: dict) -> Tensor:
    """Residual direction at `layer` that the j-lens maps onto a token's unembedding."""
    tok_id = model.tokenizer.encode(token)[0] if isinstance(token, str) else token
    return jlens["J"][layer].to(model.W_U.dtype).T @ model.W_U[:, tok_id]

def get_template_idx(template: str, tlens: dict) -> int:
    return tlens["words"].index(template)

def get_template_vecs(templates: list[str], layer: int, tlens: dict) -> Tensor:
    """[n_templates, d_model] template-lens directions at `layer`."""
    return tlens["templates"][layer, [get_template_idx(templ, tlens) for templ in templates]]

def get_template_vec(template: str, layer: int, tlens: dict) -> Tensor:
    return get_template_vecs([template], layer, tlens)[0]

def print_templates(tlens: dict, contains: str | None = None):
    for i, word in enumerate(tlens["words"]):
        if contains is None or contains.lower() in word.lower():
            print(f"{i}\t{word!r}")

def get_tlens_scores(h: Tensor, layer: int, tlens: dict) -> Tensor:
    return t.cosine_similarity(tlens["templates"][layer].to(h.device, h.dtype), h, dim=-1)

def top_templates_table(scores: Tensor, words: list[str], k: int = 10, title: str | None = None):
    top = scores.flatten().float().topk(k)
    show_table(["Idx", "Template", "Cos"], [(i, repr(words[idx]), val) for i, (idx, val) in enumerate(zip(top.indices.tolist(), top.values.tolist()))], title)

# ============================= HTML readouts ============================= #

READOUT_CSS = "<style>.ro{{display:grid;grid-template-columns:repeat({n_cols},1fr);gap:8px}} .ro table{{border-collapse:collapse;align-self:start}} .ro th{{background:#2a3f5f;text-align:left;padding:3px 6px;font-weight:normal}} .ro td{{padding:1px 6px;white-space:nowrap;text-align:left}} .ro td:last-child{{text-align:right;color:#eee}}</style>"

def fmt3(x: float) -> str:
    """3 sig figs: fixed point down to 1e-4, scientific below that."""
    return f"{x:#.3g}"

def cluster_color(c: int) -> str:
    return f"hsl({c * 0.618 % 1 * 360:.0f},70%,65%)"

def token_strip(toks: list[str], pos: int = -1, ctx: int = 12) -> str:
    """The tokens up to `ctx` either side of `pos`, alternating backgrounds, the one at `pos` underlined."""
    pos = pos % len(toks)
    return f"<div style='margin:0 0 8px;color:#ddd'>{toks_html(toks, None, pos, max(0, pos - ctx), min(len(toks), pos + ctx + 1))}</div>"

def readout_html(tables: list[tuple[str, list[tuple[list[str], str | None]]]], title: str | None = None, toks: list[str] | None = None, pos: int = -1, ctx: int = 12, n_cols: int = 4) -> str:
    """Dark monospace grid of small tables. `tables` is [(header html, [(cell htmls, row color or None), ...]), ...]. Cells are raw html, so escape names first."""
    grid = "".join(f"<table><tr><th colspan={len(rows[0][0])}>{header}</th></tr>" + "".join(f"<tr{f' style=color:{color}' if color else ''}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>" for cells, color in rows) + "</table>" for header, rows in tables)
    heading = (f"<h3 style='margin:0 0 8px'>{title}</h3>" if title else "") + (token_strip(toks, pos, ctx) if toks is not None else "")
    return f"{READOUT_CSS.format(n_cols=n_cols)}<div style='background:#111;color:#eee;font:12px monospace;padding:8px'>{heading}<div class='ro'>{grid}</div></div>"

def top_readout(scores: dict[str, Tensor], names, k: int = 10, softmax: bool = True, title: str | None = None, toks: list[str] | None = None, pos: int = -1, ctx: int = 12, n_cols: int = 4):
    """One table per entry of `scores` ({header: [n] logits or scores}) listing its top-k items. softmax=True shows probs, else raw scores.
    names maps an item id to its string: a list, or a callable like tokenizer.decode. If toks is given, shows them around pos with the token at pos underlined."""
    name = names.__getitem__ if isinstance(names, list) else names
    tables = []
    for header, s in scores.items():
        vals = s.flatten().float()
        top = (vals.softmax(-1) if softmax else vals).topk(k)
        tables.append((html.escape(header), [([html.escape(repr(name(i))), fmt3(v)], None) for i, v in zip(top.indices.tolist(), top.values.tolist())]))
    display(HTML(readout_html(tables, title, toks, pos, ctx, n_cols)))

def jlens_readout(cache, layers, pos: int, model, jlens: dict, k: int = 10, hook: str = "hook_resid_pre", toks: list[str] | None = None, title: str = "j-lens readout", ctx: int = 12, n_cols: int = 4):
    """j-lens token readout at `pos` for each layer in `layers`, from a run_with_cache cache of a single prompt."""
    scores = {f"L{layer}": get_lens_logits(cache[f"blocks.{layer}.{hook}"][0, pos], layer, model, jlens) for layer in layers}
    top_readout(scores, model.tokenizer.decode, k, True, title, toks, pos, ctx, n_cols)

def tlens_readout(cache, layers, pos: int, tlens: dict, k: int = 10, hook: str = "hook_resid_pre", toks: list[str] | None = None, title: str = "template-lens readout", ctx: int = 12, n_cols: int = 4):
    """Template-lens cosine readout at `pos` for each layer in `layers`, from a run_with_cache cache of a single prompt."""
    scores = {f"L{layer}": get_tlens_scores(cache[f"blocks.{layer}.{hook}"][0, pos], layer, tlens) for layer in layers}
    top_readout(scores, tlens["words"], k, False, title, toks, pos, ctx, n_cols)

def cluster_tlens(tlens: dict, layer: int, k: int = 256, iters: int = 50, seed: int = 0, device: str = "cuda") -> tuple[Tensor, Tensor]:
    """Spherical k-means over one layer's mean-centered template vectors. Returns (labels [n_templates], centroids [k, d])."""
    x = tlens["templates"][layer].to(device).float()
    return kmeans(x - x.mean(0), k, iters, seed)

def cluster_readout(scores: Tensor, labels: Tensor, names, n_clusters: int = 11, n_rows: int = 10, n_cols: int = 4, title: str | None = None, toks: list[str] | None = None, pos: int = -1, ctx: int = 12, softmax: bool = True):
    """Grid of tables: first is the overall top-k with each item's cluster id, then one per top cluster with each item's overall rank. Items are colored by cluster.
    softmax=True: scores are logits, cells show probs, clusters ranked by prob mass. softmax=False: cells show raw scores (e.g. cosines), clusters ranked by max score.
    names maps an item id to its string: a list, or a callable like tokenizer.decode. If toks is given, shows them (up to ctx either side of pos) with the token at pos underlined. Returns the shown cluster ids."""
    name = names.__getitem__ if isinstance(names, list) else names
    vals = scores.flatten().float()
    labels = labels.to(vals.device)
    vals = vals.softmax(-1) if softmax else vals
    k = labels.max().item() + 1
    strength = t.zeros(k, device=vals.device).index_add_(0, labels, vals) if softmax else t.full((k,), -t.inf, device=vals.device).scatter_reduce_(0, labels, vals, "amax")
    rank = vals.argsort(descending=True).argsort() + 1
    top = vals.topk(n_rows)
    tables = [("top-k overall", [([html.escape(repr(name(i))), f"c{labels[i].item()}", fmt3(p)], cluster_color(labels[i].item())) for i, p in zip(top.indices.tolist(), top.values.tolist())])]
    top_strength = strength.topk(n_clusters)
    for m, c in zip(top_strength.values.tolist(), top_strength.indices.tolist()):
        idxs = (labels == c).nonzero().flatten()
        top = vals[idxs].topk(min(n_rows, len(idxs)))
        sel = idxs[top.indices]
        header = f"<span style='color:{cluster_color(c)}'>c{c}</span> &middot; {'mass' if softmax else 'max'} {m:.3f} &middot; n={len(idxs)}"
        tables.append((header, [([html.escape(repr(name(i))), f"<span style='color:#999'>#{r}</span>", fmt3(p)], cluster_color(c)) for i, r, p in zip(sel.tolist(), rank[sel].tolist(), top.values.tolist())]))
    display(HTML(readout_html(tables, title, toks, pos, ctx, n_cols)))
    return top_strength.indices.tolist()
