import html
import secrets

import torch as t
from torch import Tensor
from huggingface_hub import hf_hub_download
from IPython.display import HTML, display
from safetensors import safe_open

from mechtools.stats import kmeans
from mechtools.tables import show_table
from mechtools.tokens import to_ids, toks_html

LENS_REPO = "camilablank/workspace-lenses"

# ============================= loading and scoring ============================= #

def load_jlens(path: str, device: str | t.device = "cpu") -> dict:
    """A j-lens .pt from the workspace-lenses repo: {"J": [n_layers, d_model, d_model], "provenance": ...}."""
    return t.load(hf_hub_download(repo_id=LENS_REPO, filename=path), map_location=device, weights_only=False)

def load_tlens(path: str, device: str | t.device = "cpu") -> dict:
    """A template lens safetensors from the workspace-lenses repo plus its row -> text map: {"meta", "templates": [n_layers, n_templates, d_model], "word_ids", "words"}."""
    local_path = hf_hub_download(repo_id=LENS_REPO, filename=path)
    words_path = hf_hub_download(repo_id=LENS_REPO, filename=path.replace("templates", "template_words").replace(".safetensors", ".txt"))
    with safe_open(local_path, framework="pt", device=str(device)) as f:
        tlens = {"meta": f.metadata(), "templates": f.get_tensor("templates"), "word_ids": f.get_tensor("word_ids")}
    tlens["words"] = [line.split("\t", 1)[1] for line in open(words_path).read().splitlines()]
    return tlens

def jlens_transport(h: Tensor, layer: int, jlens: dict) -> Tensor:
    return jlens["J"][layer].to(h.device, h.dtype) @ h

def get_lens_logits(h: Tensor, layer: int, model, jlens: dict) -> Tensor:
    return model.unembed(model.ln_final(jlens_transport(h, layer, jlens)))

def get_jlens_token_vec(token: str | int, layer: int, model, jlens: dict) -> Tensor:
    """Residual direction at `layer` that the j-lens maps onto a token's unembedding. A str must be exactly one token (encoded without BOS); pass the id otherwise."""
    if isinstance(token, str):
        ids = model.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{token!r} is {len(ids)} tokens ({ids}); pass one token's string or its id")
        token = ids[0]
    return jlens["J"][layer].to(model.W_U.dtype).T @ model.W_U[:, token]

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

READOUT_CSS = "<style>.ro{{display:grid;grid-template-columns:repeat({n_cols},1fr);gap:8px}} .ro table{{border-collapse:collapse;align-self:start}} .ro th{{background:#2a3f5f;text-align:left;padding:3px 6px;font-weight:normal}} .ro td{{padding:1px 6px;white-space:nowrap;text-align:left}} .ro td:last-child{{text-align:right;color:#eee}} .tb{{margin:0 0 8px}} .tb button{{background:#222;color:#aaa;border:1px solid #444;padding:2px 8px;font:inherit;cursor:pointer}} .tb button.on{{background:#2a3f5f;color:#fff}} .pn{{display:grid}} .pane{{grid-area:1/1;visibility:hidden}} .pane.on{{visibility:visible}}</style>"

def fmt3(x: float) -> str:
    """3 sig figs: fixed point down to 1e-4, scientific below that."""
    return f"{x:#.3g}"

def cluster_color(c: int) -> str:
    return f"hsl({c * 0.618 % 1 * 360:.0f},70%,65%)"

def get_toks(input_src, tokenizer=None, add_special_tokens: bool = True) -> tuple[list[str] | None, list[int] | None]:
    """(str tokens, ids) of a readout's input_src: None or a list of str tokens as is (ids None); a string, ids (tensor or list), or a conversation is tokenized, which needs tokenizer. add_special_tokens applies to a string (see to_ids)."""
    if input_src is None or (isinstance(input_src, list) and isinstance(input_src[0], str)):
        return input_src, None
    assert tokenizer is not None, "input_src needs tokenizing, pass tokenizer"
    ids = to_ids(input_src, tokenizer, add_special_tokens)
    return [tokenizer.decode(i) for i in ids], ids

def per_pos(fn, pos: int | list[int]):
    """fn(pos) for an int, {p: fn(p)} over a list of positions."""
    return fn(pos) if isinstance(pos, int) else {p: fn(p) for p in pos}

def token_strip(toks: list[str], ids: list[int] | None = None, pos: int | list[int] = -1, ctx: int = 32) -> str:
    """The tokens up to `ctx` either side of the position(s), alternating backgrounds, hover showing index, id (if given) and repr.
    An int underlines that token; a list of positions marks each one, clickable as the position tabs of the enclosing `tabbed` widget."""
    ps = [p % len(toks) for p in ([pos] if isinstance(pos, int) else pos)]
    return f"<div style='margin:0 0 8px;color:#ddd'>{toks_html(toks, ids, ps[0] if isinstance(pos, int) else ps, max(0, min(ps) - ctx), min(len(toks), max(ps) + ctx + 1))}</div>"

def readout_grid(tables: list[tuple[str, list[tuple[list[str], str | None]]]]) -> str:
    """Grid of small tables. `tables` is [(header html, [(cell htmls, row color or None), ...]), ...]. Cells are raw html, so escape names first."""
    return "<div class='ro'>" + "".join(f"<table><tr><th colspan={len(rows[0][0])}>{header}</th></tr>" + "".join(f"<tr{f' style=color:{color}' if color else ''}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>" for cells, color in rows) + "</table>" for header, rows in tables) + "</div>"

def readout_html(body: str, title: str | None = None, toks: list[str] | None = None, ids: list[int] | None = None, pos: int = -1, ctx: int = 32, n_cols: int = 4) -> str:
    """Dark monospace frame around `body` (a readout_grid, or tabbed grids) with an optional title and token strip above it."""
    heading = (f"<h3 style='margin:0 0 8px'>{title}</h3>" if title else "") + (token_strip(toks, ids, pos, ctx) if toks is not None else "")
    return f"{READOUT_CSS.format(n_cols=n_cols)}<div style='background:#111;color:#eee;font:12px monospace;padding:8px'>{heading}{body}</div>"

def tabbed(panes: dict[str, str] | dict[str, dict[str, str]], head: str = "", bars: bool = True) -> str:
    """A bar of tabs over the panes, one shown at a time, or two bars when the panes are nested dicts (every outer key having the same inner keys). A bar with a single tab is not shown, and bars=False hides both bars, leaving the tabs in `head` and the arrow keys as the only way to switch.
    Click a tab, or click anywhere in the widget and use the left/right (first bar) and up/down (second bar) arrow keys. `head` is html above the bars; any element in it with a data-p attribute is a clickable tab of the second bar and carries class 'on' while selected.
    The panes are stacked in one grid cell, so the widget keeps the height of its tallest pane and does not resize as tabs change."""
    uid = secrets.token_hex(4)
    rows = {k: v if isinstance(v, dict) else {"": v} for k, v in panes.items()}
    tab_bars = "".join(f"<div class='tb'{' hidden' if len(ls) == 1 or not bars else ''}>{''.join(f'<button>{l}</button>' for l in ls)}</div>" for ls in [list(rows), list(next(iter(rows.values())))])
    divs = "<div class='pn'>" + "".join(f"<div class='pane' data-k='{i},{j}'>{p}</div>" for i, row in enumerate(rows.values()) for j, p in enumerate(row.values())) + "</div>"
    js = f"const r=document.getElementById('{uid}'),bars=[...r.querySelectorAll('.tb')],panes=[...r.querySelectorAll('.pane')],marks=[...r.querySelectorAll('[data-p]')],cur=[0,0];const show=()=>{{bars.forEach((bar,l)=>[...bar.children].forEach((x,j)=>x.classList.toggle('on',j==cur[l])));panes.forEach(x=>x.classList.toggle('on',x.dataset.k==cur.join(',')));marks.forEach(x=>x.classList.toggle('on',x.dataset.p==cur[1]))}};bars.forEach((bar,l)=>[...bar.children].forEach((x,j)=>x.onclick=()=>{{cur[l]=j;show()}}));marks.forEach(x=>x.onclick=()=>{{cur[1]=+x.dataset.p;show()}});const K={{ArrowLeft:[0,-1],ArrowRight:[0,1],ArrowUp:[1,-1],ArrowDown:[1,1]}};r.onkeydown=e=>{{const m=K[e.key];if(m){{const n=bars[m[0]].children.length;cur[m[0]]=(cur[m[0]]+m[1]+n)%n;show();e.preventDefault();e.stopPropagation()}}}};show()"
    return f"<div id='{uid}' tabindex=0 style='outline:none'>{head}{tab_bars}{divs}</div><script>(()=>{{{js}}})()</script>"

def top_readout(scores: dict[str, Tensor], names, k: int = 10, softmax: bool = True, title: str | None = None, input_src=None, pos: int = -1, ctx: int = 32, n_cols: int = 4, tokenizer=None):
    """One table per entry of `scores` ({header: [n] logits or scores}) listing its top-k items. softmax=True shows probs, else raw scores.
    names maps an item id to its string: a list, or a callable like tokenizer.decode. input_src (see get_toks) shows the tokens up to ctx either side of pos, the one at pos underlined."""
    name = names.__getitem__ if isinstance(names, list) else names
    tables = []
    for header, s in scores.items():
        vals = s.flatten().float()
        top = (vals.softmax(-1) if softmax else vals).topk(k)
        tables.append((html.escape(header), [([html.escape(repr(name(i))), fmt3(v)], None) for i, v in zip(top.indices.tolist(), top.values.tolist())]))
    display(HTML(readout_html(readout_grid(tables), title, *get_toks(input_src, tokenizer), pos, ctx, n_cols)))

NEXT_COLOR = "#e88"

def show_logits(input_src, model=None, logits=None, tokenizer=None, k: int = 10, pos: list[int] | None = None, title: str | None = "logits", ctx: int = 32, n_cols: int = 4, add_special_tokens: bool = True):
    """Top-k next-token table for one position of the input at a time: click a token in the strip above (or use the up/down arrows) to see what the model predicts after it.
    Pass `model` to run it on the input, or `logits` [seq, vocab] (or [1, seq, vocab]) from your own forward pass. The row of the input's actual next token is colored, and appended below the top-k when it is not in it.
    input_src (see get_toks) is a string (add_special_tokens=False for a self-rendered template string), ids, a conversation, or str tokens; `pos` restricts the readout to those positions, all of them by default."""
    assert (model is None) != (logits is None), "pass either model or logits"
    tokenizer = tokenizer if tokenizer is not None else model.tokenizer
    toks, ids = get_toks(input_src, tokenizer, add_special_tokens)
    if logits is None:
        logits = model(t.tensor(ids)[None])
    logits = logits.squeeze(0) if logits.ndim == 3 else logits
    positions = list(range(len(toks))) if pos is None else [p % len(toks) for p in pos]
    panes = {}
    for p in positions:
        probs = logits[p].float().softmax(-1)
        top = probs.topk(k)
        nxt = ids[p + 1] if ids is not None and p + 1 < len(ids) else None
        row = lambda rank, i, prob: ([f"<span style='color:#999'>#{rank}</span>", html.escape(repr(tokenizer.decode(i))), fmt3(prob)], NEXT_COLOR if i == nxt else None)
        rows = [row(rank, i, prob) for rank, (i, prob) in enumerate(zip(top.indices.tolist(), top.values.tolist()), 1)]
        if nxt is not None and nxt not in top.indices:
            rows.append(row((probs > probs[nxt]).sum().item() + 1, nxt, probs[nxt].item()))
        panes[f"p{p}"] = readout_grid([(f"p{p} &middot; {html.escape(repr(toks[p]))} &rarr;", rows)])
    body = tabbed({"": panes}, token_strip(toks, ids, positions, ctx), bars=False)
    display(HTML(readout_html(body, title, n_cols=n_cols)))

def jlens_readout(cache, layers, pos: int, model, jlens: dict, k: int = 10, hook: str = "hook_resid_pre", input_src=None, title: str = "j-lens readout", ctx: int = 32, n_cols: int = 4):
    """j-lens token readout at `pos` for each layer in `layers`, from a run_with_cache cache of a single prompt."""
    scores = {f"L{layer}": get_lens_logits(cache[f"blocks.{layer}.{hook}"][0, pos], layer, model, jlens) for layer in layers}
    top_readout(scores, model.tokenizer.decode, k, True, title, input_src, pos, ctx, n_cols, model.tokenizer)

def tlens_readout(cache, layers, pos: int, tlens: dict, k: int = 10, hook: str = "hook_resid_pre", input_src=None, title: str = "template-lens readout", ctx: int = 32, n_cols: int = 4, tokenizer=None):
    """Template-lens cosine readout at `pos` for each layer in `layers`, from a run_with_cache cache of a single prompt."""
    scores = {f"L{layer}": get_tlens_scores(cache[f"blocks.{layer}.{hook}"][0, pos], layer, tlens) for layer in layers}
    top_readout(scores, tlens["words"], k, False, title, input_src, pos, ctx, n_cols, tokenizer)

def vocab_vecs(model, embed: bool = False) -> Tensor:
    """Mean-centered float token vectors [vocab, d]: the rows of W_U.T, or of W_E when embed=True."""
    x = (model.W_E if embed else model.W_U.T).float()
    return x - x.mean(0)

def cluster_vocab(model, k: int = 1024, iters: int = 150, seed: int = 0, embed: bool = False) -> tuple[Tensor, Tensor]:
    """Spherical k-means over vocab_vecs(model, embed). Returns (labels [vocab], centroids [k, d]), for jlens_cluster_readout and plot_vocab_umap."""
    return kmeans(vocab_vecs(model, embed), k, iters, seed)

def cluster_tlens(tlens: dict, layer: int, k: int = 256, iters: int = 50, seed: int = 0, device: str = "cuda") -> tuple[Tensor, Tensor]:
    """Spherical k-means over one layer's mean-centered template vectors. Returns (labels [n_templates], centroids [k, d])."""
    x = tlens["templates"][layer].to(device).float()
    return kmeans(x - x.mean(0), k, iters, seed)

def cluster_tables(scores: Tensor, labels: Tensor, name, n_clusters: int, n_rows: int, softmax: bool) -> tuple[list, list[int]]:
    """Tables for one cluster readout: the overall top-k with each item's cluster id, then one per top cluster with each item's overall rank. Returns (tables, shown cluster ids)."""
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
    return tables, top_strength.indices.tolist()

def cluster_readout(scores: dict[str, Tensor] | dict[str, dict[int, Tensor]], labels: Tensor | dict[str, Tensor], names, n_clusters: int = 11, n_rows: int = 10, n_cols: int = 4, title: str | None = None, input_src=None, pos: int = -1, ctx: int = 32, softmax: bool = True, tokenizer=None) -> dict:
    """Tabbed grids, one per entry of `scores`: {tab: [n] logits or scores}, all at sequence position `pos`, or {tab: {pos: scores}} for a second bar of position tabs below the first (left/right arrows switch tab, up/down switch position).
    Each grid: the overall top-k with each item's cluster id, then one table per top cluster with each item's overall rank. Items are colored by cluster.
    labels is [n] cluster ids shared by all tabs, or {tab: labels} when the clustering differs per tab. softmax=True: scores are logits, cells show probs, clusters ranked by prob mass. softmax=False: cells show raw scores (e.g. cosines), clusters ranked by max score.
    names maps an item id to its string: a list, or a callable like tokenizer.decode. input_src (see get_toks) shows a token strip above the bars, spanning ctx tokens either side of the positions read out; those tokens are marked and clicking one switches to its position. Returns the shown cluster ids, keyed like scores."""
    name = names.__getitem__ if isinstance(names, list) else names
    toks, ids = get_toks(input_src, tokenizer)
    nested = isinstance(next(iter(scores.values())), dict)
    positions = list(next(iter(scores.values()))) if nested else [pos]
    panes, shown = {}, {}
    for label, s in scores.items():
        lab = labels[label] if isinstance(labels, dict) else labels
        panes[label], shown[label] = {}, {}
        for p, sp in (s if nested else {pos: s}).items():
            tables, shown[label][p] = cluster_tables(sp, lab, name, n_clusters, n_rows, softmax)
            panes[label][f"p{p}"] = readout_grid(tables)
    strip = token_strip(toks, ids, positions, ctx) if toks is not None else ""
    display(HTML(readout_html(tabbed(panes, strip), title, n_cols=n_cols)))
    return shown if nested else {label: v[pos] for label, v in shown.items()}

def jlens_cluster_readout(cache, layers, pos: int | list[int], model, jlens: dict, labels: Tensor, n_clusters: int = 11, n_rows: int = 10, hook: str = "hook_resid_pre", input_src=None, title: str = "j-lens cluster readout", ctx: int = 32, n_cols: int = 4) -> dict:
    """Tabbed j-lens cluster readout, one tab per layer and, if pos is a list, a second bar of tabs per position. labels is a clustering of the vocab, e.g. kmeans over the mean-centered model.W_U.T."""
    scores = {f"L{layer}": per_pos(lambda p: get_lens_logits(cache[f"blocks.{layer}.{hook}"][0, p], layer, model, jlens), pos) for layer in layers}
    return cluster_readout(scores, labels, model.tokenizer.decode, n_clusters, n_rows, n_cols, title, input_src, pos, ctx, True, model.tokenizer)

def tlens_cluster_readout(cache, layers, pos: int | list[int], tlens: dict, k: int = 256, seed: int = 0, n_clusters: int = 11, n_rows: int = 10, hook: str = "hook_resid_pre", input_src=None, title: str = "template-lens cluster readout", ctx: int = 32, n_cols: int = 4, tokenizer=None) -> dict:
    """Tabbed template-lens cluster readout, one tab per layer and, if pos is a list, a second bar of tabs per position. Each layer's templates are clustered by cluster_tlens(tlens, layer, k, seed=seed)."""
    scores = {f"L{layer}": per_pos(lambda p: get_tlens_scores(cache[f"blocks.{layer}.{hook}"][0, p], layer, tlens), pos) for layer in layers}
    labels = {f"L{layer}": cluster_tlens(tlens, layer, k, seed=seed)[0] for layer in layers}
    return cluster_readout(scores, labels, tlens["words"], n_clusters, n_rows, n_cols, title, input_src, pos, ctx, False, tokenizer)
