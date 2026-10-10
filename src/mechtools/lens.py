import html
import json
import math
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
    """Residual direction at `layer` that the j-lens maps onto a token's unembedding. A str must be exactly one token (tokenized without special tokens, so no BOS)."""
    if isinstance(token, str):
        ids = model.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{token!r} is {len(ids)} tokens ({[model.tokenizer.decode(i) for i in ids]}); pass one token or its id")
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
# A readout is a dark frame (readout_html) around a grid of small tables (readout_grid), or around a tabbed widget whose panes are such grids. A grid is data first,
# [(header, rows, color, cols), ...] (the Table of readout_grid's docstring), and html second: readout_grid renders a grid in Python, and tabbed renders the first pane that way
# and ships the others as one JSON payload that a small script renders in the browser when a pane is shown, so a widget of fifty panes holds one pane's DOM and its html is a few bytes per row.

FG = "#eee"  # the frame's foreground, set once as the --fg custom property on the frame; the rules and the cells' inherited color come from it
READOUT_RULES = [".rg{display:grid;grid-template-columns:repeat(var(--n),1fr);gap:8px}", "table{border-collapse:collapse;border-spacing:0;border:0;margin:0;table-layout:auto;align-self:start;font:inherit;color:var(--fg)}", "tr{background:none}", "th{background:#2a3f5f;border:0;padding:3px 6px;text-align:left;font-weight:normal;color:inherit}", "td{border:0;padding:1px 6px;white-space:nowrap;text-align:left;color:inherit}", "td.d{color:#999}", "td.v{text-align:right;color:var(--fg)}", "td.w{white-space:pre-wrap}", "td.w>div{max-width:100ch}", ".tb{margin:0 0 8px}", ".tb button{background:#222;color:#aaa;border:1px solid #444;padding:2px 8px;font:inherit;cursor:pointer}", ".tb button.on{background:#2a3f5f;color:#fff}", ".pane{visibility:visible}"]
# readout_css scopes every rule to one frame's id, so each has an id's specificity: it beats a host stylesheet's class rules (JupyterLab's (0,2,1) right-aligned td, its (0,2,2)
# tbody stripes and hover, its centered fixed-layout tables; VS Code's right-aligned, striped rows) and the global .ro / .pane rules that readouts saved by earlier versions carry,
# whatever order the outputs are in. The table rule resets everything a host sets on a table (border, margin, layout, font, color), so a readout looks the same in every host.
# The grid's column count is the --n variable set on each frame, so widgets with different n_cols coexist. No rule colors a bare td, th, tr or table beyond the frame's --fg:
# a row's color is set on its tr and reaches its cells by inheritance, except a 'd' column (dim, #999) and a 'v' column (the value column, the frame's color).

def readout_css(uid: str) -> str:
    """READOUT_RULES in one <style>, each scoped to the frame with id `uid`."""
    return "<style>" + "".join(f"#{uid} {rule}" for rule in READOUT_RULES) + "</style>"

def fmt3(x: float) -> str:
    """3 sig figs: fixed point down to 1e-4, scientific below that."""
    return f"{x:#.3g}"

def cluster_color(c: int) -> str:
    """The CSS color of cluster id c: hsl(floor(c * 0.618 % 1 * 360 + 0.5),70%,65%), hues on the golden-angle spiral. The widget's script uses the same expression, so a pane built in the browser matches one rendered here (JS's toFixed and Python's :.0f round an exact .5 differently)."""
    return f"hsl({math.floor(c * 0.618 % 1 * 360 + 0.5)},70%,65%)"

def css_color(c: int | str | None) -> str | None:
    """A table's row or header color as CSS: a cluster id (int) through cluster_color, a CSS color string as is, None as None. The widget's script maps the same values the same way."""
    return cluster_color(c) if isinstance(c, int) else c

def get_toks(input_src, tokenizer=None, add_special_tokens: bool = True) -> tuple[list[str] | None, list[int] | None]:
    """(str tokens, ids) of a readout's input_src: None or a list of str tokens as is (ids None); a string, ids (tensor or list), or a conversation is tokenized, which needs tokenizer. add_special_tokens applies to a string (see to_ids)."""
    if input_src is None:
        return None, None
    if isinstance(input_src, list) and not input_src:
        raise ValueError("input_src is empty")
    if isinstance(input_src, list) and isinstance(input_src[0], str):
        return input_src, None
    assert tokenizer is not None, "input_src needs tokenizing, pass tokenizer"
    ids = to_ids(input_src, tokenizer, add_special_tokens)
    return [tokenizer.decode(i) for i in ids], ids

def per_pos(fn, pos: int | list[int]):
    """fn(pos) for an int, {p: fn(p)} over a list of positions."""
    return fn(pos) if isinstance(pos, int) else {p: fn(p) for p in pos}

def token_strip(toks: list[str], ids: list[int] | None = None, pos: int | list[int] = -1, ctx: int = 32) -> str:
    """The tokens up to `ctx` either side of the position(s), hover showing index, id (if given) and repr.
    An int outlines that token in amber; a list of positions outlines each one, clickable as the position tabs of the enclosing `tabbed` widget, the first one selected until a script says otherwise."""
    ps = [p % len(toks) for p in ([pos] if isinstance(pos, int) else pos)]
    return f"<div style='margin:0 0 8px;color:#ddd'>{toks_html(toks, ids, ps[0] if isinstance(pos, int) else ps, max(0, min(ps) - ctx), min(len(toks), max(ps) + ctx + 1))}</div>"

def norm_table(table: tuple) -> tuple:
    """(header, rows, color, cols) of a table given with color and cols optional (see readout_grid). Raises for a header that is not a str or a list of segments, a row that is not a list of str cells (the (cell htmls, color) tuples of the html form readout_grid took before 2026-10 among them), and a per-row color list with other than one entry per row."""
    header, rows, color, cols = (*table, None, None)[:4]
    if not isinstance(header, str) and not (isinstance(header, (list, tuple)) and all(isinstance(s, str) or (isinstance(s, (list, tuple)) and len(s) == 2 and isinstance(s[0], str)) for s in header)):
        raise TypeError(f"a table header is a str or a list of segments, each a str or (str, color); got {header!r}")
    for cells in rows:
        if not isinstance(cells, (list, tuple)) or not all(isinstance(c, str) for c in cells):
            raise TypeError(f"a table row is a list of str cells, which readout_grid escapes (cell html and (cells, color) tuples are not taken); got {cells!r}")
    if isinstance(color, (list, tuple)) and len(color) != len(rows):
        raise ValueError(f"{len(color)} row colors for {len(rows)} rows")
    return header, rows, color, cols

def header_html(header: str | list) -> str:
    """A table header: a str, or a list of segments joined by middots, a segment being a str or (str, color) for a colored one. Text is escaped."""
    return " &middot; ".join(f'<span style="color:{html.escape(css_color(s[1]))}">{html.escape(s[0], False)}</span>' if isinstance(s, (tuple, list)) else html.escape(s, False) for s in ([header] if isinstance(header, str) else header))

def readout_grid(tables: list[tuple]) -> str:
    """Grid of small tables as html, the static form of what tabbed's script builds from the same data. `tables` is [Table, ...] where a Table is (header, rows, color, cols) with color and cols optional:
    header a str, or a list of segments joined by middots, a segment a str or (str, color); rows a list of cell lists, every cell text (escaped here, so pass names unescaped); color None, one color for every row, or a list of one per row, a color being a cluster id (an int, through cluster_color) or a CSS color string; cols a list of per-column classes, None for none: 'd' dims the column (a rank), 'v' is the value column, right-aligned in the frame's color whatever the row's, and 'w' lets a text column wrap, pre-wrap in a div at most 100ch wide (samples and answers).
    A row's color goes on its tr, quoted and escaped, and every cell but a 'd' or 'v' one inherits it; rows without a color take the frame's foreground."""
    def cell(x, cls):
        return (f"<td class={cls}>" + (f"<div>{html.escape(x, False)}</div>" if cls == "w" else html.escape(x, False)) + "</td>") if cls else f"<td>{html.escape(x, False)}</td>"
    def row(cells, color, cols):
        return f"<tr{f' style="color:{html.escape(css_color(color))}"' if color is not None else ''}>" + "".join(cell(x, cols[j] if cols and j < len(cols) else None) for j, x in enumerate(cells)) + "</tr>"
    def table(header, rows, color, cols):
        return f"<table><tr><th colspan={len(rows[0]) if rows else 1}>{header_html(header)}</th></tr>" + "".join(row(cells, c, cols) for cells, c in zip(rows, color if isinstance(color, (list, tuple)) else [color] * len(rows))) + "</table>"
    return "<div class='rg'>" + "".join(table(*norm_table(tb)) for tb in tables) + "</div>"

def readout_html(body: str, title: str | None = None, toks: list[str] | None = None, ids: list[int] | None = None, pos: int = -1, ctx: int = 32, n_cols: int = 4) -> str:
    """Dark monospace frame around `body` (a readout_grid, or a tabbed widget of grids) with an optional title and token strip above it; its grids have `n_cols` columns.
    The frame has a random id and carries the readout css scoped to it (readout_css), and its foreground is the --fg custom property set on it, the one place the color is written."""
    uid = "f" + secrets.token_hex(4)  # a letter first: an id selector cannot start with a digit
    heading = (f"<h3 style='margin:0 0 8px'>{title}</h3>" if title else "") + (token_strip(toks, ids, pos, ctx) if toks is not None else "")
    return f"{readout_css(uid)}<div id='{uid}' style='--n:{n_cols};--fg:{FG};background:#111;color:var(--fg);font:12px monospace;padding:8px'>{heading}{body}</div>"

# The widget's script. D is the payload (the first pane's entry is null: it is the prerendered `first`); `grid` builds the elements readout_grid writes, through textContent, never innerHTML,
# so the payload is text, and `pane` builds a pane each time it is shown (a swap of a built pane costs the same layout, and keeping them would grow the DOM back to every pane).
# The height: `fit` reads one header row's and one cell row's height off the shown pane and estimates every data pane's grid height from its row counts (exact for one-line cells),
# `hold` raises the pane area's min-height to the larger of that floor and the shown pane, so one-line readouts have their tallest pane's height from the start and a wrapping
# text readout grows once when a taller pane first shows; neither ever shrinks. A widget without a size yet (a collapsed cell) is measured by a ResizeObserver when it gets one.
TABBED_JS = "const D=JSON.parse(document.getElementById(r.id+'d').textContent),bars=[...r.querySelectorAll('.tb')],box=r.querySelector('.pn'),marks=[...r.querySelectorAll('[data-p]')],cur=[0,0],first=box.firstElementChild;let floor=0;const col=c=>typeof c=='number'?`hsl(${Math.floor(c*0.618%1*360+0.5)},70%,65%)`:c;const seg=(th,s)=>{if(typeof s=='string')th.append(s);else{const e=th.appendChild(document.createElement('span'));e.style.color=col(s[1]);e.textContent=s[0]}};const grid=ts=>{const g=document.createElement('div');g.className='rg';for(const[h,rows,k,cols]of ts){const tb=g.appendChild(document.createElement('table')),th=tb.insertRow().appendChild(document.createElement('th'));th.colSpan=rows.length?rows[0].length:1;(typeof h=='string'?[h]:h).forEach((s,i)=>{if(i)th.append(' \\u00b7 ');seg(th,s)});rows.forEach((cells,i)=>{const tr=tb.insertRow(),c=col(Array.isArray(k)?k[i]:k);if(c!=null)tr.style.color=c;cells.forEach((x,j)=>{const td=tr.insertCell(),cl=cols&&cols[j];if(cl)td.className=cl;(cl=='w'?td.appendChild(document.createElement('div')):td).textContent=x})})}return g};const pane=p=>{const d=document.createElement('div');d.className='pane';if(typeof p=='string')d.innerHTML=p;else d.appendChild(grid(p));return d};const hold=()=>{floor=Math.max(floor,box.offsetHeight);box.style.minHeight=floor+'px'};const fit=()=>{const tb=[...box.querySelectorAll('table')].find(t=>t.rows.length>1)||box.querySelector('table');if(!tb||!tb.offsetHeight)return false;const th=tb.rows[0].offsetHeight,td=tb.rows[1]?tb.rows[1].offsetHeight:th,n=+getComputedStyle(box).getPropertyValue('--n')||1;for(const p of D.panes.flat())if(Array.isArray(p)){let h=0;for(let i=0;i<p.length;i+=n)h+=Math.max(...p.slice(i,i+n).map(t=>th+t[1].length*td))+(i?8:0);floor=Math.max(floor,h)}hold();return true};const show=()=>{bars.forEach((bar,l)=>[...bar.children].forEach((x,j)=>x.classList.toggle('on',j==cur[l])));marks.forEach(x=>x.classList.toggle('on',x.dataset.p==cur[1]));const p=D.panes[cur[0]][cur[1]];box.replaceChildren(p==null?first:pane(p));hold()};bars.forEach((bar,l)=>[...bar.children].forEach((x,j)=>x.onclick=()=>{cur[l]=j;show()}));marks.forEach(x=>x.onclick=()=>{cur[1]=+x.dataset.p;show()});const K={ArrowLeft:[0,-1],ArrowRight:[0,1],ArrowUp:[1,-1],ArrowDown:[1,1]};r.onkeydown=e=>{const m=K[e.key];if(m){const n=bars[m[0]].children.length;cur[m[0]]=(cur[m[0]]+m[1]+n)%n;show();e.preventDefault();e.stopPropagation()}};show();fit()||(window.ResizeObserver&&new ResizeObserver((e,o)=>fit()&&o.disconnect()).observe(box))"

def tabbed(panes: dict[str, list | str] | dict[str, dict[str, list | str]], head: str = "", bars: bool = True) -> str:
    """A bar of tabs over the panes, one shown at a time, or two bars when the panes are nested dicts (every outer key having the same inner keys). A pane is a list of tables (see readout_grid) or raw html. A bar with a single tab is not shown, and bars=False hides both bars, leaving the tabs in `head` and the arrow keys as the only way to switch.
    Click a tab, or click anywhere in the widget and use the left/right (first bar) and up/down (second bar) arrow keys. `head` is html above the bars; any element in it with a data-p attribute is a clickable tab of the second bar and carries class 'on' while selected (the first one from the start).
    The first pane is rendered into the html by readout_grid; the others travel as one JSON payload, a <script type=application/json> holding {"tabs", "inner", "panes": [[pane per inner tab] per tab]} with every table in its 4-tuple form (the first pane's entry null), and the widget's script builds a pane's DOM each time it is shown, so the document holds one pane's tables however many tabs there are. A widget of one pane is that pane and its inert bars, with no payload or script. Where scripts do not run (an untrusted notebook, a GitHub preview) the widget shows its first pane under inert bars with the first tab of each bar marked.
    The pane area holds the height of the tallest pane: the script estimates every pane's from its row counts and one measured row, which is exact for one-line cells (the cluster and logits readouts), and a readout with wrapping text cells grows once when a taller pane first shows; the widget never shrinks on a switch, so the page does not jump. Tab labels are escaped; raw html panes are not."""
    if not panes:
        raise ValueError("no panes")
    uid = "t" + secrets.token_hex(4)
    rows = {k: v if isinstance(v, dict) else {"": v} for k, v in panes.items()}
    inner = list(next(iter(rows.values())))
    if any(list(r) != inner for r in rows.values()):
        raise ValueError("every tab needs the same inner keys in the same order")
    data = {"tabs": [str(k) for k in rows], "inner": [str(k) for k in inner], "panes": [[p if isinstance(p, str) else [norm_table(tb) for tb in p] for p in r.values()] for r in rows.values()]}
    first, data["panes"][0][0] = data["panes"][0][0], None
    buttons = lambda ls: "".join(f"<button{' class=on' if j == 0 else ''}>{html.escape(str(l))}</button>" for j, l in enumerate(ls))
    tab_bars = "".join(f"<div class='tb'{' hidden' if len(ls) == 1 or not bars else ''}>{buttons(ls)}</div>" for ls in [list(rows), inner])
    body = f"<div id='{uid}' tabindex=0 style='outline:none'>{head}{tab_bars}<div class='pn'><div class='pane'>{first if isinstance(first, str) else readout_grid(first)}</div></div>"
    if len(rows) * len(inner) == 1:
        return body + "</div>"
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")  # no "<" in script data, so no header or cell can close the script
    return f"{body}<script type=application/json id='{uid}d'>{payload}</script></div><script>(()=>{{const r=document.getElementById('{uid}');{TABBED_JS}}})()</script>"

def top_readout(scores: dict[str, Tensor], names, k: int = 10, softmax: bool = True, title: str | None = None, input_src=None, pos: int = -1, ctx: int = 32, n_cols: int = 4, tokenizer=None):
    """One table per entry of `scores` ({header: [n] logits or scores}) listing its top-k items (k clipped to n). softmax=True shows probs, else raw scores.
    names maps an item id to its string: a list, or a callable like tokenizer.decode. input_src (see get_toks) shows the tokens up to ctx either side of pos, the one at pos outlined in amber; a string is tokenized with special tokens added, so pass a self-rendered template string as ids."""
    name = names.__getitem__ if isinstance(names, list) else names
    tables = []
    for header, s in scores.items():
        vals = s.flatten().float()
        top = (vals.softmax(-1) if softmax else vals).topk(min(k, len(vals)))
        tables.append((header, [[repr(name(i)), fmt3(v)] for i, v in zip(top.indices.tolist(), top.values.tolist())], None, [None, "v"]))
    display(HTML(readout_html(readout_grid(tables), title, *get_toks(input_src, tokenizer), pos, ctx, n_cols)))

NEXT_COLOR = "#e88"

def show_logits(input_src, model=None, logits=None, tokenizer=None, k: int = 10, pos: list[int] | None = None, title: str | None = "logits", ctx: int = 32, n_cols: int = 4, add_special_tokens: bool = True):
    """Top-k next-token table for one position of the input at a time: click a token in the strip above (or use the up/down arrows) to see what the model predicts after it.
    Pass `model` to run it on the input, or `logits` [seq, vocab] (or [1, seq, vocab]) from your own forward pass. The row of the input's actual next token is colored, and appended below the top-k when it is not in it.
    input_src (see get_toks) is a string (add_special_tokens=False for a self-rendered template string), ids, a conversation, or str tokens (which cannot be run through a model: pass logits with them); `pos` restricts the readout to those positions, all of them by default."""
    assert (model is None) != (logits is None), "pass either model or logits"
    tokenizer = tokenizer if tokenizer is not None else model.tokenizer
    toks, ids = get_toks(input_src, tokenizer, add_special_tokens)
    if toks is None:
        raise ValueError("show_logits needs input_src")
    if logits is None:
        if ids is None:
            raise ValueError("input_src is str tokens, which cannot be run through the model: pass a string, ids or a conversation, or pass logits")
        logits = model(t.tensor(ids)[None])
    logits = logits.squeeze(0) if logits.ndim == 3 and logits.shape[0] == 1 else logits
    if logits.ndim != 2 or logits.shape[0] != len(toks):
        raise ValueError(f"logits should be [seq, vocab] for the {len(toks)} input tokens, got {tuple(logits.shape)}")
    positions = list(range(len(toks))) if pos is None else [p % len(toks) for p in pos]
    panes = {}
    for p in positions:
        probs = logits[p].float().softmax(-1)
        top = probs.topk(min(k, len(probs)))
        nxt = ids[p + 1] if ids is not None and p + 1 < len(ids) else None
        row = lambda rank, i, prob: ([f"#{rank}", repr(tokenizer.decode(i)), fmt3(prob)], NEXT_COLOR if i == nxt else None)
        rows = [row(rank, i, prob) for rank, (i, prob) in enumerate(zip(top.indices.tolist(), top.values.tolist()), 1)]
        if nxt is not None and nxt not in top.indices:
            rows.append(row((probs > probs[nxt]).sum().item() + 1, nxt, probs[nxt].item()))
        panes[f"p{p}"] = [([f"p{p}", f"{toks[p]!r} →"], [cells for cells, _ in rows], [c for _, c in rows] if nxt is not None else None, ["d", None, "v"])]
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

def cluster_tlens(tlens: dict, layer: int, k: int = 256, iters: int = 50, seed: int = 0, device: str | None = None) -> tuple[Tensor, Tensor]:
    """Spherical k-means over one layer's mean-centered template vectors, on `device` (cuda when available, else the templates' own device). Returns (labels [n_templates], centroids [k, d])."""
    device = device or ("cuda" if t.cuda.is_available() else tlens["templates"].device)
    x = tlens["templates"][layer].to(device).float()
    return kmeans(x - x.mean(0), k, iters, seed)

def cluster_tables(scores: Tensor, labels: Tensor, name, n_clusters: int, n_rows: int, softmax: bool) -> tuple[list, list[int]]:
    """Tables (readout_grid's form) for one cluster readout: the overall top-k with each item's cluster id, each row colored by its cluster, then one per top cluster (of the non-empty ones, at most n_clusters) with each item's overall rank in a dim column, its rows colored by the cluster and its header naming the cluster in that color; the value column stays in the frame's color. n_rows is clipped to the items available. Returns (tables, shown cluster ids)."""
    vals = scores.flatten().float()
    labels = labels.to(vals.device).flatten()
    if len(labels) != len(vals):
        raise ValueError(f"{len(vals)} scores but {len(labels)} labels")
    vals = vals.softmax(-1) if softmax else vals
    k = labels.max().item() + 1
    counts = t.bincount(labels, minlength=k)
    strength = t.zeros(k, device=vals.device).index_add_(0, labels, vals) if softmax else t.full((k,), -t.inf, device=vals.device).scatter_reduce_(0, labels, vals, "amax")
    strength[counts == 0] = -t.inf
    rank = vals.argsort(descending=True).argsort() + 1
    top = vals.topk(min(n_rows, len(vals)))
    top_labels = labels[top.indices].tolist()
    tables = [("top-k overall", [[repr(name(i)), f"c{c}", fmt3(p)] for i, c, p in zip(top.indices.tolist(), top_labels, top.values.tolist())], top_labels, [None, None, "v"])]
    top_strength = strength.topk(min(n_clusters, int((counts > 0).sum())))
    for m, c in zip(top_strength.values.tolist(), top_strength.indices.tolist()):
        idxs = (labels == c).nonzero().flatten()
        top = vals[idxs].topk(min(n_rows, len(idxs)))
        sel = idxs[top.indices]
        header = [(f"c{c}", c), f"{'mass' if softmax else 'max'} {m:.3f}", f"n={len(idxs)}"]
        tables.append((header, [[repr(name(i)), f"#{r}", fmt3(p)] for i, r, p in zip(sel.tolist(), rank[sel].tolist(), top.values.tolist())], c, [None, "d", "v"]))
    return tables, top_strength.indices.tolist()

def cluster_readout(scores: dict[str, Tensor] | dict[str, dict[int, Tensor]], labels: Tensor | dict[str, Tensor], names, n_clusters: int = 11, n_rows: int = 10, n_cols: int = 4, title: str | None = None, input_src=None, pos: int = -1, ctx: int = 32, softmax: bool = True, tokenizer=None) -> dict:
    """Tabbed grids, one per entry of `scores`: {tab: [n] logits or scores}, all at sequence position `pos`, or {tab: {pos: scores}} for a second bar of position tabs below the first (left/right arrows switch tab, up/down switch position).
    Each grid: the overall top-k with each item's cluster id, then one table per top cluster with each item's overall rank. Rows are colored by cluster (their name and cluster or rank cells; the value column stays in the frame's color).
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
            panes[label][f"p{p}"], shown[label][p] = cluster_tables(sp, lab, name, n_clusters, n_rows, softmax)
    strip = token_strip(toks, ids, positions, ctx) if toks is not None else ""
    display(HTML(readout_html(tabbed(panes, strip), title, n_cols=n_cols)))
    return shown if nested else {label: v[pos] for label, v in shown.items()}

def jlens_cluster_readout(cache, layers, pos: int | list[int], model, jlens: dict, labels: Tensor, n_clusters: int = 11, n_rows: int = 10, hook: str = "hook_resid_pre", input_src=None, title: str = "j-lens cluster readout", ctx: int = 32, n_cols: int = 4) -> dict:
    """Tabbed j-lens cluster readout, one tab per layer and, if pos is a list, a second bar of tabs per position. labels is a clustering of the vocab, e.g. cluster_vocab(model)[0]."""
    scores = {f"L{layer}": per_pos(lambda p: get_lens_logits(cache[f"blocks.{layer}.{hook}"][0, p], layer, model, jlens), pos) for layer in layers}
    return cluster_readout(scores, labels, model.tokenizer.decode, n_clusters, n_rows, n_cols, title, input_src, pos, ctx, True, model.tokenizer)

def tlens_cluster_readout(cache, layers, pos: int | list[int], tlens: dict, k: int = 256, seed: int = 0, n_clusters: int = 11, n_rows: int = 10, hook: str = "hook_resid_pre", input_src=None, title: str = "template-lens cluster readout", ctx: int = 32, n_cols: int = 4, tokenizer=None) -> dict:
    """Tabbed template-lens cluster readout, one tab per layer and, if pos is a list, a second bar of tabs per position. Each layer's templates are clustered by cluster_tlens(tlens, layer, k, seed=seed) on every call, so k is the number of clusters, not rows (n_rows is the rows per table)."""
    scores = {f"L{layer}": per_pos(lambda p: get_tlens_scores(cache[f"blocks.{layer}.{hook}"][0, p], layer, tlens), pos) for layer in layers}
    labels = {f"L{layer}": cluster_tlens(tlens, layer, k, seed=seed)[0] for layer in layers}
    return cluster_readout(scores, labels, tlens["words"], n_clusters, n_rows, n_cols, title, input_src, pos, ctx, False, tokenizer)
