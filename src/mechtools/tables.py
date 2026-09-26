import html

import IPython
from IPython.display import HTML, display
from tabulate import tabulate
from torch import Tensor

from mechtools.colors import bold, endc

TABLE_CSS = "font-family:monospace;border-collapse:collapse;margin:4px 0"
CELL_CSS = "padding:1px 10px;text-align:left;border-bottom:1px solid #8884"

def html_table(headers: list[str], rows: list[tuple], title: str | None = None) -> str:
    """Floats (and 0-d tensors) to 4 significant figures, everything else escaped str."""
    def fmt(c):
        c = c.item() if isinstance(c, Tensor) and c.ndim == 0 else c
        return f"{c:.4g}" if isinstance(c, float) else html.escape(str(c))
    caption = f"<caption style='caption-side:top;font-weight:bold;text-align:left'>{html.escape(title)}</caption>" if title else ""
    head = "<tr>" + "".join(f"<th style='{CELL_CSS}'>{html.escape(h)}</th>" for h in headers) + "</tr>"
    body = "".join("<tr>" + "".join(f"<td style='{CELL_CSS}'>{fmt(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table style='{TABLE_CSS}'>{caption}{head}{body}</table>"

def print_titled_table(table_str: str, title: str | None = None):
    """Prints a tabulate table. A title goes into the top border of a rounded_outline table when it fits, else on its own line above the table."""
    if title is None:
        print(table_str)
        return
    lines = table_str.splitlines()
    inner = len(lines[0]) - 2
    if lines[0].startswith("╭") and len(title) <= inner:
        print(f"╭{'─' * inner}╮\n│{bold}{title.center(inner)}{endc}│\n├{'─' * inner}┤\n" + "\n".join(lines[1:]))
    else:
        print(f"{bold}{title}{endc}\n{table_str}")

def show_table(headers: list[str], rows: list[tuple], title: str | None = None):
    """Rich HTML table in a notebook kernel, tabulate text table otherwise."""
    if type(IPython.get_ipython()).__name__ == "ZMQInteractiveShell":  # jupyter or vscode kernel, not terminal ipython
        display(HTML(html_table(headers, rows, title)))
    else:
        print_titled_table(tabulate(rows, headers=headers, tablefmt="rounded_outline"), title)

def top_toks_table(logits: Tensor, tokenizer, k: int = 10, show_negative: bool = False, show_probs: bool = True, title: str | None = None, return_top: bool = False):
    """Table of the k most (and, with show_negative, least) likely tokens of one position's logits ([vocab], or with leading batch/position dims of size 1). return_top gives (top strs, top logits[, bottom strs, bottom logits])."""
    logits = logits.squeeze().float()
    if logits.ndim != 1:
        raise ValueError(f"top_toks_table takes the logits of one position, got shape {tuple(logits.shape)}")
    probs = logits.softmax(-1)
    sides = [logits.topk(k, largest=largest) for largest in ([True, False] if show_negative else [True])]
    strs = [[repr(tokenizer.decode([i])) for i in side.indices.tolist()] for side in sides]
    cols = [col for side, ss in zip(sides, strs) for col in [ss, side.values.tolist()] + ([probs[side.indices].tolist()] if show_probs else [])]
    headers = ["Tok", "Logit"] + (["Prob"] if show_probs else [])
    headers = [f"Top {h}" for h in headers] + [f"Bot {h}" for h in headers] if show_negative else headers
    show_table(["Idx"] + headers, [(i, *(col[i] for col in cols)) for i in range(k)], title)
    if return_top:
        return tuple(col for side, ss in zip(sides, strs) for col in (ss, side.values.tolist()))
