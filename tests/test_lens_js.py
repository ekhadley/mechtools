"""The widget's script builds the pane the Python renderer writes: for every pane of a tabbed readout, the DOM the script builds in headless Chromium is compared with the browser's parse of readout_grid's html for the same tables (tags, classes, text, every color as the CSSOM serializes it), and the pane area's min-height is checked against the panes' heights.
Needs node with playwright on NODE_PATH (npm root -g is tried) and a Chromium it can launch; skipped otherwise, which `pytest -rs` reports."""
import json
import os
import shutil
import subprocess

import pytest

from mechtools.lens import NEXT_COLOR, readout_grid, readout_html, tabbed, token_strip

PANES = {
    "L0": {"p1": [("top-k overall", [["'a'", "c3", "0.500"], ["'b<'", "c7", "0.250"]], [3, 7], [None, None, "v"]), ([("c3", 3), "mass 0.5", "n=2"], [["'a'", "#1", "0.500"], ["'x'", "#9", "0.001"]], 3, [None, "d", "v"])],
           "p2": [("top-k overall", [["'q'", "c1", "0.900"]], [1], [None, None, "v"]), ([("c1", 1), "mass 0.9", "n=1"], [["'q'", "#1", "0.900"]], 1, [None, "d", "v"]), ([("c2", 2), "mass 0.1", "n=4"], [["'r'", "#2", "0.050"], ["'s'", "#3", "0.030"], ["'t\"u'", "#4", "0.020"]], 2, [None, "d", "v"]), ("empty", [])]},
    "L1": {"p1": [("p1", [["#1", "'y'", "0.700"], ["#2", "'z'", "0.200"], ["#5", "'w'", "0.010"]], [None, NEXT_COLOR, None], ["d", None, "v"])],
           "p2": [("sample 0", [["one bullet"], ["a second, with < & \" and a\nline break"]], None, ["w"]), ("sample 1", [["only"]], 7, ["w"])]},
}  # cluster, logits and text shapes: per-row and per-table colors, cluster ids and a CSS color, header segments, d/v/w columns, an empty table, escaping and a newline

def test_script_builds_what_python_renders(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node")
    env = dict(os.environ)
    root = subprocess.run([shutil.which("npm") or "npm", "root", "-g"], capture_output=True, text=True).stdout.strip() if shutil.which("npm") else ""
    if root:
        env["NODE_PATH"] = root + (os.pathsep + env["NODE_PATH"] if env.get("NODE_PATH") else "")
    rendered = {f"{i},{j}": readout_grid(p) for i, row in enumerate(PANES.values()) for j, p in enumerate(row.values())}
    (tmp_path / "w.html").write_text(readout_html(tabbed(PANES, token_strip([f"t{i}" for i in range(12)], pos=[1, 2])), "T", n_cols=2))
    (tmp_path / "p.json").write_text(json.dumps(rendered))
    r = subprocess.run([node, os.path.join(os.path.dirname(__file__), "parity.mjs"), str(tmp_path / "w.html"), str(tmp_path / "p.json")], capture_output=True, text=True, env=env, timeout=180)
    if r.returncode == 3:
        pytest.skip(f"no browser: {r.stderr.strip()}")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    initial = out.pop("initial")
    assert set(out) == set(rendered)
    for k, v in out.items():
        assert v["built"] == v["rendered"], k  # the script's DOM for the pane is the browser's parse of the Python html
        assert all(row["color"] for row in v["built"][1]["rows"]) if k == "0,0" else True
    heights = {k: v["pane"] for k, v in out.items()}
    one_line = [heights[k] for k in heights if k != "1,2"]  # the text pane has a cell that wraps, which the row-count estimate cannot see
    assert max(one_line) <= float(initial.removesuffix("px")) <= max(heights.values()), (initial, heights)  # the floor set before any switch holds every one-line pane
    floors = [float(v["minHeight"].removesuffix("px")) for v in out.values()]
    assert floors == sorted(floors) and floors[-1] >= max(heights.values())  # never shrinks; grows to the wrapping pane once shown
