import inspect

import numpy as np
import pytest
import torch as t

from mechtools import plots
from mechtools.plots import *

def test_to_numpy():
    assert to_numpy(np.arange(3)).tolist() == [0, 1, 2]
    bf = to_numpy(t.tensor([1.0, 2.0], dtype=t.bfloat16))
    assert bf.dtype == np.float32 and bf.tolist() == [1.0, 2.0]
    assert to_numpy(t.tensor([1.0], requires_grad=True)).tolist() == [1.0]
    assert to_numpy([t.tensor([1.0], dtype=t.bfloat16), t.tensor([2.0], dtype=t.bfloat16)]).shape == (2, 1)  # a list of bf16 tensors
    assert to_numpy([np.float32(1.0), np.int64(2)]).tolist() == [1.0, 2.0]
    assert to_numpy([[t.tensor(1.0), t.tensor(2.0)], [t.tensor(3.0), t.tensor(4.0)]]).shape == (2, 2)
    assert to_numpy(np.float32(2.5)) == 2.5 and to_numpy(3).shape == () and to_numpy("s") == "s"
    with pytest.raises(ValueError):
        to_numpy({"a": 1})

def test_to_series():
    assert to_series([1, 2, 3]).tolist() == [1, 2, 3]
    assert to_series([np.float32(1.0), np.float32(2.0)]).tolist() == [1.0, 2.0]
    assert to_series([np.int64(1), np.int64(2)]).tolist() == [1, 2]
    assert to_series([t.tensor(1.0), t.tensor(2.0)]).tolist() == [1.0, 2.0]  # a list of 0-d tensors is one series
    multi = to_series([t.arange(3), t.arange(3) * 2])
    assert isinstance(multi, list) and len(multi) == 2 and multi[1].tolist() == [0, 2, 4]
    rows = to_series(t.arange(6).reshape(2, 3))
    assert isinstance(rows, list) and len(rows) == 2 and rows[1].tolist() == [3, 4, 5]
    assert to_series(t.arange(3, dtype=t.bfloat16)).dtype == np.float32

def test_imshow_facet_labels():
    for n in (5, 6):  # 5 facets wrapped at 3 leave a partial last row, which plotly stores bottom row first
        arr = np.zeros((n, 2, 2)) + np.arange(n)[:, None, None]
        fig = imshow(arr, facet_col_wrap=3, facet_labels=[f"F{i}" for i in range(n)], return_fig=True)
        default = imshow(arr, facet_col_wrap=3, return_fig=True)  # its titles name each facet's index at the same positions
        assert [a.text for a in fig.layout.annotations] == ["F" + d.text.split("=")[1] for d in default.layout.annotations]
        assert sorted(a.text for a in fig.layout.annotations) == sorted(f"F{i}" for i in range(n))
    fig = imshow(np.zeros((4, 2, 2)), facet_labels=["a", "b", "c", "d"], return_fig=True)  # no wrap
    assert [a.text for a in fig.layout.annotations] == ["a", "b", "c", "d"]
    with pytest.raises(AssertionError):
        imshow(np.zeros((2, 2, 2)), facet_labels=["a", "b", "c"], return_fig=True)
    fig = imshow(t.randn(3, 4), text=[["a"] * 4] * 3, title="T", border=True, xaxis_tickangle=45, return_fig=True)
    assert fig.data[0].texttemplate == "%{text}" and fig.layout.title.text == "T" and fig.layout.xaxis.tickangle == 45 and fig.layout.xaxis.mirror is True

def test_scatter_facet_labels_and_lines():
    x, y = np.arange(5), np.arange(5) * 2
    fig = scatter(x, y, facet_col=list("abcde"), facet_col_wrap=3, facet_labels=[f"F{i}" for i in range(5)], return_fig=True)
    default = scatter(x, y, facet_col=list("abcde"), facet_col_wrap=3, return_fig=True)
    assert [a.text for a in fig.layout.annotations] == [f"F{'abcde'.index(d.text.split('=')[1])}" for d in default.layout.annotations]
    assert len(scatter(x, y, add_line="x = 1.5", return_fig=True).data) == 2 and len(scatter(x, y, add_line="y=x", return_fig=True).data) == 2
    ref = scatter(x, y, add_line="y=3", return_fig=True).data[1]
    assert list(ref.y) == [3.0, 3.0] and list(ref.x) == [0, 4]
    with pytest.raises(ValueError):
        scatter(x, y, add_line="z=1", return_fig=True)
    with pytest.raises(ValueError):
        scatter(x, y, add_line="x=abc", return_fig=True)
    assert scatter(x, y, text=list("abcde"), textposition="top center", return_fig=True).data[0].textposition == "top center"

def test_line_and_bar_labels_and_names():
    single = line(t.arange(4.0), labels={"x": "Layer", "y": "Loss"}, return_fig=True)
    multi = line([t.arange(4.0), t.arange(4.0) * 2], labels={"x": "Layer", "y": "Loss"}, return_fig=True)
    assert single.layout.yaxis.title.text == "Loss" and multi.layout.yaxis.title.text == "Loss" and multi.layout.xaxis.title.text == "Layer"  # several series used to be titled value/index
    names = ["a", "b"]
    fig = line([t.arange(3.0), t.arange(3.0)], names=names, return_fig=True)
    assert [tr.name for tr in fig.data] == ["a", "b"] and names == ["a", "b"]  # the caller's list is not consumed
    fig = bar([t.arange(4.0), t.arange(4.0) * 2], names=names, labels={"x": "Head", "y": "Logit diff"}, return_fig=True)
    assert [tr.name for tr in fig.data] == ["a", "b"] and fig.layout.yaxis.title.text == "Logit diff" and fig.layout.xaxis.title.text == "Head"
    assert bar(t.arange(4.0), labels={"y": "v"}, x=list("abcd"), return_fig=True).layout.yaxis.title.text == "v"
    assert len(line(t.arange(6.0).reshape(2, 3), return_fig=True).data) == 2 and len(line([t.tensor(1.0), t.tensor(2.0)], return_fig=True).data) == 1
    fig = line(t.arange(3.0), xaxis_tickvals=["p", "q", "r"], markers=True, log_y=True, return_fig=True)
    assert list(fig.layout.xaxis.ticktext) == ["p", "q", "r"] and list(fig.layout.xaxis.tickvals) == [0, 1, 2] and fig.layout.yaxis.type == "log"

def test_line_secondary_y():
    fig = line([t.arange(3.0), t.arange(3.0) * 10], x=[5, 6, 7], use_secondary_yaxis=True, names=["a", "b"], labels={"x": "X", "y1": "L", "y2": "R"}, return_fig=True)
    assert [tr.name for tr in fig.data] == ["a", "b"] and list(fig.data[0].x) == [5, 6, 7] and list(fig.data[1].x) == [5, 6, 7]  # one x shared by both
    assert fig.layout.yaxis.title.text == "L" and fig.layout.yaxis2.title.text == "R" and fig.layout.xaxis.title.text == "X"
    fig = line([t.arange(3.0), t.arange(2.0)], x=[[0, 1, 2], [10, 11]], use_secondary_yaxis=True, return_fig=True)
    assert list(fig.data[1].x) == [10, 11] and fig.data[1].name == "yaxis2" and list(fig.data[0].x) == [0, 1, 2]
    fig = line([t.arange(3.0), t.arange(2.0)], use_secondary_yaxis=True, return_fig=True)
    assert list(fig.data[1].x) == [0, 1]

def test_hist():
    names = ["a", "b"]
    fig = hist([t.randn(50), t.randn(50)], names=names, labels={"x": "v", "y": "n"}, add_mean_line=True, return_fig=True)
    assert [tr.name for tr in fig.data] == ["a", "b"] and names == ["a", "b"] and fig.layout.xaxis.title.text == "v" and len(fig.layout.shapes) == 2
    fig = hist(t.randn(50), names=["a"], marginal="box", labels={"x": "v"}, return_fig=True)
    assert [tr.name for tr in fig.data] == ["a", "a"] and fig.layout.xaxis.title.text == "v"
    with pytest.raises(AssertionError):
        hist([t.randn(5), t.randn(5)], marginal="box", return_fig=True)
    assert len(hist(t.randn(2, 20), return_fig=True).data) == 2

def test_uniform_interface_and_lazy_umap():
    for f in (imshow, line, scatter, bar, hist, plot_vocab_umap):
        params = inspect.signature(f).parameters
        assert "return_fig" in params and "renderer" in params, f.__name__
    assert not hasattr(plots, "umap")  # imported inside plot_vocab_umap: its numba compilation is most of the package's import time
