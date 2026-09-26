import re

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch as t
import umap
from plotly.subplots import make_subplots
from torch import Tensor

def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (t.Tensor, t.nn.parameter.Parameter)):
        tensor = tensor.detach().cpu()
        return tensor.float().numpy() if tensor.dtype == t.bfloat16 else tensor.numpy()  # numpy has no bfloat16
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")

def to_series(values):
    """
    Normalize plot data to a 1D array, or to a list of 1D arrays if it holds several series.

    Plotly express renames the series it's handed in place, so it needs a list of them rather than a 2D array.
    """
    arr = [to_numpy(series) for series in values] if isinstance(values, list) and not isinstance(values[0], (int, float)) else to_numpy(values)
    return list(arr) if isinstance(arr, np.ndarray) and arr.ndim == 2 else arr


def drop_none(**kwargs) -> dict:
    """
    Drop unset (None) args, so that plotly's own defaults apply to them.
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def margin_dict(margin: int | dict | None) -> dict:
    """
    Layout kwargs for a margin, where an int means the same padding on all four sides.
    """
    if margin is None:
        return {}
    return {"margin": dict.fromkeys("tblr", margin) if isinstance(margin, int) else margin}


def imshow(
    tensor: t.Tensor,
    renderer=None,
    *,
    x: list | None = None,
    y: list | None = None,
    labels: dict | None = None,
    title: str | None = None,
    template: str | None = None,
    size: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    aspect: str | None = None,
    origin: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    range_color: tuple[float, float] | None = None,
    color_continuous_scale: str = "RdBu",
    color_continuous_midpoint: float | None = 0.0,
    facet_col: int | None = None,
    facet_col_wrap: int | None = None,
    animation_frame: int | None = None,
    facet_labels: list[str] | None = None,
    text: list | None = None,
    border: bool = False,
    xaxis_tickangle: float | None = None,
    margin: int | dict | None = None,
    static: bool = False,
    return_fig: bool = False,
    **layout_kwargs,
):
    """
    Heatmap of a 2D (or 3D, faceted) tensor. Extra keyword args go to fig.update_layout.

    Args:
        tensor: 2D array to plot, or 3D for one facet per leading index.
        renderer: plotly renderer to show with, e.g. "browser". None uses the default.
        x, y: tick labels for the columns / rows.
        labels: axis and colorbar names, e.g. {"x": "Head", "y": "Layer", "color": "logit diff"}.
        title: figure title.
        template: plotly theme, e.g. "simple_white".
        size: (height, width) in pixels, shorthand for passing both.
        height, width: figure size in pixels.
        aspect: "equal" for square cells, "auto" to fill the figure.
        origin: "upper" (default) puts row 0 at the top, "lower" at the bottom.
        zmin, zmax: values at the ends of the color scale. Default is the data range.
        range_color: (low, high), another way to set the color scale limits.
        color_continuous_scale: named colorscale. Diverging red-blue by default.
        color_continuous_midpoint: value mapped to the middle color. None for a non-diverging scale.
        facet_col: axis to facet over. A 3D tensor facets over axis 0 by default.
        facet_col_wrap: max facets per row before wrapping to the next row.
        animation_frame: axis to turn into an animation slider instead of facets.
        facet_labels: titles for the facets, in order.
        text: strings to write in the cells: list[list[str]] for 2D, or one of those per facet.
        border: draw a black box around the plot area.
        xaxis_tickangle: rotation of the x tick labels in degrees, applied to every facet.
        margin: padding in pixels. An int sets all four sides.
        static: show as a non-interactive image.
        return_fig: return the figure instead of showing it.
    """
    arr = to_numpy(tensor)
    if size is not None:
        height, width = size
    if arr.ndim == 3 and animation_frame is None and facet_col is None:  # otherwise px would read the 3rd axis as color channels
        facet_col = 0
    px_kwargs = drop_none(
        x=x, y=y, labels=labels, title=title, template=template, height=height, width=width,
        aspect=aspect, origin=origin, zmin=zmin, zmax=zmax, range_color=range_color,
        color_continuous_scale=color_continuous_scale, color_continuous_midpoint=color_continuous_midpoint,
        facet_col=facet_col, facet_col_wrap=facet_col_wrap, animation_frame=animation_frame,
    )
    fig = px.imshow(arr, **px_kwargs).update_layout(**layout_kwargs, **margin_dict(margin))
    if facet_labels:
        set_facet_labels(fig, facet_labels, facet_col_wrap)
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    if text:
        if arr.ndim == 2:
            # if 2D, then we assume text is a list of lists of strings
            assert isinstance(text[0], list)
            assert isinstance(text[0][0], str)
            text = [text]
        else:
            # if 3D, then text is either repeated for each facet, or different
            assert isinstance(text[0], list)
            if isinstance(text[0][0], str):
                text = [text for _ in range(len(fig.data))]
        for i, _text in enumerate(text):
            fig.data[i].update(
                text=_text, 
                texttemplate="%{text}", 
                textfont={"size": 12}
            )
    if xaxis_tickangle is not None:  # update_xaxes hits every facet, unlike update_layout(xaxis_...)
        fig.update_xaxes(tickangle=xaxis_tickangle)
    return fig if return_fig else fig.show(renderer=renderer, config={"staticPlot": static})


def reorder_list_in_plotly_way(L: list, col_wrap: int):
    '''
    Helper function, because Plotly orders figures in an annoying way when there's column wrap.
    '''
    L_new = []
    while len(L) > 0:
        L_new.extend(L[-col_wrap:])
        L = L[:-col_wrap]
    return L_new


def set_facet_labels(fig, facet_labels: list[str], facet_col_wrap: int | None):
    """
    Rename the facet titles, undoing plotly's bottom-row-first ordering when the facets wrap.
    """
    assert len(facet_labels) <= len(fig.layout.annotations), f"got {len(facet_labels)} facet_labels but the figure has {len(fig.layout.annotations)} facet titles"
    if facet_col_wrap is not None:
        facet_labels = reorder_list_in_plotly_way(facet_labels, facet_col_wrap)
    for i, label in enumerate(facet_labels):
        fig.layout.annotations[i]['text'] = label


def line(
    y: t.Tensor | list[t.Tensor],
    renderer=None,
    *,
    x=None,
    names: list[str] | None = None,
    labels: dict | None = None,
    title: str | None = None,
    template: str | None = None,
    size: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    color=None,
    markers: bool = False,
    log_y: bool = False,
    hover_name=None,
    xaxis_tickvals: list | None = None,
    use_secondary_yaxis: bool = False,
    hovermode: str = "closest",
    margin: int | dict | None = None,
    return_fig: bool = False,
    **layout_kwargs,
):
    """
    Line plot of one series, or of several if `y` is a list of them. Extra keyword args go to fig.update_layout.

    Args:
        y: 1D values to plot, or several series to plot as separate lines: either a list of 1D series or a 2D array with one line per row.
        renderer: plotly renderer to show with, e.g. "browser". None uses the default.
        x: x values for the points, shared by every line. Defaults to 0, 1, 2, ... With use_secondary_yaxis, a pair of x arrays.
        names: legend name per line, in order, one per trace.
        labels: axis names, e.g. {"x": "Layer", "y": "Loss"}. With use_secondary_yaxis, keys are "x", "y1", "y2".
        title: figure title.
        template: plotly theme, e.g. "simple_white".
        size: (height, width) in pixels, shorthand for passing both.
        height, width: figure size in pixels.
        color: values to color the lines by (one line per distinct value).
        markers: draw a marker at each point.
        log_y: log-scale the y axis.
        hover_name: label shown in bold in each point's hover box.
        xaxis_tickvals: tick labels for the x axis, placed at `x` (or at 0, 1, 2, ...).
        use_secondary_yaxis: plot y[0] against a left axis and y[1] against a right one, for series of different scales.
        hovermode: plotly hover behaviour. Defaults to one shared box for all lines at that x.
        margin: padding in pixels. An int sets all four sides.
        return_fig: return the figure instead of showing it.
    """
    if size is not None:
        height, width = size
    layout = {"hovermode": hovermode, **layout_kwargs, **margin_dict(margin)}
    if xaxis_tickvals is not None:
        layout["xaxis"] = dict(
            tickmode = "array",
            tickvals = x if x is not None else np.arange(len(xaxis_tickvals)),
            ticktext = xaxis_tickvals
        )
    if use_secondary_yaxis:
        if labels is not None:
            layout["yaxis_title_text"] = labels.get("y1", None)
            layout["yaxis2_title_text"] = labels.get("y2", None)
            layout["xaxis_title_text"] = labels.get("x", None)
        layout.update(drop_none(title=title, template=template, width=width, height=height))
        fig = make_subplots(specs=[[{"secondary_y": True}]]).update_layout(**layout)
        y0 = to_numpy(y[0])
        y1 = to_numpy(y[1])
        x0, x1 = x if x is not None else [np.arange(len(y0)), np.arange(len(y1))]
        name0, name1 = names if names is not None else ["yaxis1", "yaxis2"]
        fig.add_trace(go.Scatter(y=y0, x=x0, name=name0), secondary_y=False)
        fig.add_trace(go.Scatter(y=y1, x=x1, name=name1), secondary_y=True)
    else:
        y = to_series(y)
        px_kwargs = drop_none(
            x=x, labels=labels, title=title, template=template, height=height, width=width,
            color=color, markers=markers, log_y=log_y, hover_name=hover_name,
        )
        fig = px.line(y=y, **px_kwargs).update_layout(**layout)
        if names is not None:
            assert len(names) == len(fig.data), f"got {len(names)} names for {len(fig.data)} lines"
            for trace, name in zip(fig.data, names):
                trace.update(name=name)
    return fig if return_fig else fig.show(renderer=renderer)


def scatter(
    x,
    y,
    renderer=None,
    *,
    labels: dict | None = None,
    title: str | None = None,
    template: str | None = None,
    size: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    color=None,
    hover_name=None,
    text=None,
    opacity: float | None = None,
    trendline: str | None = None,
    facet_col=None,
    facet_col_wrap: int | None = None,
    facet_labels: list[str] | None = None,
    add_line: str | None = None,
    textposition: str | None = None,
    margin: int | dict | None = None,
    return_fig: bool = False,
    **layout_kwargs,
):
    """
    Scatter plot of y against x. Extra keyword args go to fig.update_layout.

    Args:
        x, y: point coordinates.
        renderer: plotly renderer to show with, e.g. "browser". None uses the default.
        labels: axis and legend names, e.g. {"x": "Open-proportion", "y": "Contribution"}.
        title: figure title.
        template: plotly theme, e.g. "simple_white".
        size: (height, width) in pixels, shorthand for passing both.
        height, width: figure size in pixels.
        color: values to color the points by.
        hover_name: label shown in bold in each point's hover box.
        text: labels drawn next to each point.
        opacity: point opacity, 0 to 1. Useful when points overlap.
        trendline: fit to overlay, e.g. "ols" or "lowess".
        facet_col: values to split the plot into one subplot per distinct value.
        facet_col_wrap: max facets per row before wrapping to the next row.
        facet_labels: titles for the facets, in order.
        add_line: reference line to draw, one of "y=x", "x=c" or "y=c" for a float c.
        textposition: where `text` sits relative to its point, e.g. "top center".
        margin: padding in pixels. An int sets all four sides.
        return_fig: return the figure instead of showing it.
    """
    x = to_numpy(x)
    y = to_numpy(y)
    if size is not None:
        height, width = size
    px_kwargs = drop_none(
        labels=labels, title=title, template=template, height=height, width=width, color=color,
        hover_name=hover_name, text=text, opacity=opacity, trendline=trendline,
        facet_col=facet_col, facet_col_wrap=facet_col_wrap,
    )
    fig = px.scatter(y=y, x=x, **px_kwargs).update_layout(**layout_kwargs, **margin_dict(margin))
    if add_line is not None:
        xrange = fig.layout.xaxis.range or [x.min(), x.max()]
        yrange = fig.layout.yaxis.range or [y.min(), y.max()]
        add_line = add_line.replace(" ", "")
        if add_line in ["x=y", "y=x"]:
            fig.add_trace(go.Scatter(mode='lines', x=xrange, y=xrange, showlegend=False))
        elif re.match("(x|y)=", add_line):
            try: c = float(add_line.split("=")[1])
            except: raise ValueError(f"Unrecognized add_line: {add_line}. Please use either 'x=y' or 'x=c' or 'y=c' for some float c.")
            x, y = ([c, c], yrange) if add_line[0] == "x" else (xrange, [c, c])
            fig.add_trace(go.Scatter(mode='lines', x=x, y=y, showlegend=False))
        else:
            raise ValueError(f"Unrecognized add_line: {add_line}. Please use either 'x=y' or 'x=c' or 'y=c' for some float c.")
    if facet_labels:
        set_facet_labels(fig, facet_labels, facet_col_wrap)
    if textposition is not None:
        fig.update_traces(textposition=textposition)
    return fig if return_fig else fig.show(renderer=renderer)


def bar(
    tensor,
    renderer=None,
    *,
    x=None,
    names: list[str] | None = None,
    labels: dict | None = None,
    title: str | None = None,
    template: str | None = None,
    size: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    color=None,
    text_auto=None,
    hovermode: str = "x unified",
    margin: int | dict | None = None,
    return_fig: bool = False,
    **layout_kwargs,
):
    """
    Bar chart of a 1D tensor, or of several as grouped bars. Extra keyword args go to fig.update_layout.

    Args:
        tensor: 1D values for the bar heights, or several series as grouped bars: either a list of 1D series or a 2D array with one series per row.
        renderer: plotly renderer to show with, e.g. "browser". None uses the default.
        x: labels for the bars, shared by every series. Defaults to 0, 1, 2, ...
        names: legend name per series, in order.
        labels: axis names, e.g. {"x": "Head", "y": "Logit diff"}.
        title: figure title.
        template: plotly theme, e.g. "simple_white".
        size: (height, width) in pixels, shorthand for passing both.
        height, width: figure size in pixels.
        color: values to color the bars by.
        text_auto: write each bar's value on it. True, or a format string like ".2f".
        hovermode: plotly hover behaviour. Defaults to one shared box for all bars at that x.
        margin: padding in pixels. An int sets all four sides.
        return_fig: return the figure instead of showing it.
    """
    arr = to_series(tensor)
    if size is not None:
        height, width = size
    px_kwargs = drop_none(
        x=x, labels=labels, title=title, template=template, height=height, width=width,
        color=color, text_auto=text_auto,
    )
    fig = px.bar(y=arr, **px_kwargs).update_layout(hovermode=hovermode, **layout_kwargs, **margin_dict(margin))
    if names is not None:
        for i in range(len(fig.data)):
            fig.data[i]["name"] = names[i]
    return fig if return_fig else fig.show(renderer=renderer)


def hist(
    tensor,
    renderer=None,
    *,
    names: list[str] | None = None,
    labels: dict | None = None,
    title: str | None = None,
    template: str | None = None,
    size: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    nbins: int | None = None,
    opacity: float | None = None,
    histnorm: str | None = None,
    color=None,
    marginal: str | None = None,
    add_mean_line: bool = False,
    barmode: str = "overlay",
    bargap: float = 0.0,
    hovermode: str = "x unified",
    autosize: bool = False,
    margin: int | dict | None = None,
    static: bool = False,
    return_fig: bool = False,
    **layout_kwargs,
):
    """
    Histogram of a tensor, or of several overlaid. Extra keyword args go to fig.update_layout.

    Args:
        tensor: 1D values to bin, or several series to overlay: either a list of 1D series or a 2D array with one series per row.
        renderer: plotly renderer to show with, e.g. "browser". None uses the default.
        names: legend name per series, in order.
        labels: axis names, e.g. {"x": "Logit diff", "y": "Count"}.
        title: figure title.
        template: plotly theme, e.g. "simple_white".
        size: (height, width) in pixels, shorthand for passing both.
        height, width: figure size in pixels.
        nbins: number of bins. Plotly picks one by default.
        opacity: bar opacity, 0 to 1. Useful for overlaid histograms.
        histnorm: what the bar heights mean, e.g. "probability" or "percent". Counts by default.
        color: values to split the data into differently-colored histograms by. Only for a single series.
        marginal: extra distribution plot above the bars, e.g. "box" or "rug". Only for a single series.
        add_mean_line: draw a dashed vertical line at each series' mean.
        barmode: how overlapping bars are drawn: "overlay", "group" or "stack".
        bargap: gap between bars, 0 to 1.
        hovermode: plotly hover behaviour. Defaults to one shared box for all series at that x.
        autosize: let the figure resize itself to its container.
        margin: padding in pixels. An int sets all four sides.
        static: show as a non-interactive image.
        return_fig: return the figure instead of showing it.
    """
    arr = to_series(tensor)
    if size is not None:
        height, width = size
    layout = dict(
        barmode=barmode, bargap=bargap, hovermode=hovermode, autosize=autosize,
        modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
    ) | layout_kwargs | margin_dict(margin)

    # If `arr` has a list of arrays, then just doing px.histogram doesn't work annoyingly enough
    # This is janky, even for my functions!
    if isinstance(arr, list) and isinstance(arr[0], np.ndarray):
        assert marginal is None, "Can't use `marginal` with a list of arrays"
        assert color is None, "Can't use `color` with a list of arrays"
        layout.update(drop_none(title=title, template=template, height=height, width=width))
        if labels is not None:
            layout["xaxis_title_text"] = labels.get("x", "")
            layout["yaxis_title_text"] = labels.get("y", "")
        fig = go.Figure(layout=go.Layout(**layout))
        assert names is None or len(names) == len(arr), f"got {len(names)} names for {len(arr)} series"
        for i, x in enumerate(arr):
            fig.add_trace(go.Histogram(x=x, name=names[i] if names is not None else None, nbinsx=nbins, opacity=opacity, histnorm=histnorm, bingroup="x"))  # bingroup makes the series share bin edges, like px.histogram does
    else:
        px_kwargs = drop_none(
            labels=labels, title=title, template=template, height=height, width=width,
            nbins=nbins, opacity=opacity, histnorm=histnorm, color=color, marginal=marginal,
        )
        fig = px.histogram(x=arr, **px_kwargs).update_layout(**layout)
        if names is not None:
            for i in range(len(fig.data)):
                fig.data[i]["name"] = names[i // 2 if marginal is not None else i]

    if add_mean_line:
        for series in (arr if isinstance(arr, list) else [arr]):
            fig.add_vline(x=series.mean(), line_width=3, line_dash="dash", line_color="black", annotation_text=f"Mean = {series.mean():.3f}", annotation_position="top")
    return fig if return_fig else fig.show(renderer=renderer, config={"staticPlot": static})

def plot_vocab_umap(x: Tensor, labels: Tensor, tokenizer, n_points: int = 20_000, pca_dim: int = 128, seed: int = 0):
    """UMAP of a random subset of token vectors x [vocab, d], colored by cluster, hover shows token and cluster id."""
    idx = t.randperm(len(x), generator=t.Generator().manual_seed(seed))[:n_points]
    sub = x[idx].float()
    sub = (sub @ t.pca_lowrank(sub, q=pca_dim)[2]).cpu().numpy()
    emb = umap.UMAP(metric="cosine", random_state=seed).fit_transform(sub)
    lab = labels[idx].cpu().numpy()
    hover = [f"{tokenizer.decode([i])!r}<br>cluster {c}" for i, c in zip(idx.tolist(), lab.tolist())]
    fig = go.Figure(go.Scattergl(x=emb[:, 0], y=emb[:, 1], mode="markers", text=hover, hoverinfo="text",
                                 marker=dict(size=3, color=(lab * 0.618) % 1, colorscale="Phase", showscale=False)))
    fig.update_layout(title=f"UMAP of {n_points} token vectors, {labels.max().item() + 1} clusters", width=1000, height=800, margin=dict(l=10, r=10, t=40, b=10))
    fig.show()
