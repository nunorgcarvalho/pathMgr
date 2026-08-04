"""Raster (and vector) export of a path diagram via matplotlib.

**Why matplotlib and not graphviz.** graphviz would do the layout for us, but it needs the `dot`
system binary, which is not installed on these nodes and is not pip-installable; and its layout
is the part we least need, since the pedigree case supplies its own coordinates. matplotlib is
already available, is a pure-python dependency, gives direct control over the three edge styles
(which is the thing that must not blur), renders LaTeX-ish labels through mathtext so the same
`sympy.latex` output feeds both back ends, and writes PNG, SVG and PDF from one code path.

matplotlib is imported **inside** the functions, so ``import pathmgr`` never pulls in a drawing
dependency and the core stays computable without one.
"""

from __future__ import annotations

from pathlib import Path

from ..core.model import Model
from .layout import Layout
from .style import DiagramStyle
from .tikz import highlight_sets

__all__ = ["to_image", "draw_on_axes"]


def _require_matplotlib():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "raster export needs matplotlib, which is an optional dependency: "
            "pip install 'pathmgr[render]'. The TikZ back end needs nothing extra."
        ) from exc
    return matplotlib


def _mathtext(label: str) -> str:
    """Wrap LaTeX math for matplotlib's mathtext, which needs ``$...$``."""
    return f"${label}$" if label else ""


def draw_on_axes(
    model: Model,
    axes,
    layout: Layout | None = None,
    style: DiagramStyle | None = None,
    highlight=None,
    legend: bool = False,
) -> None:
    """Draw ``model`` onto an existing matplotlib Axes. Same conventions as the TikZ back end."""
    _require_matplotlib()
    from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

    style = style or DiagramStyle()
    layout = (layout or Layout()).completed(model)

    if highlight is not None:
        hot_directed, hot_bidirected, hot_copaths = highlight_sets(highlight)
    else:
        hot_directed, hot_bidirected, hot_copaths = set(), set(), set()
    highlighting = highlight is not None

    def colour_for(default: str, hot: bool) -> str:
        if hot:
            return style.highlight_colour
        if highlighting and style.fade_unhighlighted:
            return style.faded_colour
        return default

    def width_for(base: float, hot: bool) -> float:
        return base * (style.highlight_scale if hot else 1.0)

    # -- nodes -------------------------------------------------------------------------
    for variable in model.variables:
        x, y = layout[variable.name]
        if variable.latent:
            patch = Ellipse(
                (x, y),
                style.node_width * 1.25,
                style.node_height,
                facecolor=style.node_fill,
                edgecolor=style.node_colour,
                zorder=3,
            )
        else:
            patch = Rectangle(
                (x - style.node_width / 2, y - style.node_height / 2),
                style.node_width,
                style.node_height,
                facecolor=style.node_fill,
                edgecolor=style.node_colour,
                zorder=3,
            )
        axes.add_patch(patch)
        axes.text(
            x,
            y,
            _mathtext(style.node_label(variable.name, variable.label)),
            ha="center",
            va="center",
            zorder=4,
            fontsize=9,
        )

    def edge_label(position, label, hot):
        if not label:
            return
        axes.text(
            position[0],
            position[1],
            _mathtext(label),
            ha="center",
            va="center",
            fontsize=7,
            color=colour_for("black", hot),
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
        )

    # -- directed paths ----------------------------------------------------------------
    for edge in model.directed_edges:
        hot = (edge.src, edge.dst) in hot_directed
        start, end = layout[edge.src], layout[edge.dst]
        axes.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=style.arrow_head_size,
                shrinkA=style.node_clearance,
                shrinkB=style.node_clearance,
                linewidth=width_for(style.directed_width, hot),
                color=colour_for(style.directed_colour, hot),
                zorder=2,
            )
        )
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        edge_label(midpoint, style.edge_label((edge.src, edge.dst), edge.coeff), hot)

    # -- bidirected covariances: CURVED, two arrowheads --------------------------------
    for edge in model.bidirected_edges:
        if edge.is_variance and not style.show_variances:
            continue
        hot = frozenset((edge.a, edge.b)) in hot_bidirected
        colour = colour_for(style.bidirected_colour, hot)
        width = width_for(style.bidirected_width, hot)
        if edge.is_variance:
            x, y = layout[edge.a]
            radius = style.node_height * 0.55
            axes.add_patch(
                FancyArrowPatch(
                    (x - radius, y + style.node_height * 0.5),
                    (x + radius, y + style.node_height * 0.5),
                    connectionstyle="arc3,rad=-1.6",
                    arrowstyle="<|-|>",
                    mutation_scale=style.arrow_head_size * 0.7,
                    linewidth=width,
                    color=colour,
                    zorder=2,
                )
            )
            edge_label(
                (x, y + style.node_height * 0.5 + radius * 1.5),
                style.edge_label((edge.a, edge.b), edge.value),
                hot,
            )
        else:
            start, end = layout[edge.a], layout[edge.b]
            rad = style.bidirected_bend / 100.0
            axes.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="<|-|>",
                    mutation_scale=style.arrow_head_size,
                    shrinkA=style.node_clearance,
                    shrinkB=style.node_clearance,
                    linewidth=width,
                    color=colour,
                    zorder=2,
                )
            )
            edge_label(_arc_midpoint(start, end, rad),
                       style.edge_label((edge.a, edge.b), edge.value), hot)

    # -- co-paths: STRAIGHT, no arrowheads, thicker, distinct colour -------------------
    seen_pairs: dict[frozenset, int] = {}
    for copath in model.copaths:
        pair = frozenset((copath.a, copath.b))
        seen = seen_pairs.get(pair, 0)
        seen_pairs[pair] = seen + 1
        hot = pair in hot_copaths
        start, end = layout[copath.a], layout[copath.b]
        rad = -0.12 * seen  # keep several co-paths on one couple separable
        axes.add_patch(
            FancyArrowPatch(
                start,
                end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-",  # NO arrowheads: that is the point
                shrinkA=style.node_clearance,
                shrinkB=style.node_clearance,
                linewidth=width_for(style.copath_width, hot),
                color=colour_for(style.copath_colour, hot),
                zorder=2,
            )
        )
        edge_label(_arc_midpoint(start, end, rad),
                   style.edge_label((copath.a, copath.b), copath.coefficient), hot)

    # -- frame -------------------------------------------------------------------------
    min_x, min_y, max_x, max_y = layout.bounds()
    pad = 1.0
    axes.set_xlim(min_x - pad, max_x + pad)
    axes.set_ylim(min_y - pad * 1.4, max_y + pad)
    axes.set_aspect("equal")
    axes.axis("off")

    if legend:
        _draw_legend(axes, style)

    if highlighting:
        axes.set_title(
            _mathtext(highlight.tex_path({v.name: v.label for v in model.variables if v.label})),
            fontsize=9,
        )


def _arc_midpoint(start, end, rad: float):
    """Midpoint of matplotlib's ``arc3`` quadratic Bezier, so a label sits ON its curve.

    ``arc3`` puts the control point at the chord midpoint displaced by ``rad`` times the chord
    rotated a quarter turn *clockwise* -- note the sign, which is the opposite of the obvious
    guess and put every curved-edge label on the wrong side of its arc until it was checked
    against a rendered figure. A quadratic Bezier's midpoint is ``(p0 + 2*c + p1) / 4``.
    """
    x0, y0 = start
    x1, y1 = end
    mid = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    control = (mid[0] + rad * (y1 - y0), mid[1] - rad * (x1 - x0))
    return (
        (x0 + 2 * control[0] + x1) / 4.0,
        (y0 + 2 * control[1] + y1) / 4.0,
    )


def _draw_legend(axes, style: DiagramStyle) -> None:
    """A key for the three edge types -- the antidote to confusing a co-path for a covariance."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=style.directed_colour, lw=style.directed_width, marker=">",
               markersize=5, label="directed: causal effect"),
        Line2D([0], [0], color=style.bidirected_colour, lw=style.bidirected_width, marker="d",
               markersize=4, label="bidirected: exogenous covariance"),
        Line2D([0], [0], color=style.copath_colour, lw=style.copath_width,
               label="co-path: covariance from matching"),
    ]
    axes.legend(handles=handles, loc="lower center", fontsize=7, frameon=True,
                ncol=1, borderpad=0.4)


def to_image(
    model: Model,
    path: str | Path,
    layout: Layout | None = None,
    style: DiagramStyle | None = None,
    highlight=None,
    legend: bool = False,
    dpi: int = 200,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """Render ``model`` to a file. Format follows the suffix: ``.png``, ``.svg``, ``.pdf``.

    >>> import pathmgr as pm  # doctest: +SKIP
    >>> pm.render.to_image(model, "diagram.png", legend=True)  # doctest: +SKIP
    """
    _require_matplotlib()
    import matplotlib

    matplotlib.use("Agg")  # headless: no display on a compute node
    import matplotlib.pyplot as plt

    style = style or DiagramStyle()
    completed = (layout or Layout()).completed(model)
    if figsize is None:
        min_x, min_y, max_x, max_y = completed.bounds()
        figsize = (
            max(4.0, min(20.0, (max_x - min_x) * 1.1 + 2.5)),
            max(3.0, min(20.0, (max_y - min_y) * 1.1 + 2.5)),
        )

    figure, axes = plt.subplots(figsize=figsize)
    try:
        draw_on_axes(model, axes, layout=completed, style=style, highlight=highlight,
                     legend=legend)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path
