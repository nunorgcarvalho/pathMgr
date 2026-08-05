"""Raster (and vector) export of a path diagram via matplotlib.

**Why matplotlib and not graphviz.** graphviz would do the layout for us, but it needs the `dot`
system binary, which is not installed on these nodes and is not pip-installable; and its layout is
the part we least need, since the pedigree case supplies its own coordinates. matplotlib is already
available, is a pure-python dependency, gives direct control over the three edge styles (which is
the thing that must not blur), renders LaTeX-ish labels through mathtext so the same `sympy.latex`
output feeds both back ends, and writes PNG, SVG and PDF from one code path.

matplotlib is imported **inside** the functions, so ``import pathmgr`` never pulls in a drawing
dependency and the core stays computable without one.

Edges are clipped at each node's real boundary (see :mod:`pathmgr.render.placement`) rather than
shrunk by a constant, because matplotlib will otherwise happily draw an arrow to a patch's centre
and let the patch cover the arrowhead.

Draw order matters and is chosen to match TikZ: node **fills** below the edges, edges above them,
node **text** above the edges again. So an edge that passes over an unrelated node stays visible
(matplotlib would otherwise hide it behind the patch, which TikZ does not do) while no arrow is
ever drawn across a node's label.
"""

from __future__ import annotations

from pathlib import Path

from ..core.model import Model
from .layout import Layout
from .placement import boundary_point, labelled_edges, node_rect, place_labels
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


def _arc_midpoint(start, end, rad: float):
    """Midpoint of matplotlib's ``arc3`` quadratic Bezier, so a label sits ON its curve.

    ``arc3`` puts the control point at the chord midpoint displaced by ``rad`` times the chord
    rotated a quarter turn *clockwise* -- note the sign, which is the opposite of the obvious guess
    and put every curved-edge label on the wrong side of its arc until it was checked against a
    rendered figure. A quadratic Bezier's midpoint is ``(p0 + 2*c + p1) / 4``.
    """
    x0, y0 = start
    x1, y1 = end
    mid = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    control = (mid[0] + rad * (y1 - y0), mid[1] - rad * (x1 - x0))
    return ((x0 + 2 * control[0] + x1) / 4.0, (y0 + 2 * control[1] + y1) / 4.0)


def draw_on_axes(
    model: Model,
    axes,
    layout: Layout | None = None,
    style: DiagramStyle | None = None,
    highlight=None,
    legend: bool = False,
    caption: str | None = None,
    caption_name: str | None = None,
) -> None:
    """Draw ``model`` onto an existing matplotlib Axes. Same conventions as the TikZ back end."""
    _require_matplotlib()
    from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

    style = style or DiagramStyle()
    layout = (layout or Layout()).completed(model)

    if highlight is not None:
        hot = highlight_sets(highlight)
    else:
        hot = (set(), set(), set())
    hot_directed, hot_bidirected, hot_copaths = hot
    highlighting = highlight is not None

    placements = place_labels(model, layout, style, labelled_edges(model, style))
    rects = {name: node_rect(name, model, layout, style) for name in model.names}

    def colour_for(default: str, is_hot: bool) -> str:
        if is_hot:
            return style.highlight_colour
        if highlighting and style.fade_unhighlighted:
            return style.faded_colour
        return default

    def width_for(base: float, is_hot: bool) -> float:
        return base * (style.highlight_scale if is_hot else 1.0)

    def draw_label(key, text, is_hot, fallback=None):
        if not text:
            return
        placement = placements.get(key)
        point = placement.point if placement else fallback
        if point is None:
            return
        axes.text(
            point[0],
            point[1],
            _mathtext(text),
            ha="center",
            va="center",
            fontsize=7,
            color=colour_for("black", is_hot),
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9),
        )

    # -- node fills first (below the edges); their text is drawn at a higher zorder so an
    # -- edge crossing a node stays visible without ever running across a label ---------
    for variable in model.variables:
        x, y = layout[variable.name]
        rect = rects[variable.name]
        if variable.latent:
            patch = Ellipse(
                (x, y),
                rect.width,
                rect.height,
                facecolor=style.node_fill,
                edgecolor=style.node_colour,
                zorder=1,
            )
        else:
            patch = Rectangle(
                (x - rect.width / 2, y - rect.height / 2),
                rect.width,
                rect.height,
                facecolor=style.node_fill,
                edgecolor=style.node_colour,
                zorder=1,
            )
        axes.add_patch(patch)
        axes.text(
            x,
            y,
            _mathtext(style.node_label(variable.name, variable.label)),
            ha="center",
            va="center",
            zorder=5,
            fontsize=9,
        )

    # -- edges, above the node fills ----------------------------------
    for edge in model.directed_edges:
        is_hot = (edge.src, edge.dst) in hot_directed
        start = boundary_point(
            layout[edge.src], layout[edge.dst], rects[edge.src], style.node_clearance
        )
        end = boundary_point(
            layout[edge.dst], layout[edge.src], rects[edge.dst], style.node_clearance
        )
        axes.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=style.arrow_head_size,
                shrinkA=0,
                shrinkB=0,
                linewidth=width_for(style.directed_width, is_hot),
                color=colour_for(style.directed_colour, is_hot),
                zorder=2,
            )
        )
        draw_label(
            (edge.src, edge.dst),
            style.edge_label((edge.src, edge.dst), edge.coeff),
            is_hot,
        )

    for edge in model.bidirected_edges:
        is_hot = frozenset((edge.a, edge.b)) in hot_bidirected
        if edge.is_variance and not style.draws_variance(is_hot):
            continue
        colour = colour_for(style.bidirected_colour, is_hot)
        width = width_for(style.bidirected_width, is_hot)
        text = style.edge_label((edge.a, edge.b), edge.value)
        if edge.is_variance:
            x, y = layout[edge.a]
            top = y + rects[edge.a].height / 2
            radius = max(0.16, rects[edge.a].height * 0.4)
            axes.add_patch(
                FancyArrowPatch(
                    (x - radius, top),
                    (x + radius, top),
                    connectionstyle="arc3,rad=-1.7",
                    arrowstyle="<|-|>",
                    mutation_scale=style.arrow_head_size * 0.6,
                    linewidth=width,
                    color=colour,
                    zorder=2,
                )
            )
            draw_label((edge.a, edge.a), text, is_hot, fallback=(x, top + radius * 1.75))
        else:
            rad = style.bidirected_bend / 100.0
            start = boundary_point(
                layout[edge.a], layout[edge.b], rects[edge.a], style.node_clearance
            )
            end = boundary_point(
                layout[edge.b], layout[edge.a], rects[edge.b], style.node_clearance
            )
            axes.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="<|-|>",
                    mutation_scale=style.arrow_head_size,
                    shrinkA=0,
                    shrinkB=0,
                    linewidth=width,
                    color=colour,
                    zorder=2,
                )
            )
            draw_label(
                (edge.a, edge.b), text, is_hot, fallback=_arc_midpoint(start, end, rad)
            )

    seen_pairs: dict[frozenset, int] = {}
    for copath in model.copaths:
        pair = frozenset((copath.a, copath.b))
        index = seen_pairs.get(pair, 0)
        seen_pairs[pair] = index + 1
        is_hot = pair in hot_copaths
        rad = -0.12 * index
        start = boundary_point(
            layout[copath.a], layout[copath.b], rects[copath.a], style.node_clearance
        )
        end = boundary_point(
            layout[copath.b], layout[copath.a], rects[copath.b], style.node_clearance
        )
        axes.add_patch(
            FancyArrowPatch(
                start,
                end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-",  # NO arrowheads: that is the point
                shrinkA=0,
                shrinkB=0,
                linewidth=width_for(style.copath_width, is_hot),
                color=colour_for(style.copath_colour, is_hot),
                zorder=2,
            )
        )
        draw_label(
            (copath.a, copath.b),
            style.edge_label((copath.a, copath.b), copath.coefficient),
            is_hot,
            fallback=_arc_midpoint(start, end, rad),
        )

    # -- frame -------------------------------------------------------------------------
    min_x, min_y, max_x, max_y = layout.bounds()
    pad = 0.9
    axes.set_xlim(min_x - pad, max_x + pad)
    axes.set_ylim(min_y - pad * 1.3, max_y + pad)
    axes.set_aspect("equal")
    axes.axis("off")

    if legend:
        _draw_legend(axes, style)

    if caption is None and highlighting:
        labels = {v.name: v.label for v in model.variables if v.label}
        # mathtext has no \\ line break, so render the two-line caption as two lines of text
        caption = highlight.tex_caption(labels, name=caption_name).replace("\\\\", "\n")
    if caption:
        head, _, tail = caption.partition("\n")
        title = _mathtext(head) + ("\n" + _mathtext(tail) if tail else "")
        axes.set_title(title, fontsize=8, pad=12)


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
    caption: str | None = None,
    caption_name: str | None = None,
    dpi: int = 200,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """Render ``model`` to a file. Format follows the suffix: ``.png``, ``.svg``, ``.pdf``."""
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
                     legend=legend, caption=caption, caption_name=caption_name)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path
