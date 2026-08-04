"""TikZ export of a path diagram, for dropping straight into a LaTeX writeup.

Two forms: :func:`to_tikz` emits a bare ``tikzpicture`` to paste into an existing document, and
:func:`to_standalone` wraps it in a minimal compilable file. :func:`write_pdf` compiles one.

Conventions (see :mod:`pathmgr.render.style` for why the co-path is over-differentiated):

    observed variable    rectangle
    latent variable      ellipse
    directed  a -> b     straight, one arrowhead
    bidirected a <-> b   curved, two arrowheads; a self-edge is a variance loop
    co-path   a -- b     straight, NO arrowheads, thicker, distinct colour

Required packages: ``tikz`` (with the ``shapes.geometric`` library, for the latent ellipse) and
``xcolor``. Deliberately nothing else -- the arrowheads are TikZ's built-in ``->``/``<->`` tips
rather than ``arrows.meta``, because that library is absent from a plain TinyTeX install, and
``standalone.cls`` is too. So the default document class for :func:`to_standalone` is ``article``
with ``geometry``, which is present everywhere; pass ``document_class="standalone"`` if you have
it. The emitted picture declares its own styles and depends on nothing else in the host document.

Nothing here imports the RAM engine or the tracer: a model is renderable without computing
anything. Highlighting takes an already-computed :class:`pathmgr.Chain`, so the dependency runs
the right way.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..core.model import Model
from .layout import Layout
from .style import DiagramStyle

__all__ = [
    "TIKZ_LIBRARIES",
    "TikzCompileError",
    "highlight_sets",
    "to_standalone",
    "to_tikz",
    "write_pdf",
]

#: libraries the emitted picture needs -- kept to the one that is genuinely required
TIKZ_LIBRARIES = ("shapes.geometric",)


class TikzCompileError(RuntimeError):
    """LaTeX failed to compile the emitted diagram."""


def _node_id(name: str) -> str:
    """A TikZ-safe node identifier. TikZ chokes on ``,`` ``(`` ``)`` ``.`` ``:`` ``;``."""
    return re.sub(r"[^A-Za-z0-9_]", "-", name)


def highlight_sets(chain) -> tuple[set, set, set]:
    """``(directed edges, bidirected pairs, co-path pairs)`` used by ``chain``.

    Directed edges are ordered ``(src, dst)``; the symmetric kinds are unordered. Duplicates
    collapse, which matters because a chain may legitimately traverse the same directed edge
    twice (see the revisit discussion in :mod:`pathmgr.core.tracing`).
    """
    directed = set(chain.directed_edges())
    bidirected = {frozenset(pivot) for pivot in chain.pivots}
    copaths = {frozenset(pair) for pair in chain.copath_edges()}
    return directed, bidirected, copaths


class _Colours:
    """Maps colour specs to TikZ names, emitting \\definecolor for any ``#RRGGBB`` value.

    A raw hex spec cannot go straight into TikZ: ``#`` is a TeX macro-parameter character and
    trips ``Illegal parameter number``. Named colours pass through untouched.
    """

    def __init__(self) -> None:
        self._definitions: dict[str, str] = {}

    def name(self, spec: str) -> str:
        if not spec.startswith("#"):
            return spec
        hexpart = spec[1:].upper()
        label = f"pmC{hexpart}"
        self._definitions[label] = hexpart
        return label

    def declarations(self) -> list[str]:
        return [
            f"\\definecolor{{{label}}}{{HTML}}{{{hexpart}}}"
            for label, hexpart in sorted(self._definitions.items())
        ]


def _preamble(style: DiagramStyle, colours: "_Colours") -> list[str]:
    return [
        "\\begin{tikzpicture}[",
        f"  every node/.style={{font=\\{style.font_size}}},",
        f"  pmObserved/.style={{draw={style.node_colour}, fill={style.node_fill}, "
        f"{style.observed_shape}, minimum width={style.node_width}cm, "
        f"minimum height={style.node_height}cm, inner sep=1pt}},",
        f"  pmLatent/.style={{draw={style.node_colour}, fill={style.node_fill}, "
        f"{style.latent_shape}, minimum width={style.node_width}cm, "
        f"minimum height={style.node_height}cm, inner sep=1pt}},",
        f"  pmDirected/.style={{{style.arrow_tip_directed}, "
        f"line width={style.directed_width}pt}},",
        f"  pmBidirected/.style={{{style.arrow_tip_bidirected}, "
        f"line width={style.bidirected_width}pt}},",
        # NO arrow tips on a co-path -- that is the whole point
        f"  pmCopath/.style={{line width={style.copath_width}pt}},",
        "  pmLabel/.style={midway, fill=white, inner sep=1pt},",
        "]",
    ]


def to_tikz(
    model: Model,
    layout: Layout | None = None,
    style: DiagramStyle | None = None,
    highlight=None,
    caption_chain: bool = True,
    scale: float = 1.0,
) -> str:
    """A bare ``tikzpicture`` for ``model``.

    ``layout`` may be partial -- unplaced nodes are filled in automatically. ``highlight`` takes
    a :class:`pathmgr.Chain`: its edges are drawn emphasised and, if
    ``style.fade_unhighlighted``, everything else is greyed out. With ``caption_chain`` the
    chain's own path string is emitted beneath the diagram, so the figure states which term it
    is showing.
    """
    style = style or DiagramStyle()
    layout = (layout or Layout()).completed(model)
    if scale != 1.0:
        layout = layout.scaled(scale)

    if highlight is not None:
        hot_directed, hot_bidirected, hot_copaths = highlight_sets(highlight)
    else:
        hot_directed, hot_bidirected, hot_copaths = set(), set(), set()
    highlighting = highlight is not None

    colours = _Colours()
    body = _preamble(style, colours)
    lines: list[str] = []

    # -- nodes -------------------------------------------------------------------------
    for variable in model.variables:
        x, y = layout[variable.name]
        shape = "pmLatent" if variable.latent else "pmObserved"
        label = style.node_label(variable.name, variable.label)
        body.append(
            f"  \\node[{shape}] ({_node_id(variable.name)}) at ({x:.3f},{y:.3f}) {{${label}$}};"
        )

    # -- directed paths ----------------------------------------------------------------
    for edge in model.directed_edges:
        hot = (edge.src, edge.dst) in hot_directed
        options = _edge_options(
            "pmDirected", style.directed_colour, style, hot, highlighting, colours
        )
        label = style.edge_label((edge.src, edge.dst), edge.coeff)
        body.append(
            f"  \\draw[{options}] ({_node_id(edge.src)}) -- "
            f"{_label_node(label, style, hot, highlighting, colours)}({_node_id(edge.dst)});"
        )

    # -- bidirected covariances --------------------------------------------------------
    for edge in model.bidirected_edges:
        if edge.is_variance and not style.show_variances:
            continue
        hot = frozenset((edge.a, edge.b)) in hot_bidirected
        options = _edge_options(
            "pmBidirected", style.bidirected_colour, style, hot, highlighting, colours
        )
        label = style.edge_label((edge.a, edge.b), edge.value)
        node = _label_node(label, style, hot, highlighting, colours)
        if edge.is_variance:
            body.append(
                f"  \\draw[{options}] ({_node_id(edge.a)}) to[loop above, looseness=6, "
                f"in=120, out=60] {node}({_node_id(edge.a)});"
            )
        else:
            body.append(
                f"  \\draw[{options}] ({_node_id(edge.a)}) to[bend left="
                f"{style.bidirected_bend:.0f}] {node}({_node_id(edge.b)});"
            )

    # -- co-paths ----------------------------------------------------------------------
    # Several co-paths on one couple must stay separable, so successive ones on the same pair
    # bow by increasing amounts instead of being drawn on top of each other.
    seen_pairs: dict[frozenset, int] = {}
    for copath in model.copaths:
        pair = frozenset((copath.a, copath.b))
        seen = seen_pairs.get(pair, 0)
        seen_pairs[pair] = seen + 1
        hot = pair in hot_copaths
        options = _edge_options(
            "pmCopath", style.copath_colour, style, hot, highlighting, colours
        )
        label = style.edge_label((copath.a, copath.b), copath.coefficient)
        node = _label_node(label, style, hot, highlighting, colours)
        connector = "--" if seen == 0 else f"to[bend right={12 * seen}]"
        body.append(
            f"  \\draw[{options}] ({_node_id(copath.a)}) {connector} "
            f"{node}({_node_id(copath.b)});"
        )

    # -- caption -----------------------------------------------------------------------
    if highlighting and caption_chain:
        min_x, min_y, max_x, _ = layout.bounds()
        mid_x = (min_x + max_x) / 2
        path_tex = highlight.tex_path({v.name: v.label for v in model.variables if v.label})
        if path_tex:
            body.append(
                f"  \\node[align=center] at ({mid_x:.3f},{min_y - 1.15:.3f}) "
                f"{{${path_tex}$}};"
            )

    body.append("\\end{tikzpicture}")
    lines.extend(colours.declarations())
    lines.extend(body)
    return "\n".join(lines) + "\n"


def _edge_options(
    base: str,
    colour: str,
    style: DiagramStyle,
    hot: bool,
    highlighting: bool,
    colours: "_Colours",
) -> str:
    if hot:
        return (
            f"{base}, draw={colours.name(style.highlight_colour)}, "
            f"line width={_hot_width(base, style):.2f}pt"
        )
    if highlighting and style.fade_unhighlighted:
        return f"{base}, draw={colours.name(style.faded_colour)}"
    return f"{base}, draw={colours.name(colour)}"


def _hot_width(base: str, style: DiagramStyle) -> float:
    widths = {
        "pmDirected": style.directed_width,
        "pmBidirected": style.bidirected_width,
        "pmCopath": style.copath_width,
    }
    return widths[base] * style.highlight_scale


def _label_node(
    label: str, style: DiagramStyle, hot: bool, highlighting: bool, colours: "_Colours"
) -> str:
    if not label:
        return ""
    colour = ""
    if hot:
        colour = f", text={colours.name(style.highlight_colour)}"
    elif highlighting and style.fade_unhighlighted:
        colour = f", text={colours.name(style.faded_colour)}"
    return f"node[pmLabel{colour}] {{${label}$}} "


def to_standalone(
    model: Model,
    layout: Layout | None = None,
    style: DiagramStyle | None = None,
    highlight=None,
    document_class: str = "article",
    **kwargs,
) -> str:
    """A complete, compilable LaTeX file containing the diagram.

    Defaults to ``article`` because ``standalone.cls`` is not part of a plain TinyTeX install.
    Pass ``document_class="standalone"`` for a tightly cropped figure if you have that class.
    """
    picture = to_tikz(model, layout=layout, style=style, highlight=highlight, **kwargs)
    libraries = ",".join(TIKZ_LIBRARIES)
    if document_class == "standalone":
        header = ["\\documentclass[border=6pt]{standalone}"]
    else:
        # standalone.cls is often absent, so size the page to the drawing instead of leaving
        # the figure in the corner of a fixed sheet. Padding covers node boxes, edge labels,
        # variance loops and the optional chain caption, none of which are in `bounds()`.
        completed = (layout or Layout()).completed(model)
        min_x, min_y, max_x, max_y = completed.bounds()
        pad = 3.0
        width = max(max_x - min_x + 2 * pad, 6.0)
        height = max(max_y - min_y + 2 * pad, 6.0)
        header = [
            f"\\documentclass{{{document_class}}}",
            f"\\usepackage[margin=0cm, paperwidth={width:.1f}cm, "
            f"paperheight={height:.1f}cm]{{geometry}}",
            "\\pagestyle{empty}",
        ]
    return "\n".join(
        header
        + [
            "\\usepackage{tikz}",
            f"\\usetikzlibrary{{{libraries}}}",
            "\\usepackage{xcolor}",
            "\\usepackage{amsmath,amssymb}",
            "\\begin{document}",
            "\\centering",
            "\\vspace*{\\fill}",
            picture.rstrip(),
            "\\vspace*{\\fill}",
            "\\end{document}",
            "",
        ]
    )


def write_pdf(
    model: Model,
    path: str | Path,
    layout: Layout | None = None,
    style: DiagramStyle | None = None,
    highlight=None,
    engine: str = "pdflatex",
    **kwargs,
) -> Path:
    """Compile the diagram to a PDF at ``path``. Raises :class:`TikzCompileError` on failure.

    Calls ``pdflatex`` directly rather than ``latexmk``: on this cluster's compute nodes the
    system perl is incomplete and breaks ``latexmk``, and a single diagram needs no rerun logic.
    """
    path = Path(path)
    binary = shutil.which(engine)
    if binary is None:
        raise TikzCompileError(
            f"{engine!r} is not on PATH, so the diagram cannot be compiled. The TikZ source is "
            f"still available from to_tikz()/to_standalone() without any LaTeX install."
        )
    source = to_standalone(model, layout=layout, style=style, highlight=highlight, **kwargs)
    with tempfile.TemporaryDirectory() as tmp:
        tex = Path(tmp) / "diagram.tex"
        tex.write_text(source)
        result = subprocess.run(
            [binary, "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        produced = Path(tmp) / "diagram.pdf"
        if result.returncode != 0 or not produced.exists():
            log = (result.stdout or "") + (result.stderr or "")
            texlog = Path(tmp) / "diagram.log"
            if texlog.exists():
                log = texlog.read_text(errors="replace")
            # a TeX log is mostly one enormous line of package paths; the diagnostics are the
            # lines beginning "!", so show those and their context rather than a blind tail
            interesting: list[str] = []
            lines_ = log.splitlines()
            for i, line in enumerate(lines_):
                if line.startswith("!"):
                    interesting.extend(lines_[i : i + 6])
            detail = "\n".join(interesting[:40]) or "\n".join(lines_[-25:])
            raise TikzCompileError(f"{engine} failed:\n{detail}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(produced.read_bytes())
    return path
