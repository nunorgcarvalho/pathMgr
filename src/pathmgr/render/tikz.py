"""TikZ export of a path diagram, for dropping straight into a LaTeX writeup.

Two forms: :func:`to_tikz` emits a bare ``tikzpicture`` to paste into an existing document, and
:func:`to_standalone` wraps it in a minimal compilable file. :func:`write_pdf` compiles one.

Conventions (see :mod:`pathmgr.render.style` for why the co-path is over-differentiated):

    observed variable    rectangle
    latent variable      ellipse
    directed  a -> b     straight, one arrowhead
    bidirected a <-> b   curved, two arrowheads; a self-edge is a variance loop
    co-path   a -- b     straight, NO arrowheads, thicker, distinct colour

Required packages: ``tikz`` with ``shapes.geometric`` (for the latent ellipse) and ``arrows.meta``
(for the default Stealth arrow tips), plus ``xcolor``. ``popstatgenwriteups``' ``config.sty`` loads
all of these. For a snippet going into a document that may not load ``arrows.meta``, use
:meth:`pathmgr.render.DiagramStyle.portable`, which falls back to TikZ's built-in ``->``/``<->``
tips; :func:`to_standalone` emits only the libraries the chosen style actually needs.

``standalone.cls`` is genuinely absent from a plain TinyTeX install and cannot be added without a
TeX Live infrastructure update, so :func:`to_standalone` defaults to ``article`` with the page
sized to the drawing. Pass ``document_class="standalone"`` if you have that class.

The emitted picture declares its own styles and depends on nothing else in the host document.

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
from .placement import LOOP_HALF_ANGLE, labelled_edges, place_labels, route_edges
from .style import DiagramStyle

__all__ = [
    "ARROW_LIBRARY",
    "TIKZ_LIBRARIES",
    "TikzCompileError",
    "required_libraries",
    "highlight_sets",
    "to_standalone",
    "to_tikz",
    "write_pdf",
]

#: libraries always needed: the latent ellipse
TIKZ_LIBRARIES = ("shapes.geometric",)
#: additionally needed by the default Stealth arrow tips
ARROW_LIBRARY = "arrows.meta"


def required_libraries(style: DiagramStyle) -> tuple[str, ...]:
    """The TikZ libraries a picture drawn with ``style`` needs."""
    if style.needs_arrows_meta:
        return TIKZ_LIBRARIES + (ARROW_LIBRARY,)
    return TIKZ_LIBRARIES


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


def _preamble(style: DiagramStyle, colours: "_Colours", leaders: bool = False) -> list[str]:
    return [
        "\\begin{tikzpicture}[",
        f"  every node/.style={{font=\\{style.font_size}}},",
        # sized by their CONTENTS: `inner sep` is the padding, the minimums only a floor for a
        # one-character label. A uniform `minimum width/height` is what makes a diagram crowded.
        f"  pmObserved/.style={{draw={style.node_colour}, fill={style.node_fill}, "
        f"{style.observed_shape}, inner sep={style.rectangle_inset}cm, "
        f"minimum width={style.node_min_width}cm, minimum height={style.node_min_height}cm}},",
        f"  pmLatent/.style={{draw={style.node_colour}, fill={style.node_fill}, "
        f"{style.latent_shape}, inner xsep={style.ellipse_xsep}cm, "
        f"inner ysep={style.ellipse_ysep}cm, "
        f"minimum width={style.node_min_width}cm, minimum height={style.node_min_height}cm}},",
        f"  pmDirected/.style={{{style.arrow_tip_directed}, "
        f"line width={style.directed_width}pt}},",
        f"  pmBidirected/.style={{{style.arrow_tip_bidirected}, "
        f"line width={style.bidirected_width}pt}},",
        # NO arrow tips on a co-path -- that is the whole point
        f"  pmCopath/.style={{line width={style.copath_width}pt}},",
        "  pmLabel/.style={midway, fill=white, inner sep=1pt},",
        *(
            [
                f"  pmLeader/.style={{line width={style.leader_width}pt, "
                f"draw={colours.name(style.leader_colour)}}},"
            ]
            if leaders
            else []
        ),
        "]",
    ]


def to_tikz(
    model: Model,
    layout: Layout | None = None,
    style: DiagramStyle | None = None,
    highlight=None,
    caption_chain: bool = True,
    caption: str | None = None,
    caption_name: str | None = None,
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
    lines: list[str] = []
    placements = place_labels(model, layout, style, labelled_edges(model, style))
    leaders = sorted(
        (key, placement)
        for key, placement in placements.items()
        if placement.leader_to is not None
    )
    body = _preamble(style, colours, leaders=bool(leaders))
    emitted_labels: set[str] = set()
    # bend only the edges whose straight path would run through a third node; everything else is
    # left exactly as it was, so a figure with no crossings is byte-identical to before routing
    bends = route_edges(model, layout, style) if style.route_edges_around_nodes else {}

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
        node = _label_node(
            label, style, hot, highlighting, colours,
            placements.get((edge.src, edge.dst)), _direction(layout, edge.src, edge.dst),
            name=_leader_name(leaders, (edge.src, edge.dst), emitted_labels),
        )
        connector = _bend_connector(bends.get((edge.src, edge.dst), 0.0))
        body.append(
            f"  \\draw[{options}] ({_node_id(edge.src)}) {connector} "
            f"{node}({_node_id(edge.dst)});"
        )

    # -- bidirected covariances --------------------------------------------------------
    for edge in model.bidirected_edges:
        hot = frozenset((edge.a, edge.b)) in hot_bidirected
        # a style flag governs CONTEXT; it never suppresses a highlighted edge
        if edge.is_variance and not style.draws_variance(hot):
            continue
        options = _edge_options(
            "pmBidirected", style.bidirected_colour, style, hot, highlighting, colours
        )
        label = style.edge_label((edge.a, edge.b), edge.value)
        loop = placements.get((edge.a, edge.a)) if edge.is_variance else None
        key = (edge.a, edge.a) if edge.is_variance else (edge.a, edge.b)
        node = _label_node(
            label, style, hot, highlighting, colours,
            loop if edge.is_variance else placements.get((edge.a, edge.b)),
            None if edge.is_variance else _direction(layout, edge.a, edge.b),
            name=_leader_name(leaders, key, emitted_labels),
        )
        if edge.is_variance:
            body.append(
                f"  \\draw[{options}] ({_node_id(edge.a)}) {_loop_connector(loop)} "
                f"{node}({_node_id(edge.a)});"
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
        label = style.copath_label(copath)
        node = _label_node(
            label, style, hot, highlighting, colours,
            placements.get((copath.a, copath.b)), _direction(layout, copath.a, copath.b),
            name=_leader_name(leaders, (copath.a, copath.b), emitted_labels),
        )
        if seen == 0:
            connector = _bend_connector(bends.get((copath.a, copath.b), 0.0))
        else:
            connector = f"to[bend right={12 * seen}]"
        body.append(
            f"  \\draw[{options}] ({_node_id(copath.a)}) {connector} "
            f"{node}({_node_id(copath.b)});"
        )

    # -- leader lines ------------------------------------------------------------------
    # Drawn last so they sit over the edges but under nothing; TikZ clips each at the named
    # label node's border, which is why the label had to be named rather than positioned twice.
    for index, (_key, placement) in enumerate(leaders):
        name = f"pmLbl{index}"
        if name not in emitted_labels:
            continue  # its edge was not drawn (e.g. a variance loop hidden while highlighting)
        x, y = placement.leader_to
        body.append(f"  \\draw[pmLeader] ({name}) -- ({x:.3f},{y:.3f});")

    # -- caption -----------------------------------------------------------------------
    if highlighting and caption_chain:
        min_x, min_y, max_x, _ = layout.bounds()
        mid_x = (min_x + max_x) / 2
        labels = {v.name: v.label for v in model.variables if v.label}
        if caption is None:
            name = None
            if caption_name:
                name = caption_name
            caption = highlight.tex_caption(
                labels, name=name, **style.caption_options()
            )
        if caption:
            # A caption may be two lines -- the Wright chain, then the product it contributes.
            # `\\` is illegal INSIDE math mode, so each line gets its own $...$ and the break
            # sits between them; align=center then stacks them.
            stacked = "\\\\".join(f"${line}$" for line in caption.split("\\\\") if line)
            body.append(
                f"  \\node[align=center] at ({mid_x:.3f},{min_y - 1.35:.3f}) {{{stacked}}};"
            )

    body.append("\\end{tikzpicture}")
    lines.extend(colours.declarations())
    lines.extend(body)
    return "\n".join(lines) + "\n"


def _bend_connector(bend: float) -> str:
    """``--`` for a straight edge, ``to[bend ...]`` for a routed one.

    Straight is emitted as plain ``--`` rather than ``to[bend left=0]`` so that an unrouted figure
    is byte-identical to what it was before edge routing existed.
    """
    if not bend:
        return "--"
    side = "left" if bend > 0 else "right"
    return f"to[bend {side}={abs(bend):.0f}]"


def _direction(layout: Layout, a: str, b: str) -> tuple[float, float]:
    (ax, ay), (bx, by) = layout[a], layout[b]
    return (bx - ax, by - ay)


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


def _loop_connector(placement) -> str:
    """The ``to[...]`` for a variance self-loop.

    With no placement the loop had no reason to move, and this emits the exact string the back end
    has always emitted -- which is what keeps an uncrowded figure byte-identical. Only a loop that
    actually collided gets the general form, whose ``out``/``in`` reproduce ``loop above`` when the
    direction is 90 degrees.
    """
    if placement is None or placement.loop_direction is None:
        return "to[loop above, looseness=6, in=120, out=60]"
    direction = placement.loop_direction
    looseness = placement.loop_looseness
    out_angle = (direction - LOOP_HALF_ANGLE) % 360
    in_angle = (direction + LOOP_HALF_ANGLE) % 360
    return (
        f"to[loop, looseness={looseness:g}, in={in_angle:.0f}, out={out_angle:.0f}]"
    )


def _leader_name(leaders, key, emitted: set[str] | None = None) -> str | None:
    """A stable TikZ node name for a label that a leader has to reach, else ``None``.

    Records the name in ``emitted`` so the leader pass can tell which nodes really exist: an edge
    that is skipped (a variance loop hidden by ``draws_variance`` while a chain is highlighted)
    never creates its node, and drawing to it is a hard TeX error, not a cosmetic one.
    """
    for index, (leader_key, _placement) in enumerate(leaders):
        if leader_key == key:
            name = f"pmLbl{index}"
            if emitted is not None:
                emitted.add(name)
            return name
    return None


def _label_node(
    label: str,
    style: DiagramStyle,
    hot: bool,
    highlighting: bool,
    colours: "_Colours",
    placement=None,
    direction: tuple[float, float] | None = None,
    name: str | None = None,
) -> str:
    """A TikZ ``node`` for an edge label, honouring the collision-avoiding placement.

    ``pos=`` moves it along the edge and the shifts move it perpendicular; both come from
    :func:`pathmgr.render.placement.place_labels`, which both back ends share so they agree.
    """
    if not label:
        return ""
    options = ["pmLabel"]
    if name is not None:
        # naming the node lets TikZ draw the leader to it and clip at its border for us
        options.append(f"name={name}")
    if hot:
        options.append(f"text={colours.name(style.highlight_colour)}")
    elif highlighting and style.fade_unhighlighted:
        options.append(f"text={colours.name(style.faded_colour)}")
    if placement is not None:
        if abs(placement.position - 0.5) > 1e-9:
            options.append(f"pos={placement.position:.3f}")
        if abs(placement.offset) > 1e-9 and direction is not None:
            dx, dy = direction
            length = (dx * dx + dy * dy) ** 0.5 or 1.0
            options.append(f"xshift={-placement.offset * dy / length:.3f}cm")
            options.append(f"yshift={placement.offset * dx / length:.3f}cm")
    return "node[" + ", ".join(options) + "] {$" + label + "$} "


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
    style = style or DiagramStyle()
    picture = to_tikz(model, layout=layout, style=style, highlight=highlight, **kwargs)
    libraries = ", ".join(required_libraries(style))
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
        # A PDF appearing is NOT proof of success: with -interaction=nonstopmode pdflatex
        # writes one anyway after a hard error. Count the "!" diagnostics in the log instead.
        texlog_path = Path(tmp) / "diagram.log"
        hard_errors = 0
        if texlog_path.exists():
            hard_errors = sum(
                1 for line in texlog_path.read_text(errors="replace").splitlines()
                if line.startswith("!")
            )
        if result.returncode != 0 or not produced.exists() or hard_errors:
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
            raise TikzCompileError(
                f"{engine} reported {hard_errors} hard error(s):\n{detail}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(produced.read_bytes())
    return path
