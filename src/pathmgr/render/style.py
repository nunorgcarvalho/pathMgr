"""Diagram conventions: shapes, edge styles, and how coefficients become labels.

Shared by the TikZ and raster back ends so the two cannot drift apart on the thing that
matters most -- **the three edge types must be unmistakable from one another**.

    directed    a -> b     straight, ONE arrowhead
    bidirected  a <-> b    CURVED, TWO arrowheads
    co-path     a -- b     straight, NO arrowheads, thicker, distinct colour

The co-path distinction is deliberately over-invested in. It is a different *kind* of thing --
covariance induced by matching, which propagates to the causes of the matched variables -- and a
reader who mistakes it for a covariance arrow will apply the wrong tracing rules and get a wrong
answer by hand. So it differs from a bidirected edge on three axes at once (arrowheads, curvature,
weight) and stays distinguishable in greyscale, where colour is gone but "no arrowheads and
thicker" survives. :func:`DiagramStyle.legend_entries` exists so a figure can say so outright.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from ..core.tracing import tex

__all__ = ["DiagramStyle", "coefficient_label", "text_width"]

#: LaTeX control sequences that occupy no horizontal space of their own
_ZERO_WIDTH = (r"\left", r"\right", r"\,", r"\!", r"\;", r"\ ", r"\quad", r"\qquad")
#: characters that typeset much narrower than a letter -- delimiters, operators, punctuation
_NARROW_CHARS = frozenset("[](){}+-=,.|/")


def _group(text: str, i: int) -> tuple[str, int]:
    """The braced group -- or single token -- starting at ``i``, and the index after it."""
    if i >= len(text):
        return "", i
    if text[i] == "{":
        level, j = 0, i
        while j < len(text):
            if text[j] == "{":
                level += 1
            elif text[j] == "}":
                level -= 1
                if level == 0:
                    return text[i + 1 : j], j + 1
            j += 1
        return text[i + 1 :], len(text)
    if text[i] == "\\":
        j = i + 1
        while j < len(text) and text[j].isalpha():
            j += 1
        return text[i:j], j
    return text[i], i + 1


def text_width(label: str, style: "DiagramStyle") -> float:
    """Estimated typeset width of a math label, in cm.

    A **rough** estimate on purpose -- it only has to be close enough to keep labels off each other
    and off the lines. TikZ sizes its own boxes from the real typeset label; this is what the
    collision scoring in both back ends has to work with, and a two-pass compile that reads back
    real node dimensions is the proper fix and a much larger job.

    It replaces a plain character count, which was wrong in a way that mattered. Measured against
    fifteen actually-compiled labels drawn from the writeup figures, the character count was
    **23.7% mean / 68.3% worst** error; this is **7.4% mean / 20.6% worst**. Two structural facts
    account for almost all of the gap, and both are the same trick:

    - **A superscript and a subscript on the same base stack vertically**, so they cost
      ``max(sup, sub)`` and not their sum. Without this, exactly the symbols this project uses
      everywhere (``z^{(m)}_{o1,1}``) came out 55-68% too *wide* -- so the label thought it needed
      far more room than it does, and placement fled space it could have used.
    - **A fraction is as wide as its wider part**, again not the sum.

    Scripts are then scaled by ``script_scale``, and delimiters get their own narrower width.
    """
    total, i, n = 0.0, 0, len(label)
    base_width: float | None = None
    sup = sub = 0.0

    def flush(carried: float) -> float:
        nonlocal base_width, sup, sub
        if base_width is not None:
            carried += base_width + max(sup, sub)
        base_width, sup, sub = None, 0.0, 0.0
        return carried

    while i < n:
        skipped = False
        for token in _ZERO_WIDTH:
            if label.startswith(token, i):
                i += len(token)
                skipped = True
                break
        if skipped:
            continue
        char = label[i]
        if label.startswith(r"\frac", i):
            total = flush(total)
            i += len(r"\frac")
            numerator, i = _group(label, i)
            denominator, i = _group(label, i)
            base_width = max(text_width(numerator, style), text_width(denominator, style))
            continue
        if char in "^_":
            content, i = _group(label, i + 1)
            scripted = text_width(content, style) * style.script_scale
            if char == "^":
                sup = max(sup, scripted)
            else:
                sub = max(sub, scripted)
            continue
        if char == "\\":
            total = flush(total)
            _token, i = _group(label, i)
            base_width = style.glyph_width
            continue
        if char in "{}$ ":
            i += 1
            continue
        total = flush(total)
        base_width = style.narrow_glyph_width if char in _NARROW_CHARS else style.glyph_width
        i += 1
    return flush(total)


def coefficient_label(
    value: sp.Expr, omit_unit: bool = True, latex_names: dict | None = None
) -> str:
    """A coefficient as LaTeX math *without* delimiters, or ``""`` to draw no label.

    ``rho_y`` becomes ``\\rho_{y}``, not ``rho_y``. A coefficient of exactly 1 renders as nothing
    by default, following Sunde ("from here on, we will omit the unit path coefficients"), which
    keeps a pedigree diagram readable.

    ``latex_names`` applies here as well as in captions, so a figure is internally consistent: a
    document that calls a sum ``\\VPo`` wants it called that on the co-path label *and* in the
    caption underneath, not one of each.
    """
    if omit_unit and value == 1:
        return ""
    return tex(value, latex_names)


@dataclass
class DiagramStyle:
    """Everything cosmetic, in one place, shared by both back ends."""

    # -- node shapes ------------------------------------------------------------------
    # Sized by their CONTENTS, not to a uniform footprint. A `minimum width/height` forces every
    # node to the same box whatever its label, which is where padding (and crowding) comes from.
    # These are the paddings around the typeset label; the minimums are a floor for a very short
    # label like a single letter, not a target.
    observed_shape: str = "rectangle"
    latent_shape: str = "ellipse"
    #: padding around the label inside a rectangle, in cm
    rectangle_inset: float = 0.06
    #: horizontal padding inside an ellipse. Split from the vertical because they are not equally
    #: expensive: these diagrams are wide and short of horizontal room, the labels that crowd worst
    #: are wide ones (``z^{(m)}_{o1,1}``), and an ellipse already wastes more width than a rectangle
    #: for the same text. Vertical padding is cheap here and stays generous.
    ellipse_xsep: float = 0.07
    #: vertical padding inside an ellipse
    ellipse_ysep: float = 0.13
    #: floor on node size, in cm, so a one-character label still gets a sane box. MEASURED: only
    #: ``node_min_width`` ever binds, and only for a single-glyph label like ``V`` (0.405 -> 0.420).
    #: ``node_min_height`` binds for nothing -- height comes out as
    #: ``raster_line_height + 2 * inset``, which is 0.52 for a rectangle and 0.60 for an ellipse,
    #: both already above it. Lowering it would change no figure.
    node_min_width: float = 0.38
    node_min_height: float = 0.34
    #: estimated width in cm of one base glyph in a math label. Fitted against fifteen compiled
    #: labels from the writeup figures; see :func:`text_width` for the residual error.
    glyph_width: float = 0.225
    #: width of a delimiter or operator, which typesets much narrower than a letter
    narrow_glyph_width: float = 0.097
    #: how much narrower a sub/superscript is than a base glyph
    script_scale: float = 0.6
    raster_line_height: float = 0.34

    # -- edge appearance --------------------------------------------------------------
    directed_width: float = 0.7
    bidirected_width: float = 0.7
    #: thicker than the arrows: the greyscale-safe half of the co-path distinction
    copath_width: float = 1.6
    #: how far a bidirected edge bows, in degrees of TikZ bend / matplotlib rad
    bidirected_bend: float = 30.0

    # -- colours (also used by the raster back end) ------------------------------------
    node_colour: str = "black"
    node_fill: str = "white"
    directed_colour: str = "black"
    bidirected_colour: str = "black"
    #: the colourful half of the co-path distinction
    copath_colour: str = "#B03A2E"
    highlight_colour: str = "#1F77B4"
    faded_colour: str = "#BBBBBB"

    # -- labels -----------------------------------------------------------------------
    #: draw a label for a coefficient of exactly 1
    show_unit_coefficients: bool = False
    #: draw bidirected self-edges (variances) as short self-loops
    show_variances: bool = True
    #: per-edge label overrides, keyed by ``(src, dst)`` for directed / ``(a, b)`` otherwise
    label_overrides: dict[tuple[str, str], str] = field(default_factory=dict)
    #: per-variable label overrides, taking precedence over ``Variable.label``
    node_label_overrides: dict[str, str] = field(default_factory=dict)
    #: **where** a label goes, as ``(a, b) -> (position, offset)``, bypassing the search for that
    #: one label. The escape hatch: automation will never be perfect on the hardest figures, and the
    #: node coordinates in these diagrams are hand-authored, so a user who can place a node but not
    #: its label has no recourse. An overridden label is still an obstacle, so the others avoid it.
    label_placement_overrides: dict[tuple[str, str], tuple[float, float]] = field(
        default_factory=dict
    )
    #: where a variance self-loop goes, as ``node -> (direction in degrees, looseness)``
    loop_overrides: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: edges whose label is not to be drawn at all. Distinct from an empty ``label_overrides``
    #: entry: this removes the label from placement entirely rather than drawing an empty box, and
    #: it reads as intent at the call site.
    suppressed_labels: frozenset[tuple[str, str]] = frozenset()
    #: render a symbol or subexpression under the name the *document* uses for it, e.g.
    #: ``{V_A0 + V_E: r"\VPo"}``. The analogue of ``node_label_overrides``, keyed by expression
    #: rather than by node, and the channel captions use for symbols the surrounding prose has
    #: already named. Composite keys work, not only plain symbols -- see
    #: :func:`pathmgr.core.tracing.tex`.
    latex_names: dict = field(default_factory=dict)
    font_size: str = "small"

    # -- arrow tips --------------------------------------------------------------------
    # `arrows.meta` Stealth tips by default: TikZ's built-in `->` draws a hairline tip that is
    # hard to see once a diagram is more than a few nodes wide. popstatgenwriteups' config.sty
    # loads arrows.meta, and `to_standalone` adds it automatically when these tips need it.
    # Use `DiagramStyle.portable()` for a snippet going into a document that may not load it.
    arrow_tip_directed: str = "-{Stealth[length=2mm]}"
    arrow_tip_bidirected: str = "{Stealth[length=2mm]}-{Stealth[length=2mm]}"
    #: raster-only: arrowhead size in points. Must stay legible on a large figure, so it is
    #: deliberately larger than matplotlib's default of 10.
    arrow_head_size: float = 20.0
    #: extra gap in cm between a node's boundary and where an edge starts/ends. The boundary
    #: itself is computed per node from its actual size -- a constant clearance under-shortens
    #: for a wide ellipse and over-shortens for a small node.
    node_clearance: float = 0.06

    # -- label placement ---------------------------------------------------------------
    #: fractions along an edge to try when the midpoint label would collide
    label_positions: tuple[float, ...] = (0.5, 0.38, 0.62, 0.28, 0.72)
    #: perpendicular offsets in cm to try, in order. 0 first, so simple diagrams are unchanged.
    label_offsets: tuple[float, ...] = (0.0, 0.22, -0.22, 0.4, -0.4)
    #: assumed label box size in cm, for collision scoring
    label_pad: float = 0.1
    #: set False to go back to plain midpoint placement
    avoid_label_collisions: bool = True
    #: allow a label that cannot be placed legibly near its edge to move far and be connected back
    #: by a hairline. Standard cartographic practice: it turns an unresolvable collision into a
    #: legible one. Only fires when moving far is worth ``LEADER_PENALTY``, so uncrowded diagrams
    #: never grow one.
    leader_lines: bool = True
    #: perpendicular offsets in cm reachable only with a leader
    leader_offsets: tuple[float, ...] = (0.75, 1.1, 1.5)
    #: hairline width for a leader, in pt
    leader_width: float = 0.3
    leader_colour: str = "#777777"

    # -- coefficient coding -------------------------------------------------------------
    #: lift a coefficient that appears on many edges off those edges and into the legend, coding it
    #: as a colour + dash pattern instead. **Off by default**: it changes a dense figure by design,
    #: and doing that silently to every existing figure is not acceptable.
    code_repeated_coefficients: bool = False
    #: how many edges a coefficient must appear on before it is worth coding
    coefficient_code_threshold: int = 5
    #: coding is additionally suppressed while a chain is highlighted, because a highlighted figure
    #: exists to let a reader multiply along the chain edge by edge -- removing the factors is
    #: exactly wrong there. Set True to override for a figure that wants both.
    code_with_highlight: bool = False

    # -- edge routing ------------------------------------------------------------------
    #: bends (TikZ degrees) to try for an edge that would otherwise cross a third node.
    #: 0 first, so an edge with a clear path is never touched and clean figures do not change.
    edge_bends: tuple[float, ...] = (
        0.0, 10.0, -10.0, 18.0, -18.0, 26.0, -26.0, 35.0, -35.0, 45.0, -45.0, 55.0, -55.0
    )
    #: how much clearance a routed edge should leave around an intervening node, in cm. A path
    #: grazing an ellipse reads as a doubled node border, so this is deliberately not zero.
    edge_clearance: float = 0.12
    #: set False to draw every edge straight, crossings and all
    route_edges_around_nodes: bool = True

    # -- highlighting -----------------------------------------------------------------
    #: multiply edge width by this for a highlighted edge
    highlight_scale: float = 2.6
    #: draw non-highlighted edges in ``faded_colour`` when a chain is highlighted
    fade_unhighlighted: bool = True

    # -- helpers ----------------------------------------------------------------------
    def edge_label(self, key: tuple[str, str], value: sp.Expr) -> str:
        """LaTeX math for an edge, honouring any override or suppression for that edge.

        Suppression is checked here rather than in each back end so it cannot be honoured by the
        placement pass and then quietly ignored by whichever renderer forgot -- the failure the
        moved-loop regression was made of.
        """
        if self.suppresses_label(key):
            return ""
        if key in self.label_overrides:
            return self.label_overrides[key]
        reversed_key = (key[1], key[0])
        if reversed_key in self.label_overrides:
            return self.label_overrides[reversed_key]
        return coefficient_label(
            value, omit_unit=not self.show_unit_coefficients, latex_names=self.latex_names
        )

    def _either_way(self, mapping, key):
        """Look ``key`` up in ``mapping`` either way round.

        Matches the convention ``label_overrides`` already uses: a caller should not have to think
        about which end of an edge they wrote first, least of all for a symmetric edge.
        """
        if key in mapping:
            return mapping[key]
        return mapping.get((key[1], key[0]))

    def placement_override(self, key: tuple[str, str]):
        """``(position, offset)`` for this edge's label, or ``None`` to let the search decide."""
        return self._either_way(self.label_placement_overrides, key)

    def loop_override(self, node: str):
        """``(direction, looseness)`` for this node's variance loop, or ``None``."""
        return self.loop_overrides.get(node)

    def suppresses_label(self, key: tuple[str, str]) -> bool:
        """True if this edge's label should not be drawn at all."""
        return key in self.suppressed_labels or (key[1], key[0]) in self.suppressed_labels

    def coefficient_coding(self, model, highlighting: bool = False):
        """The coefficient coding for ``model`` under this style, or an empty one.

        Edges with an explicit ``label_overrides`` entry are exempt: an override is a deliberate
        statement about that one edge, and silently eliding it would override the override.
        """
        from .coding import CoefficientCoding, code_coefficients

        if not self.code_repeated_coefficients:
            return CoefficientCoding()
        if highlighting and not self.code_with_highlight:
            return CoefficientCoding()
        exempt = frozenset(self.label_overrides)
        return code_coefficients(
            model,
            threshold=self.coefficient_code_threshold,
            exempt=exempt,
            omit_unit=not self.show_unit_coefficients,
        )

    def caption_options(self) -> dict:
        """The caption's share of the style, as plain kwargs for ``Chain.tex_caption``.

        The caption is the one part of a figure that renders arbitrary symbolic expressions, so it
        is the part most likely to need adjusting -- and until this existed it was the only label
        text that did not go through the style at all. That produced figures whose caption
        contradicted the diagram above it: with ``show_unit_coefficients=True`` the diagram drew
        every factor of 1 and the caption dropped them, on the very figure whose job is to let a
        reader check the product edge by edge.

        Returned as kwargs rather than passed as a style object because :mod:`pathmgr.core` may not
        import :mod:`pathmgr.render`.
        """
        return {
            "omit_unit": not self.show_unit_coefficients,
            "latex_names": self.latex_names or None,
        }

    def copath_label(self, copath) -> str:
        """LaTeX math for a co-path's label, bracketed when it is a **correlation**.

        The bracket is not decoration. A co-path carries one of two different quantities -- a raw
        ``mu``, where ``Cov = mu*Var[a]*Var[b]``, or the correlation it induces -- and they differ
        by a factor of the variances. A reader looking at ``\\rho_y`` on an edge cannot otherwise
        tell which they are being shown, and the two are numerically equal only when the
        endpoints have unit variance. ``[\\rho_y]`` reads "on the correlation scale".

        Overrides still win, as everywhere else in this class.
        """
        key = (copath.a, copath.b)
        if self.suppresses_label(key):
            return ""
        if key in self.label_overrides or (key[1], key[0]) in self.label_overrides:
            return self.edge_label(key, copath.declared)
        if not copath.is_standardized:
            return self.edge_label(key, copath.coefficient)
        inner = coefficient_label(
            copath.correlation, omit_unit=False, latex_names=self.latex_names
        )
        return rf"\left[{inner}\right]"

    def node_label(self, name: str, variable_label: str | None) -> str:
        """Math-mode text for a node: an override, else ``Variable.label``, else the name.

        A ``Variable.label`` of ``"$g_i$"`` is unwrapped, since both back ends supply their own
        math delimiters.
        """
        if name in self.node_label_overrides:
            text = self.node_label_overrides[name]
        elif variable_label:
            text = variable_label
        else:
            return sp.latex(sp.Symbol(name))
        text = text.strip()
        if text.startswith("$") and text.endswith("$") and len(text) > 1:
            return text[1:-1]
        return text

    @classmethod
    def portable(cls, **kwargs) -> "DiagramStyle":
        """A style using only TikZ's built-in arrow tips, needing no ``arrows.meta``.

        Slightly less legible, but a snippet emitted with it pastes into any document that loads
        plain ``tikz``.
        """
        return cls(arrow_tip_directed="->", arrow_tip_bidirected="<->", **kwargs)

    @property
    def needs_arrows_meta(self) -> bool:
        """True if the arrow tips require the ``arrows.meta`` library."""
        return any(
            tip not in ("->", "<->", "-", "<-")
            for tip in (self.arrow_tip_directed, self.arrow_tip_bidirected)
        )

    def draws_variance(self, highlighted: bool) -> bool:
        """Whether to draw a variance self-loop.

        **A style flag governs what CONTEXT is drawn; it must never suppress an edge that is part
        of the highlighted chain.** `show_variances=False` is for decluttering the surroundings,
        and hiding a highlighted edge makes the figure contradict its own caption -- in the
        allele-level chain the two `z <-> z` variances carry the `1/2 * 1/2` that produces the
        whole `/4` in `beta^2 rho_y / (4 V_P)`, so omitting them leaves a reader tracing the chain
        by hand off by a factor of four with nothing in the figure to explain it.

        Any future filter must follow the same rule; there is a test that enumerates them.
        """
        return self.show_variances or highlighted

    def node_size(self, label: str, latent: bool) -> tuple[float, float]:
        """Estimated ``(width, height)`` in cm for a node holding ``label``.

        Used by the raster back end, and by label-collision scoring in both. TikZ sizes its own
        boxes from the real typeset label; this only has to be close enough to keep labels off
        them.
        """
        text = text_width(label, self)
        x_inset = self.ellipse_xsep if latent else self.rectangle_inset
        y_inset = self.ellipse_ysep if latent else self.rectangle_inset
        width = max(self.node_min_width, text + 2 * x_inset)
        height = max(self.node_min_height, self.raster_line_height + 2 * y_inset)
        if latent:
            # an ellipse has to be wider than the text box it contains: circumscribing a w-by-h box
            # needs sqrt(2)*w in principle, and TikZ lands nearer 1.25 in practice for these sizes.
            width *= 1.25
        return (width, height)

    def legend_entries(self) -> tuple[tuple[str, str], ...]:
        """``(edge kind, one-line meaning)`` pairs, for a figure that spells the rules out."""
        return (
            ("directed", "causal effect, traced backward then forward"),
            ("bidirected", "exogenous covariance; a chain uses exactly one"),
            ("copath", "covariance from matching; reaches the causes, no variance induced"),
        )
