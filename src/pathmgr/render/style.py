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

__all__ = ["DiagramStyle", "coefficient_label"]

#: LaTeX control sequences that take up no visual width when estimating a label's size
_MARKUP = ("\\left", "\\right", "\\tfrac", "\\frac", "\\text", "\\mathrm", "{", "}", "$", "\\,")


def _visible_length(label: str) -> int:
    """Roughly how many characters wide a LaTeX label prints as.

    A crude estimate on purpose: it only has to be close enough to keep an edge label off a node
    box. A control sequence like ``\\rho`` prints as one glyph, and braces print as nothing.
    """
    text = label
    for token in _MARKUP:
        text = text.replace(token, "")
    # each remaining backslash-word is one glyph
    parts = text.split("\\")
    length = len(parts[0])
    for part in parts[1:]:
        stripped = part.lstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        length += 1 + len(stripped)
    return max(1, length)


def coefficient_label(value: sp.Expr, omit_unit: bool = True) -> str:
    """A coefficient as LaTeX math *without* delimiters, or ``""`` to draw no label.

    ``rho_y`` becomes ``\\rho_{y}``, not ``rho_y``. A coefficient of exactly 1 renders as nothing
    by default, following Sunde ("from here on, we will omit the unit path coefficients"), which
    keeps a pedigree diagram readable.
    """
    if omit_unit and value == 1:
        return ""
    return sp.latex(value)


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
    rectangle_inset: float = 0.09
    #: padding inside an ellipse. Larger, because an ellipse has proportionally less usable area
    #: than its bounding box -- that is geometry, not extra padding.
    ellipse_inset: float = 0.16
    #: floor on node size, in cm, so a one-character label still gets a sane box
    node_min_width: float = 0.42
    node_min_height: float = 0.34
    #: raster-only: nominal cm per character, used to estimate a label's width when sizing a box
    #: before it is typeset. Only affects the raster back end; TikZ measures the real thing.
    raster_char_width: float = 0.115
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
        """LaTeX math for an edge, honouring any override for that edge."""
        if key in self.label_overrides:
            return self.label_overrides[key]
        reversed_key = (key[1], key[0])
        if reversed_key in self.label_overrides:
            return self.label_overrides[reversed_key]
        return coefficient_label(value, omit_unit=not self.show_unit_coefficients)

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
        text = _visible_length(label)
        inset = self.ellipse_inset if latent else self.rectangle_inset
        width = max(self.node_min_width, text * self.raster_char_width + 2 * inset)
        height = max(self.node_min_height, self.raster_line_height + 2 * inset)
        if latent:
            # an ellipse must be wider than its bounding text to contain it
            width *= 1.25
        return (width, height)

    def legend_entries(self) -> tuple[tuple[str, str], ...]:
        """``(edge kind, one-line meaning)`` pairs, for a figure that spells the rules out."""
        return (
            ("directed", "causal effect, traced backward then forward"),
            ("bidirected", "exogenous covariance; a chain uses exactly one"),
            ("copath", "covariance from matching; reaches the causes, no variance induced"),
        )
