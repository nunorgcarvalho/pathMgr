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
    observed_shape: str = "rectangle"
    latent_shape: str = "ellipse"
    node_width: float = 0.85
    node_height: float = 0.6

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
    # TikZ's built-in tips, so the output needs no `arrows.meta` (absent from TinyTeX).
    # Set these to e.g. "-{Latex[length=1.8mm]}" if the host document loads arrows.meta.
    arrow_tip_directed: str = "->"
    arrow_tip_bidirected: str = "<->"
    #: raster-only: arrowhead size in points. Must stay legible on a large figure, so it is
    #: deliberately larger than matplotlib's default of 10.
    arrow_head_size: float = 20.0
    #: raster-only: gap in points between a node's edge and where an arrow starts/ends
    node_clearance: float = 15.0

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

    def legend_entries(self) -> tuple[tuple[str, str], ...]:
        """``(edge kind, one-line meaning)`` pairs, for a figure that spells the rules out."""
        return (
            ("directed", "causal effect, traced backward then forward"),
            ("bidirected", "exogenous covariance; a chain uses exactly one"),
            ("copath", "covariance from matching; reaches the causes, no variance induced"),
        )
