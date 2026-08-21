"""Encoding a repeated coefficient in an edge's appearance instead of in a label on every edge.

`pair_offspring_2v` carries eight `1/2` labels and eight `1/4` labels, identical, all fighting for
space in the most congested band of the figure. No placement algorithm beats not drawing sixteen
redundant labels. So: when a coefficient appears on enough edges, drop its inline label everywhere
and give those edges a distinct appearance, with a legend to decode it.

This is the same idea `show_unit_coefficients` already applies to coefficients of 1, generalised --
Sunde's "from here on, we will omit the unit path coefficients" with the value stated once instead
of assumed.

Four things here are load-bearing rather than cosmetic.

**Two channels, not one.** Colour alone would fail this module's own stated principle: the co-path
distinction is over-invested in precisely so it "stays distinguishable in greyscale". A colour-only
coefficient encoding is strictly worse than today when printed -- every coded edge becomes an
identical black line *with its label now removed*. So every code carries a **dash pattern** as well
as a colour, and the dash is what survives greyscale, photocopying, and a colourblind reader.

**Solid is reserved.** Every coded pattern is non-solid, so a coded edge is distinguishable from an
ordinary unlabelled one, not merely from the other coded ones.

**Keyed on the VALUE, never the edge type.** The eight `1/4` self-loops in `pair_offspring_2v` are
equal only because `a^(0)_kk = 0` in the base population; the corresponding ones in
`reduced_pair_offspring` are `1/4 - a^(1)_kk/2` and differ per variant. Grouping by "self-loop"
would state one number for four different ones -- a wrong figure, not an ugly one. Directed and
bidirected are counted separately as well: a `1/2` path coefficient and a `1/2` covariance are
different claims.

**Assignment is stable, not merely deterministic.** Determinism is the standing rule. Stability is
the extra one this feature needs: adding an edge to a model must not reshuffle the other
coefficients' appearances, or every figure in a writeup churns on an unrelated change. Assignment
is therefore by a hash of the coefficient's canonical form, with deterministic probing on
collision, so an existing coefficient keeps its slot when a new one appears -- which a sorted order
would not guarantee.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import sympy as sp

from ..core.model import Model

__all__ = ["CodedCoefficient", "CoefficientCoding", "code_coefficients"]

#: Colourblind-safe qualitative hues (Okabe-Ito and Paul Tol), deliberately EXCLUDING anything near
#: the three meanings colour already carries in this package: the co-path red ``#B03A2E``, the
#: highlight blue ``#1F77B4``, and the faded grey ``#BBBBBB``. Coefficient identity is a fourth
#: meaning on an already-crowded channel, which is why it never travels alone.
PALETTE: tuple[str, ...] = (
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#999933",  # olive
    "#44AA99",  # teal
    "#56B4E9",  # sky blue -- last, being the nearest to the reserved highlight blue
)

#: Dash patterns, as (on, off) runs in pt. **None is solid**: an ordinary edge must stay
#: distinguishable from a coded one after the colour is gone.
DASHES: tuple[tuple[float, ...], ...] = (
    (3.0, 2.0),
    (1.0, 2.0),
    (5.0, 2.0),
    (3.0, 2.0, 1.0, 2.0),
    (6.0, 3.0),
    (1.0, 1.0),
)


@dataclass(frozen=True)
class CodedCoefficient:
    """One coefficient that has been lifted off the edges and into the legend."""

    value: sp.Expr
    #: ``"directed"`` or ``"bidirected"`` -- counted and coded separately, never merged
    kind: str
    colour: str
    dash: tuple[float, ...]
    #: the edges now carrying this appearance instead of a label
    edges: tuple[tuple[str, str], ...]

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, sp.srepr(self.value))

    def __str__(self) -> str:
        return f"{self.kind} {sp.sstr(self.value)} on {len(self.edges)} edges"


@dataclass
class CoefficientCoding:
    """What was coded, and the lookup the back ends draw from.

    ``coded`` is **the elided set** -- exposed rather than turned into prose, because the figures'
    captions are hand-written in the document's voice and carry more than the coefficient. The
    legend is derived from this, so legend and diagram cannot drift apart.
    """

    coded: tuple[CodedCoefficient, ...] = ()
    #: ``(kind, edge key) -> CodedCoefficient`` for the edges whose label is suppressed
    by_edge: dict[tuple[str, tuple[str, str]], CodedCoefficient] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.coded)

    def for_edge(self, kind: str, key: tuple[str, str]) -> CodedCoefficient | None:
        found = self.by_edge.get((kind, key))
        if found is not None:
            return found
        return self.by_edge.get((kind, (key[1], key[0])))

    def legend_entries(self) -> tuple[tuple[str, str, tuple[float, ...], str], ...]:
        """``(colour, dash, latex)`` per coded coefficient, derived from what was elided.

        Never written independently of :attr:`coded`. ``style.py`` records a real incident where a
        caption contradicted the diagram above it, and a legend that disagrees with the edges it
        decodes is the same failure with a worse consequence -- the reader cannot recover the model.
        """
        return tuple(
            (entry.kind, entry.colour, entry.dash, sp.latex(entry.value)) for entry in self.coded
        )


def _slot(kind: str, value: sp.Expr, taken: set[int]) -> int:
    """A palette slot for this coefficient: hashed, so existing ones keep their slot.

    Sorting the coefficients and assigning in order would be deterministic but *not* stable -- a
    new coefficient sorting early shifts every later one, and every figure in the document churns.
    Hashing the canonical form fixes each coefficient's slot independently of what else exists;
    probing forward on a collision moves only the newcomer.
    """
    digest = hashlib.sha256(f"{kind}|{sp.srepr(value)}".encode()).digest()
    start = int.from_bytes(digest[:8], "big") % len(PALETTE)
    for step in range(len(PALETTE)):
        slot = (start + step) % len(PALETTE)
        if slot not in taken:
            return slot
    return start  # more coefficients than slots: reuse rather than fail, and say so in the docs


def code_coefficients(
    model: Model,
    threshold: int,
    exempt: frozenset[tuple[str, str]] = frozenset(),
    omit_unit: bool = True,
) -> CoefficientCoding:
    """Decide which coefficients to lift off the edges, and what each should look like.

    ``threshold`` is the number of edges a coefficient must appear on. ``exempt`` names edges whose
    label the caller has overridden explicitly -- those are never elided, because an override is a
    deliberate statement about that one edge and silently removing it would override the override.

    ``omit_unit`` mirrors the style: a coefficient of 1 that is **already** not drawn must not be
    coded. Coding it would dash two dozen edges and add a legend entry to remove a label nobody was
    looking at -- pure cost. Only coefficients that would otherwise appear on the diagram qualify.

    Co-paths are left out entirely: their colour already means "co-path", there are few of them,
    and recolouring one would collide with the distinction the whole style module is built around.
    """
    census: dict[tuple[str, str], list[tuple[str, str]]] = {}
    values: dict[tuple[str, str], sp.Expr] = {}

    def note(kind: str, value: sp.Expr, key: tuple[str, str]) -> None:
        if key in exempt or (key[1], key[0]) in exempt:
            return
        if omit_unit and sp.sympify(value) == 1:
            return  # not drawn in the first place; see `omit_unit` above
        canonical = (kind, sp.srepr(sp.sympify(value)))
        census.setdefault(canonical, []).append(key)
        values.setdefault(canonical, sp.sympify(value))

    for edge in model.directed_edges:
        note("directed", edge.coeff, (edge.src, edge.dst))
    for edge in model.bidirected_edges:
        note("bidirected", edge.value, (edge.a, edge.b))

    eligible = [
        (canonical, edges) for canonical, edges in census.items() if len(edges) >= threshold
    ]
    # iterate in a canonical order so the RESULT is deterministic; the slot each one gets is
    # decided by the hash, so the order here does not influence appearance
    eligible.sort(key=lambda item: item[0])

    taken: set[int] = set()
    coded: list[CodedCoefficient] = []
    by_edge: dict[tuple[str, tuple[str, str]], CodedCoefficient] = {}
    for (kind, _canonical_value), edges in eligible:
        value = values[(kind, _canonical_value)]
        slot = _slot(kind, value, taken)
        taken.add(slot)
        entry = CodedCoefficient(
            value=value,
            kind=kind,
            colour=PALETTE[slot],
            dash=DASHES[slot],
            edges=tuple(edges),
        )
        coded.append(entry)
        for key in edges:
            by_edge[(kind, key)] = entry
    return CoefficientCoding(coded=tuple(coded), by_edge=by_edge)
