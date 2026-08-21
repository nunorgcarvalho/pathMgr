"""Where to put edge labels, and where an edge should stop at a node's boundary.

Shared by both back ends so a diagram is laid out the same way whichever one draws it.

Two jobs:

- :func:`boundary_point` -- where an edge meets a node's outline, computed from that node's
  *actual* size. A constant clearance under-shortens for a wide ellipse and over-shortens for a
  small node, which is how arrowheads end up hidden underneath boxes.
- :func:`place_labels` -- a greedy, **deterministic** pass that nudges a coefficient label off
  whatever it would otherwise sit on. The exact midpoint is always the first candidate, so a
  diagram with no collisions is byte-identical to what it was before this existed. Determinism is
  not a nicety: identical model in, identical TikZ out, or figure diffs in a writeup become
  unreadable.

Resisting a global optimiser is deliberate. Greedy candidate-and-score is enough for diagrams of
this size, and it is predictable, which matters more here than optimality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.model import Model
from .layout import Layout
from .style import DiagramStyle

__all__ = [
    "EdgeCrossing",
    "EdgePath",
    "LabelPlacement",
    "Rect",
    "boundary_point",
    "edge_node_crossings",
    "edge_paths",
    "label_rect",
    "labelled_edges",
    "loop_label_point",
    "loop_path",
    "place_labels",
]


@dataclass(frozen=True)
class Rect:
    """An axis-aligned box in diagram coordinates, used only for overlap scoring."""

    x: float
    y: float
    width: float
    height: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.x - self.width / 2,
            self.y - self.height / 2,
            self.x + self.width / 2,
            self.y + self.height / 2,
        )

    def overlap(self, other: "Rect") -> float:
        """Area of intersection with ``other``; 0 if they do not touch."""
        ax0, ay0, ax1, ay1 = self.bounds
        bx0, by0, bx1, by1 = other.bounds
        dx = min(ax1, bx1) - max(ax0, bx0)
        dy = min(ay1, by1) - max(ay0, by0)
        return dx * dy if dx > 0 and dy > 0 else 0.0


@dataclass(frozen=True)
class LabelPlacement:
    """Where one edge's label goes.

    ``position`` is the fraction along the edge (TikZ ``pos=``), ``offset`` the perpendicular
    displacement in cm, and ``point`` the resulting coordinate for back ends that place directly.

    For a **variance self-loop** the label is not positioned along a chord, so ``loop_direction``
    (degrees) and ``loop_looseness`` carry the choice instead and the back ends must draw the loop
    there. A self-loop only appears in the placements dict **at all** when its default position
    collided: absent means "draw it the way you always have", which is what keeps an uncrowded
    figure byte-identical.
    """

    position: float
    offset: float
    point: tuple[float, float]
    loop_direction: float | None = None
    loop_looseness: float | None = None


@dataclass(frozen=True)
class EdgeCrossing:
    """A straight edge whose path passes through a third node's box."""

    source: str
    target: str
    through: str
    kind: str = "directed"

    def __str__(self) -> str:
        return f"{self.source} -> {self.target}  crosses  {self.through}"


def _bezier_point(start, end, bend: float, fraction: float):
    """A point on the edge's path. ``bend`` of 0 is the straight segment.

    A bent edge is drawn by TikZ's ``to[bend]`` and matplotlib's ``arc3``; both are close enough to
    a quadratic Bezier through a control point displaced perpendicular to the chord that sampling
    that Bezier is a fair test of what will be drawn.
    """
    x0, y0 = start
    x1, y1 = end
    if bend == 0.0:
        return (x0 + (x1 - x0) * fraction, y0 + (y1 - y0) * fraction)
    rad = bend / 100.0
    mid = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    control = (mid[0] - rad * (y1 - y0), mid[1] + rad * (x1 - x0))
    inverse = 1 - fraction
    return (
        inverse**2 * x0 + 2 * inverse * fraction * control[0] + fraction**2 * x1,
        inverse**2 * y0 + 2 * inverse * fraction * control[1] + fraction**2 * y1,
    )


def _path_hits_rect(start, end, rect: Rect, bend: float = 0.0, samples: int = 80) -> bool:
    """Does the edge's path pass inside ``rect``? Sampled, which is ample at diagram scale."""
    left, bottom, right, top = rect.bounds
    for i in range(1, samples):
        x, y = _bezier_point(start, end, bend, i / samples)
        if left <= x <= right and bottom <= y <= top:
            return True
    return False


def route_edges(
    model: Model, layout: Layout, style: DiagramStyle, margin: float | None = None
) -> dict[tuple[str, str], float]:
    """A bend for each straight edge that would otherwise run through a third node.

    Reuses the shape of the label-placement pass from task-20260804-205013: a deterministic
    candidate-and-score sweep, smallest change first. **Zero is always the first candidate**, so an
    edge with a clear path is left exactly as it was and a figure with no crossings is byte-identical
    to before this existed -- which is what keeps the already-approved small figures unchanged.

    Only edges that actually cross get bent; bending everything would make the clean figures worse
    to fix a problem they do not have.
    """
    margin = style.edge_clearance if margin is None else margin
    rects = {
        name: node_rect(name, model, layout, style) for name in model.names if name in layout
    }
    padded = {
        name: Rect(r.x, r.y, r.width + 2 * margin, r.height + 2 * margin)
        for name, r in rects.items()
    }

    bends: dict[tuple[str, str], float] = {}
    for source, target, _kind in _straight_edges(model):
        if source not in layout or target not in layout:
            continue
        start = boundary_point(layout[source], layout[target], rects[source], 0.0)
        end = boundary_point(layout[target], layout[source], rects[target], 0.0)
        obstacles = [r for name, r in padded.items() if name not in (source, target)]
        for candidate in style.edge_bends:
            if not any(_path_hits_rect(start, end, r, candidate) for r in obstacles):
                if candidate:
                    bends[(source, target)] = candidate
                break
    return bends


def _straight_edges(model: Model) -> list[tuple[str, str, str]]:
    """Edges drawn as straight lines: directed paths, and the first co-path of each pair."""
    out: list[tuple[str, str, str]] = [
        (e.src, e.dst, "directed") for e in model.directed_edges
    ]
    seen: dict[frozenset, int] = {}
    for copath in model.copaths:
        pair = frozenset((copath.a, copath.b))
        index = seen.get(pair, 0)
        seen[pair] = index + 1
        if index == 0:  # later ones on the same pair are already bowed apart
            out.append((copath.a, copath.b, "copath"))
    return out


def edge_node_crossings(
    model: Model,
    layout: Layout,
    style: DiagramStyle,
    margin: float = 0.0,
    bends: dict[tuple[str, str], float] | None = None,
) -> list[EdgeCrossing]:
    """Every straight edge whose path runs through a node that is not one of its endpoints.

    Purely a layout property -- no covariance is affected -- but an arrow driven through a
    variable's box reads as a mistake, and where it clips an ellipse tangentially it can be
    misread as a doubled border or a variance self-loop. The count grows with pedigree depth, so
    this is kept as a test rather than a one-off check: see
    ``tests/test_render.py::test_no_edge_crosses_a_third_node``.

    Node extents come from the style, so the check tracks whatever the current node sizing is.
    Only straight edges are examined; curved ones (bidirected, and bowed co-paths) route around by
    construction.
    """
    rects = {
        name: node_rect(name, model, layout, style)
        for name in model.names
        if name in layout
    }
    if margin:
        rects = {
            name: Rect(r.x, r.y, r.width + 2 * margin, r.height + 2 * margin)
            for name, r in rects.items()
        }

    bends = {} if bends is None else bends
    unpadded = {
        name: node_rect(name, model, layout, style)
        for name in model.names
        if name in layout
    }
    crossings: list[EdgeCrossing] = []
    for source, target, kind in _straight_edges(model):
        if source not in layout or target not in layout:
            continue
        start = boundary_point(layout[source], layout[target], unpadded[source], 0.0)
        end = boundary_point(layout[target], layout[source], unpadded[target], 0.0)
        bend = bends.get((source, target), 0.0)
        for name, rect in rects.items():
            if name in (source, target):
                continue
            if _path_hits_rect(start, end, rect, bend):
                crossings.append(EdgeCrossing(source, target, name, kind))
    return crossings


def node_rect(name: str, model: Model, layout: Layout, style: DiagramStyle) -> Rect:
    """The bounding box of a node, from its label and shape."""
    variable = model.var(name)
    label = style.node_label(name, variable.label)
    width, height = style.node_size(label, variable.latent)
    x, y = layout[name]
    return Rect(x, y, width, height)


def boundary_point(
    origin: tuple[float, float],
    towards: tuple[float, float],
    rect: Rect,
    clearance: float,
) -> tuple[float, float]:
    """Where the segment from ``origin`` to ``towards`` leaves the box around ``origin``.

    Treats the node as its bounding rectangle, which is exact for an observed variable and
    slightly conservative for an ellipse -- erring on the side of a small visible gap rather than
    an arrowhead buried under the node.
    """
    dx = towards[0] - origin[0]
    dy = towards[1] - origin[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return origin
    ux, uy = dx / length, dy / length
    half_w = rect.width / 2 + clearance
    half_h = rect.height / 2 + clearance
    # distance along the ray to each side of the box; take the nearer crossing
    tx = half_w / abs(ux) if abs(ux) > 1e-12 else math.inf
    ty = half_h / abs(uy) if abs(uy) > 1e-12 else math.inf
    t = min(tx, ty, length)
    return (origin[0] + ux * t, origin[1] + uy * t)


def _candidate_point(
    start: tuple[float, float],
    end: tuple[float, float],
    position: float,
    offset: float,
    bow: float = 0.0,
) -> tuple[float, float]:
    """A point ``position`` of the way along the edge, displaced ``offset`` perpendicular."""
    x = start[0] + (end[0] - start[0]) * position
    y = start[1] + (end[1] - start[1]) * position
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return (x + offset, y)
    # matplotlib's arc3 bows toward the chord rotated clockwise; match it so a label on a curved
    # edge starts from the curve rather than the chord
    x += bow * dy
    y -= bow * dx
    return (x - offset * dy / length, y + offset * dx / length)


def label_rect(
    point: tuple[float, float], text: str, style: DiagramStyle
) -> Rect:
    """The box a label occupies. One definition, so scoring and measuring cannot disagree."""
    width, height = style.node_size(text, latent=False)
    return Rect(
        point[0],
        point[1],
        width + 2 * style.label_pad,
        style.raster_line_height + 2 * style.label_pad,
    )


def labelled_edges(
    model: Model, style: DiagramStyle
) -> list[tuple[tuple[str, str], str, float, str]]:
    """The edges that carry a label, in a fixed draw order, as ``(key, text, bow, kind)``.

    Lives here rather than in either back end because both must build it identically -- the order
    is what makes :func:`place_labels` deterministic, and if the two disagreed their figures would
    place labels differently.

    **Variance self-loops are included** (``kind == "variance"``). They used to be excluded, which
    meant their label was positioned by a hard-coded rule that checked nothing -- and a label
    sitting on its own loop's arc is the most visible defect in a crowded figure.
    """
    out: list[tuple[tuple[str, str], str, float, str]] = []
    for edge in model.directed_edges:
        text = style.edge_label((edge.src, edge.dst), edge.coeff)
        if text:
            out.append(((edge.src, edge.dst), text, 0.0, "directed"))
    for edge in model.bidirected_edges:
        if edge.is_variance:
            if not style.draws_variance(False):
                continue
            text = style.edge_label((edge.a, edge.a), edge.value)
            if text:
                out.append(((edge.a, edge.a), text, 0.0, "variance"))
            continue
        text = style.edge_label((edge.a, edge.b), edge.value)
        if text:
            out.append(((edge.a, edge.b), text, style.bidirected_bend / 100.0, "bidirected"))
    seen: dict[frozenset, int] = {}
    for copath in model.copaths:
        pair = frozenset((copath.a, copath.b))
        index = seen.get(pair, 0)
        seen[pair] = index + 1
        text = style.copath_label(copath)
        if text:
            out.append(((copath.a, copath.b), text, -0.12 * index, "copath"))
    return out


#: how far around the loop's direction the arc leaves and re-enters the node, in degrees.
#: 30 reproduces TikZ's ``loop above, in=120, out=60``.
LOOP_HALF_ANGLE = 30.0

#: the loop direction and looseness the back ends have always used
DEFAULT_LOOP_DIRECTION = 90.0
DEFAULT_LOOP_LOOSENESS = 6.0

#: candidate loop directions, in degrees, tried in this order. Above first, so a loop that has no
#: reason to move stays exactly where it has always been drawn.
LOOP_DIRECTIONS: tuple[float, ...] = (90.0, 0.0, 180.0, 270.0, 45.0, 135.0, 315.0, 225.0)
#: candidate loop tightnesses, tried in this order for each direction
LOOP_LOOSENESSES: tuple[float, ...] = (6.0, 8.0, 4.5, 11.0)

#: how many times to sweep the greedy pass. A single pass leaves the last labels with whatever
#: space remains and never revisits the first; two or three sweeps let early labels move once the
#: crowd exists, and it converges long before this in practice.
MAX_SWEEPS = 3
# Scoring weights. Rect overlaps are scored as a FRACTION of the label's own area, not as a raw
# area: a label box is a small fraction of a cm^2, so a raw area sits at ~0.01-0.6 while a covered
# edge length sits at ~1, and an unnormalised sum makes a label-on-a-node look cheaper than a
# label-on-a-line. Measured: with raw areas, label-node collisions across the battery went UP
# (47 -> 57) while everything else improved, because labels were escaping edges onto nodes.
#: a label fully covering another label counts 1.0
LABEL_OVERLAP_WEIGHT = 1.0
#: Weighted ABOVE the label term, which is not obvious and was settled by sweeping it and
#: measuring across the writeup figures and the battery (label-label / label-edge / label-node /
#: ambiguous counts):
#:     1.0 -> 26 / 119 / 52 / 48
#:     2.0 -> 28 / 117 / 49 / 50
#:     3.0 -> 28 / 119 / 39 / 52     <- chosen: the only setting where all four beat the old pass
#:     5.0 -> 28 / 122 / 35 / 53
#: (the old pass scored 113 / 343 / 47 / 84.) At 1.0 labels escape edges by grazing node corners,
#: which is how label-node came out WORSE than before while everything else improved.
NODE_OVERLAP_WEIGHT = 3.0
#: per centimetre of a FOREIGN edge covered. A label's own edge is exempt.
FOREIGN_EDGE_WEIGHT = 0.30
#: per centimetre by which a label is closer to a foreign edge than to its own
AMBIGUITY_WEIGHT = 0.15
#: per centimetre of a self-loop's ARC that lands inside somebody else's node. Weighted above the
#: label terms: moving a loop's arc across a neighbour to buy its label some room trades a small
#: defect for a large one.
LOOP_ARC_WEIGHT = 2.0
#: per centimetre of a self-loop's own arc covered by its own label. The one place a label is not
#: allowed to sit on its own edge -- see the comment at the use site.
OWN_LOOP_ARC_WEIGHT = 1.5
#: multiples of half a label's height to try pushing a loop label out past its own arc
LOOP_CLEARANCES: tuple[float, ...] = (1.0, 1.5, 2.1, 2.8)


@dataclass(frozen=True)
class EdgePath:
    """The polyline a back end will actually draw for one edge, sampled.

    Exists so label placement and the collision metric score against the *drawn* geometry rather
    than a chord. ``key`` identifies the edge so a label can tell its **own** edge from a foreign
    one -- a distinction that matters more than it looks, because a label is allowed to sit on its
    own edge (that is what the white fill is for) and must not sit on anybody else's.
    """

    key: tuple[str, str]
    kind: str
    points: tuple[tuple[float, float], ...]

    def distance_to(self, point: tuple[float, float]) -> float:
        """Shortest distance from ``point`` to the sampled path."""
        px, py = point
        return min(math.hypot(px - x, py - y) for x, y in self.points)

    def length_inside(self, rect: "Rect") -> float:
        """Length of path lying inside ``rect``.

        The natural measure for a label sitting on a line: a rectangle-rectangle *area* is
        meaningless against a zero-width path, and the length a label covers is exactly how much
        of the line its white fill erases.
        """
        left, bottom, right, top = rect.bounds
        total = 0.0
        for (x0, y0), (x1, y1) in zip(self.points, self.points[1:]):
            inside0 = left <= x0 <= right and bottom <= y0 <= top
            inside1 = left <= x1 <= right and bottom <= y1 <= top
            segment = math.hypot(x1 - x0, y1 - y0)
            if inside0 and inside1:
                total += segment
            elif inside0 or inside1:
                total += segment / 2.0  # crossing the boundary: charge half, it is a sample
        return total


def loop_path(
    centre: tuple[float, float],
    rect: Rect,
    direction: float = DEFAULT_LOOP_DIRECTION,
    looseness: float = DEFAULT_LOOP_LOOSENESS,
    samples: int = 24,
) -> tuple[tuple[float, float], ...]:
    """The arc of a variance self-loop, as a sampled polyline.

    TikZ draws it with ``to[loop, out=d-30, in=d+30, looseness=L]``; matplotlib with a strongly
    bowed ``arc3``. Both are approximated by a quadratic Bezier that leaves and re-enters the node
    at those angles and bulges along ``direction`` by an amount set by ``looseness`` -- close
    enough to decide whether a label lands on the arc, which is all this is for.
    """
    out_angle = math.radians(direction - LOOP_HALF_ANGLE)
    in_angle = math.radians(direction + LOOP_HALF_ANGLE)
    radius_x, radius_y = rect.width / 2.0, rect.height / 2.0
    start = (centre[0] + radius_x * math.cos(out_angle), centre[1] + radius_y * math.sin(out_angle))
    end = (centre[0] + radius_x * math.cos(in_angle), centre[1] + radius_y * math.sin(in_angle))
    reach = max(radius_x, radius_y) * looseness / 6.0 * 1.15
    theta = math.radians(direction)
    control = (
        (start[0] + end[0]) / 2.0 + reach * math.cos(theta),
        (start[1] + end[1]) / 2.0 + reach * math.sin(theta),
    )
    points = []
    for i in range(samples + 1):
        t = i / samples
        u = 1 - t
        points.append(
            (
                u * u * start[0] + 2 * u * t * control[0] + t * t * end[0],
                u * u * start[1] + 2 * u * t * control[1] + t * t * end[1],
            )
        )
    return tuple(points)


def loop_label_point(
    centre: tuple[float, float],
    rect: Rect,
    direction: float = DEFAULT_LOOP_DIRECTION,
    looseness: float = DEFAULT_LOOP_LOOSENESS,
    clearance: float = 0.0,
) -> tuple[float, float]:
    """Where a self-loop's label goes: just beyond the far side of its own arc.

    Past the apex, not on it. The label sitting *on* its own loop is the single most visible defect
    in a crowded figure, and it happens because the loop's own arc was never treated as an obstacle
    for the loop's own label.
    """
    path = loop_path(centre, rect, direction, looseness)
    apex = max(
        path,
        key=lambda p: (p[0] - centre[0]) * math.cos(math.radians(direction))
        + (p[1] - centre[1]) * math.sin(math.radians(direction)),
    )
    theta = math.radians(direction)
    return (apex[0] + clearance * math.cos(theta), apex[1] + clearance * math.sin(theta))


def edge_paths(
    model: Model,
    layout: Layout,
    style: DiagramStyle,
    loop_choices: dict[str, tuple[float, float]] | None = None,
) -> list[EdgePath]:
    """Every drawn edge as a sampled path, in the back ends' draw order.

    ``loop_choices`` maps a node to its chosen ``(direction, looseness)``; anything absent uses the
    default, which is what the back ends draw when a loop did not need to move.
    """
    loop_choices = loop_choices or {}
    bends = route_edges(model, layout, style) if style.route_edges_around_nodes else {}
    rects = {n: node_rect(n, model, layout, style) for n in model.names if n in layout}
    out: list[EdgePath] = []

    def sampled(a: str, b: str, bend: float, samples: int = 32):
        return tuple(
            _bezier_point(layout[a], layout[b], bend, i / samples) for i in range(samples + 1)
        )

    for edge in model.directed_edges:
        if edge.src in layout and edge.dst in layout:
            bend = bends.get((edge.src, edge.dst), 0.0)
            out.append(EdgePath((edge.src, edge.dst), "directed", sampled(edge.src, edge.dst, -bend)))
    for edge in model.bidirected_edges:
        if edge.a not in layout or edge.b not in layout:
            continue
        if edge.is_variance:
            if not style.draws_variance(False):
                continue
            direction, looseness = loop_choices.get(
                edge.a, (DEFAULT_LOOP_DIRECTION, DEFAULT_LOOP_LOOSENESS)
            )
            out.append(
                EdgePath(
                    (edge.a, edge.a),
                    "variance",
                    loop_path(layout[edge.a], rects[edge.a], direction, looseness),
                )
            )
        else:
            out.append(
                EdgePath((edge.a, edge.b), "bidirected", sampled(edge.a, edge.b, style.bidirected_bend))
            )
    seen: dict[frozenset, int] = {}
    for copath in model.copaths:
        if copath.a not in layout or copath.b not in layout:
            continue
        pair = frozenset((copath.a, copath.b))
        index = seen.get(pair, 0)
        seen[pair] = index + 1
        bow = -12.0 * index if index else -bends.get((copath.a, copath.b), 0.0)
        out.append(EdgePath((copath.a, copath.b), "copath", sampled(copath.a, copath.b, bow)))
    return out


def place_labels(
    model: Model,
    layout: Layout,
    style: DiagramStyle,
    labelled_edges: list[tuple[tuple[str, str], str, float, str]],
) -> dict[tuple[str, str], LabelPlacement]:
    """Choose a placement for each labelled edge, deterministically.

    ``labelled_edges`` is ``(key, text, bow, kind)`` in the order the back end will draw them; that
    order fixes the result, so both back ends must pass the same one.

    What a candidate is scored against, and why each term is there:

    - **other labels** and **node boxes** -- overlap area. Both make a label unreadable.
    - **foreign edges** -- the length of somebody else's line the label's white fill would erase.
      A label's **own** edge is exempt: sitting on it is what the fill is for, and penalising it
      would send every label fleeing the edge it annotates, which is worse than the disease.
    - **ambiguity** -- how much closer the label sits to a foreign edge than to its own. A label
      can have zero overlap and still be unreadable, floating between two crossing edges belonging
      to neither. Zero overlap is necessary, not sufficient.

    Two properties are preserved deliberately. The exact midpoint with no offset is always the
    first candidate and wins on a tie, so **a diagram with no collisions comes out byte-identical
    to what it did before any of this existed**. And a self-loop gets an entry here only if its
    default placement collided -- absent means the back end draws it exactly as it always has.

    The pass is swept up to :data:`MAX_SWEEPS` times. A single greedy pass gives the labels drawn
    last whatever space is left and never revisits the early ones; sweeping lets an early label
    move once it can see the crowd that arrived after it. It stops as soon as a sweep changes
    nothing.
    """
    nodes = [node_rect(name, model, layout, style) for name in model.names if name in layout]
    rects = {name: node_rect(name, model, layout, style) for name in model.names if name in layout}
    active = [
        (key, text, bow, kind)
        for key, text, bow, kind in labelled_edges
        if key[0] in layout and key[1] in layout
    ]

    def loop_defaults() -> dict[str, tuple[float, float]]:
        return {
            key[0]: (DEFAULT_LOOP_DIRECTION, DEFAULT_LOOP_LOOSENESS)
            for key, _t, _b, kind in active
            if kind == "variance"
        }

    chosen: dict[tuple[str, str], LabelPlacement] = {}
    loop_choices = loop_defaults()

    def candidates_for(key, text, bow, kind):
        """(point, position, offset, loop_direction, loop_looseness) in preference order."""
        if kind == "variance":
            node = key[0]
            rect = rects[node]
            half_height = style.raster_line_height / 2.0 + style.label_pad
            for direction in LOOP_DIRECTIONS:
                for looseness in LOOP_LOOSENESSES:
                    for step in LOOP_CLEARANCES:
                        point = loop_label_point(
                            layout[node], rect, direction, looseness,
                            clearance=half_height * step,
                        )
                        yield (point, 0.5, 0.0, direction, looseness)
            return
        start_point, end_point = layout[key[0]], layout[key[1]]
        for position in style.label_positions:
            for offset in style.label_offsets:
                yield (
                    _candidate_point(start_point, end_point, position, offset, bow),
                    position,
                    offset,
                    None,
                    None,
                )

    for sweep in range(MAX_SWEEPS):
        paths = edge_paths(model, layout, style, loop_choices)
        by_key: dict[tuple[str, str], list[EdgePath]] = {}
        for path in paths:
            by_key.setdefault(path.key, []).append(path)
        changed = False

        for key, text, bow, kind in active:
            others = [
                label_rect(placement.point, other_text, style)
                for (other_key, other_text, _b, _k), placement in (
                    (entry, chosen[entry[0]]) for entry in active if entry[0] in chosen
                )
                if other_key != key
            ]
            own_paths = by_key.get(key, [])
            foreign = [path for path in paths if path.key != key]

            best_cost = None
            best = None
            foreign_nodes = [
                rect for name, rect in rects.items() if not (kind == "variance" and name == key[0])
            ]
            for point, position, offset, direction, looseness in candidates_for(
                key, text, bow, kind
            ):
                rect = label_rect(point, text, style)
                arc_penalty = 0.0
                if kind == "variance":
                    # the loop's ARC must not be flung across a neighbour to make room for its
                    # label. Scoring only the label sent loops sideways into adjacent nodes, which
                    # showed up as label-node collisions going UP across the battery.
                    arc = loop_path(layout[key[0]], rects[key[0]], direction, looseness)
                    arc_path = EdgePath(key, "variance", arc)
                    arc_penalty = LOOP_ARC_WEIGHT * sum(
                        arc_path.length_inside(node) for node in foreign_nodes
                    )
                    # ...and the loop's own arc is an obstacle for its OWN label. Everywhere else
                    # a label may sit on its own edge, but a `1/4` bisected by the very loop it
                    # annotates is the most visible defect in a crowded figure, so this one case
                    # is the exception.
                    arc_penalty += OWN_LOOP_ARC_WEIGHT * arc_path.length_inside(rect)
                own_area = max(rect.width * rect.height, 1e-9)
                penalty = LABEL_OVERLAP_WEIGHT * sum(
                    rect.overlap(other) for other in others
                ) / own_area
                penalty += NODE_OVERLAP_WEIGHT * sum(
                    rect.overlap(node) for node in nodes
                ) / own_area
                penalty += FOREIGN_EDGE_WEIGHT * sum(path.length_inside(rect) for path in foreign)
                ambiguity = 0.0
                if own_paths and foreign:
                    own_distance = min(path.distance_to(point) for path in own_paths)
                    foreign_distance = min(path.distance_to(point) for path in foreign)
                    ambiguity = max(0.0, own_distance - foreign_distance)
                cost = (
                    penalty + AMBIGUITY_WEIGHT * ambiguity + arc_penalty,
                    abs(position - 0.5),
                    abs(offset),
                    0.0 if direction is None else LOOP_DIRECTIONS.index(direction),
                    0.0 if looseness is None else LOOP_LOOSENESSES.index(looseness),
                )
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best = (point, position, offset, direction, looseness)
                if cost[0] == 0.0 and cost[1] == 0.0 and cost[2] == 0.0 and cost[3] == 0.0 and cost[4] == 0.0:
                    break  # the default is already clear; nothing can beat it

            point, position, offset, direction, looseness = best
            if not style.avoid_label_collisions:
                if kind == "variance":
                    chosen.pop(key, None)
                    continue
                point = _candidate_point(layout[key[0]], layout[key[1]], 0.5, 0.0, bow)
                position, offset, direction, looseness = 0.5, 0.0, None, None

            if kind == "variance" and best_cost is not None and best_cost[0] == 0.0 and (
                direction == DEFAULT_LOOP_DIRECTION and looseness == DEFAULT_LOOP_LOOSENESS
            ):
                # the loop is fine where it has always been: leave it to the back end's own rule so
                # the emitted figure is unchanged
                if chosen.pop(key, None) is not None:
                    changed = True
                loop_choices[key[0]] = (DEFAULT_LOOP_DIRECTION, DEFAULT_LOOP_LOOSENESS)
                continue

            placement = LabelPlacement(position, offset, point, direction, looseness)
            if chosen.get(key) != placement:
                changed = True
            chosen[key] = placement
            if kind == "variance":
                loop_choices[key[0]] = (direction, looseness)

        if not changed:
            break

    return chosen
