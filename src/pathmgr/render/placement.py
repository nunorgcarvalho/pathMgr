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

__all__ = ["LabelPlacement", "Rect", "boundary_point", "labelled_edges", "place_labels"]


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
    """

    position: float
    offset: float
    point: tuple[float, float]


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


def labelled_edges(model: Model, style: DiagramStyle) -> list[tuple[tuple[str, str], str, float]]:
    """The edges that carry a label, in a fixed draw order, as ``(key, text, bow)``.

    Lives here rather than in either back end because both must build it identically -- the order
    is what makes :func:`place_labels` deterministic, and if the two disagreed their figures would
    place labels differently. Variance self-loops are excluded: their label is positioned relative
    to the loop, not to a chord between two nodes.
    """
    out: list[tuple[tuple[str, str], str, float]] = []
    for edge in model.directed_edges:
        text = style.edge_label((edge.src, edge.dst), edge.coeff)
        if text:
            out.append(((edge.src, edge.dst), text, 0.0))
    for edge in model.bidirected_edges:
        if edge.is_variance:
            continue
        text = style.edge_label((edge.a, edge.b), edge.value)
        if text:
            out.append(((edge.a, edge.b), text, style.bidirected_bend / 100.0))
    seen: dict[frozenset, int] = {}
    for copath in model.copaths:
        pair = frozenset((copath.a, copath.b))
        index = seen.get(pair, 0)
        seen[pair] = index + 1
        text = style.edge_label((copath.a, copath.b), copath.coefficient)
        if text:
            out.append(((copath.a, copath.b), text, -0.12 * index))
    return out


def place_labels(
    model: Model,
    layout: Layout,
    style: DiagramStyle,
    labelled_edges: list[tuple[tuple[str, str], str, float]],
) -> dict[tuple[str, str], LabelPlacement]:
    """Choose a placement for each labelled edge, greedily and deterministically.

    ``labelled_edges`` is ``(key, label text, bow)`` in the order the back end will draw them;
    that order fixes the result, so both back ends must pass the same one. Edges are processed in
    the given order and each takes the best candidate against everything already placed.
    """
    obstacles: list[Rect] = [
        node_rect(name, model, layout, style) for name in model.names if name in layout
    ]
    placements: dict[tuple[str, str], LabelPlacement] = {}

    for key, text, bow in labelled_edges:
        a, b = key
        if a not in layout or b not in layout:
            continue
        start, end = layout[a], layout[b]
        width, height = style.node_size(text, latent=False)
        width += 2 * style.label_pad
        height = style.raster_line_height + 2 * style.label_pad

        if not style.avoid_label_collisions:
            point = _candidate_point(start, end, 0.5, 0.0, bow)
            placements[key] = LabelPlacement(0.5, 0.0, point)
            obstacles.append(Rect(point[0], point[1], width, height))
            continue

        best: tuple[float, float, float, tuple[float, float]] | None = None
        for position in style.label_positions:
            for offset in style.label_offsets:
                point = _candidate_point(start, end, position, offset, bow)
                candidate = Rect(point[0], point[1], width, height)
                penalty = sum(candidate.overlap(other) for other in obstacles)
                # prefer the midpoint and no offset, all else equal: the tie-breakers keep the
                # output stable and keep simple diagrams looking exactly as they did
                cost = (
                    penalty,
                    abs(position - 0.5),
                    abs(offset),
                )
                if best is None or cost < best[:3]:
                    best = (*cost, point)
                    best_choice = (position, offset, point)
                if penalty == 0.0 and position == 0.5 and offset == 0.0:
                    break  # the default is already clear; nothing can beat it
            else:
                continue
            break

        position, offset, point = best_choice
        placements[key] = LabelPlacement(position, offset, point)
        obstacles.append(Rect(point[0], point[1], width, height))

    return placements
