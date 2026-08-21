"""What is still wrong with a diagram, measured rather than eyeballed.

You cannot tell whether a label-placement change worked by looking at a figure: crowded diagrams
have dozens of labels, the eye goes to the worst one, and a change that fixes it while making four
others slightly worse looks like a win. So placement changes are judged by
:func:`collision_report`, and the numbers go in the commit message.

Three kinds of collision are counted separately, because they are not equally bad:

- **label vs label** -- two labels on top of each other. Both unreadable.
- **label vs a FOREIGN edge** -- the label's white fill erases part of somebody else's line. Worse
  than it looks: a silently deleted arrow cannot be noticed by a reader, whereas an obvious
  overlap at least announces itself.
- **label vs node** -- a label over a node's box.

A label overlapping its **own** edge is not counted and is not a defect. That is what the white
fill is for and how a correctly placed label has always looked; penalising it would make every
label flee the edge it annotates.

Rect-versus-rect overlaps are reported as **area**; a label over a path is reported as the **length
of path covered**, because the area of intersection with a zero-width line is zero by construction
and the covered length is exactly how much of the line disappears.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.model import Model
from .layout import Layout
from .placement import (
    EdgePath,
    Rect,
    edge_paths,
    label_rect,
    labelled_edges,
    node_rect,
    place_labels,
)
from .style import DiagramStyle

__all__ = ["Collision", "CollisionReport", "collision_report"]


@dataclass(frozen=True)
class Collision:
    """One overlap. ``amount`` is an area in cm^2, or a path length in cm for an edge."""

    kind: str
    label: tuple[str, str]
    against: str
    amount: float

    def __str__(self) -> str:
        label = f"{self.label[0]}--{self.label[1]}" if self.label[0] != self.label[1] else self.label[0]
        return f"{self.kind}: label[{label}] vs {self.against} ({self.amount:.4f})"


@dataclass
class CollisionReport:
    """Every outstanding overlap in one diagram, plus the totals worth quoting."""

    collisions: list[Collision] = field(default_factory=list)
    #: labels whose nearest foreign edge is closer than their own -- placed, but unattributable
    ambiguous: list[tuple[tuple[str, str], float]] = field(default_factory=list)
    n_labels: int = 0

    def of_kind(self, kind: str) -> list[Collision]:
        return [c for c in self.collisions if c.kind == kind]

    @property
    def total(self) -> float:
        return sum(c.amount for c in self.collisions)

    def summary(self) -> str:
        """One line per kind. This is what goes in a commit message."""
        parts = []
        for kind in ("label-label", "label-edge", "label-node"):
            hits = self.of_kind(kind)
            parts.append(f"{kind}={len(hits)}/{sum(h.amount for h in hits):.3f}")
        parts.append(f"ambiguous={len(self.ambiguous)}")
        return f"labels={self.n_labels}  " + "  ".join(parts)

    def __str__(self) -> str:
        lines = [self.summary()]
        for collision in sorted(
            self.collisions, key=lambda c: (-c.amount, c.kind, c.label, c.against)
        ):
            lines.append(f"  {collision}")
        for key, margin in sorted(self.ambiguous, key=lambda item: item[1]):
            lines.append(f"  ambiguous: label[{key[0]}--{key[1]}] margin {margin:+.3f}")
        return "\n".join(lines)


def collision_report(
    model: Model,
    layout: Layout,
    style: DiagramStyle | None = None,
) -> CollisionReport:
    """Measure what still overlaps what, after placement has done its best.

    Runs the real placement pass, so the numbers describe the figure that would actually be drawn.
    """
    style = style or DiagramStyle()
    layout = layout.completed(model)
    edges = labelled_edges(model, style)
    placements = place_labels(model, layout, style, edges)
    loop_choices = {
        key[0]: (p.loop_direction, p.loop_looseness)
        for key, p in placements.items()
        if p.loop_direction is not None
    }
    paths = edge_paths(model, layout, style, loop_choices)
    nodes = {name: node_rect(name, model, layout, style) for name in model.names if name in layout}

    report = CollisionReport(n_labels=len(edges))
    placed: list[tuple[tuple[str, str], Rect]] = []
    for key, text, _bow, _kind in edges:
        placement = placements.get(key)
        if placement is None:
            continue
        rect = label_rect(placement.point, text, style)
        for other_key, other in placed:
            area = rect.overlap(other)
            if area > 0:
                report.collisions.append(
                    Collision("label-label", key, f"label[{other_key[0]}--{other_key[1]}]", area)
                )
        for name, node in nodes.items():
            area = rect.overlap(node)
            if area > 0:
                report.collisions.append(Collision("label-node", key, f"node[{name}]", area))
        own, nearest_foreign, foreign_key = None, None, None
        for path in paths:
            if path.key == key:
                own = min(own, path.distance_to(placement.point)) if own is not None else (
                    path.distance_to(placement.point)
                )
                continue
            covered = path.length_inside(rect)
            if covered > 0:
                report.collisions.append(
                    Collision("label-edge", key, f"edge[{path.key[0]}--{path.key[1]}]", covered)
                )
            distance = path.distance_to(placement.point)
            if nearest_foreign is None or distance < nearest_foreign:
                nearest_foreign, foreign_key = distance, path.key
        if own is not None and nearest_foreign is not None and nearest_foreign < own:
            report.ambiguous.append((key, nearest_foreign - own))
        placed.append((key, rect))
    return report
