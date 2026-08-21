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

__all__ = ["Collision", "CollisionReport", "Diagnosis", "collision_report", "diagnose"]


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
    # the coding removes labels, so the metric must be built with it -- otherwise it scores labels
    # that are not drawn and reports collisions the reader will never see
    coding = style.coefficient_coding(model)
    edges = labelled_edges(model, style, coding)
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


@dataclass
class Diagnosis:
    """Everything wrong with a figure, in one object a generator can assert on.

    Two kinds of finding, deliberately together. **Layout** problems make a figure hard to read.
    **Model** problems make it untrustworthy, and a diagram that draws beautifully from a model that
    cannot be resolved is the worse failure of the two -- so an API that answers "is this fine to
    draw AND fine to trust" beats one that only counts overlaps.

    Follows :func:`~pathmgr.render.placement.edge_node_crossings`: it reports, it does not fix.
    """

    collisions: CollisionReport
    #: straight edges whose path runs through a third node's box
    crossings: tuple = ()
    #: model-level problems: things that will not compute, or will compute the wrong number
    model_issues: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True if there is nothing outstanding. What a figure generator asserts on."""
        return not self.collisions.collisions and not self.collisions.ambiguous and (
            not self.crossings and not self.model_issues
        )

    def summary(self) -> str:
        return (
            f"{self.collisions.summary()}  crossings={len(self.crossings)}  "
            f"model={len(self.model_issues)}"
        )

    def __str__(self) -> str:
        lines = [self.summary(), *str(self.collisions).splitlines()[1:]]
        for crossing in self.crossings:
            lines.append(f"  crossing: {crossing}")
        for issue in self.model_issues:
            lines.append(f"  model: {issue}")
        return "\n".join(lines)


def _model_issues(model: Model) -> tuple[str, ...]:
    """Model-level findings worth surfacing next to the layout ones.

    The co-path case is here because it cost a real user a real result: declaring two co-paths that
    share a node -- every half-sibling or in-law pedigree, i.e. one person with two mates -- cannot
    resolve a declared ``correlation=``, because assortment changes the variance the second one
    would have to be resolved against. The engines raise ``CoPathVarianceError`` rather than return
    a wrong number, and this reports the same condition *without* raising, so a figure script can
    see it coming.
    """
    issues: list[str] = []
    for issue in model.validate():
        if issue.severity == "error":
            issues.append(f"[{issue.severity}] {issue.message}")
    standardized = [c for c in model.copaths if c.is_standardized]
    for copath in standardized:
        shared = [
            other
            for other in model.copaths
            if other is not copath and {other.a, other.b} & {copath.a, copath.b}
        ]
        if shared:
            issues.append(
                f"co-path {copath.a!r} -- {copath.b!r} is declared by correlation but shares a "
                f"node with another co-path; the engines will raise CoPathVarianceError. Use an "
                f"explicit coefficient= (raw mu) for these."
            )
            break
    return tuple(issues)


def diagnose(
    model: Model,
    layout: Layout | None = None,
    style: DiagramStyle | None = None,
) -> Diagnosis:
    """What is still wrong with this figure -- usable as an assertion from a generator script.

    >>> import pathmgr as pm
    >>> from pathmgr.render import Layout
    >>> report = diagnose(pm.from_text("y ~ b*x\\nx ~~ V*x\\ny ~~ V*y"), Layout())
    >>> report.ok
    True

    A figure-generation script can then fail loudly when a model change quietly makes a figure
    worse, rather than shipping it:

        assert diagnose(model, layout, style).ok, diagnose(model, layout, style)
    """
    from .placement import edge_node_crossings

    style = style or DiagramStyle()
    layout = (layout or Layout()).completed(model)
    return Diagnosis(
        collisions=collision_report(model, layout, style),
        crossings=tuple(edge_node_crossings(model, layout, style)),
        model_issues=_model_issues(model),
    )
