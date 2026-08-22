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
    #: things a human may want to look at that are NOT defects -- see :func:`_consistency_advisories`
    advisories: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True if there is nothing outstanding. What a figure generator asserts on.

        **Advisories deliberately do not affect this.** They are not defects, and a generator that
        asserts ``.ok`` must not start failing because a figure has two identically-labelled edges
        placed differently -- which is often the *correct* outcome.
        """
        return not self.collisions.collisions and not self.collisions.ambiguous and (
            not self.crossings and not self.model_issues
        )

    def summary(self) -> str:
        return (
            f"{self.collisions.summary()}  crossings={len(self.crossings)}  "
            f"model={len(self.model_issues)}  advisories={len(self.advisories)}"
        )

    def __str__(self) -> str:
        lines = [self.summary(), *str(self.collisions).splitlines()[1:]]
        for crossing in self.crossings:
            lines.append(f"  crossing: {crossing}")
        for issue in self.model_issues:
            lines.append(f"  model: {issue}")
        for advisory in self.advisories:
            lines.append(f"  advisory: {advisory}")
        return "\n".join(lines)


#: how far apart two identical labels' perpendicular offsets must be, in cm, before the difference
#: counts as "materially different treatment". Chosen as roughly one label height: below that the
#: two read as the same treatment with a nudge, which is not worth mentioning.
CONSISTENCY_OFFSET_TOLERANCE = 0.35


def _consistency_advisories(model, layout, style, placements, texts) -> tuple[str, ...]:
    """Identical labels given materially different treatment -- an ADVISORY, not a defect.

    Two structurally symmetric nodes carrying the same label, one placed inline and the other pushed
    out with a leader, reads as lopsided to a human while satisfying every metric here: both are
    collision-free, so nothing else in this module can see it.

    It is reported and **not** fixed, and that is a considered choice rather than laziness. On the
    figure that prompted it, the asymmetric layout was the *only* zero-collision configuration and
    every symmetric alternative cost collisions -- so a rule enforcing consistency would have to be
    "accept collisions to gain symmetry", an editorial judgement that varies per figure and that no
    measurement here can adjudicate. Handing a human the facts is the right shape.

    Which is also why the wording says nothing is wrong: a reader who "fixes" every advisory without
    measuring will make figures worse.
    """
    by_text: dict[str, list[tuple[tuple[str, str], object]]] = {}
    for key, placement in placements.items():
        text = texts.get(key)
        if text:
            by_text.setdefault(text, []).append((key, placement))

    out: list[str] = []
    for text, entries in sorted(by_text.items()):
        if len(entries) < 2:
            continue
        leadered = sorted(key for key, p in entries if p.leader_to is not None)
        plain = sorted(key for key, p in entries if p.leader_to is None)
        offsets = [p.offset for _key, p in entries]
        spread = max(offsets) - min(offsets)

        reason = None
        if leadered and plain:
            reason = (
                f"{len(leadered)} of {len(entries)} use a leader line and the rest do not"
            )
        elif spread > CONSISTENCY_OFFSET_TOLERANCE:
            reason = f"their offsets differ by {spread:.2f}cm"
        if reason is None:
            continue

        hint = ""
        if leadered and plain:
            model_key, target = plain[0], leadered[0]
            if target[0] == target[1] and model_key[0] == model_key[1]:
                matched = placements[model_key]
                hint = (
                    f" To make them match, copy the inline one's loop: "
                    f"loop_overrides={{{target[0]!r}: "
                    f"({matched.loop_direction:g}, {matched.loop_looseness:g})}}."
                )
            else:
                hint = (
                    " `label_placement_overrides` can pin them to the same treatment."
                )
        out.append(
            f"{len(entries)} labels read {text!r} but are placed differently ({reason}). "
            f"Nothing is wrong -- the placer optimised for collisions, and forcing them to match "
            f"may COST collisions, so measure before changing it.{hint}"
        )
    return tuple(out)


def _model_issues(model: Model) -> tuple[str, ...]:
    """Model-level findings worth surfacing next to the layout ones.

    This used to also report "a co-path declared by ``correlation=`` shares a node with another
    co-path", which was a real limitation and which a real user hit. task-20260805-170500 removed
    the limitation -- the engines now resolve co-paths in dependency order -- so the finding was
    deleted with it rather than left to warn about something that works. A diagnostic that reports a
    problem you no longer have is worse than none: it sends the reader to fix the wrong thing.
    """
    issues = [
        f"[{issue.severity}] {issue.message}"
        for issue in model.validate()
        if issue.severity == "error"
    ]
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
    coding = style.coefficient_coding(model)
    edges = labelled_edges(model, style, coding)
    placements = place_labels(model, layout, style, edges)
    texts = {key: text for key, text, _bow, _kind in edges}
    return Diagnosis(
        collisions=collision_report(model, layout, style),
        crossings=tuple(edge_node_crossings(model, layout, style)),
        model_issues=_model_issues(model),
        advisories=_consistency_advisories(model, layout, style, placements, texts),
    )
