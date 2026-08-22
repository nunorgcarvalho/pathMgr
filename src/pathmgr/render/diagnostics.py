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

__all__ = [
    "Collision",
    "CollisionReport",
    "Diagnosis",
    "Extent",
    "collision_report",
    "diagnose",
    "extent",
]


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
    #: labels whose nearest foreign edge is closer than their own -- placed, but unattributable.
    #: **Leadered labels are excluded**: a hairline back to the edge resolves attribution *by
    #: construction*, which is the entire reason leaders exist, so counting them here would
    #: undercount the feature on exactly the figures it was built for.
    ambiguous: list[tuple[tuple[str, str], float]] = field(default_factory=list)
    #: labels that WOULD have been ambiguous but carry a leader line. Reported separately rather
    #: than silently dropped: "moved far and connected" is a different state from "unattributable",
    #: and a reader wants the count -- it is how many hairlines the figure has. Folding them into a
    #: quietly better `ambiguous` number would improve a metric for a reason nobody could see.
    leadered: list[tuple[tuple[str, str], float]] = field(default_factory=list)
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
        if self.leadered:
            parts.append(f"leadered={len(self.leadered)}")
        return f"labels={self.n_labels}  " + "  ".join(parts)

    def __str__(self) -> str:
        lines = [self.summary()]
        for collision in sorted(
            self.collisions, key=lambda c: (-c.amount, c.kind, c.label, c.against)
        ):
            lines.append(f"  {collision}")
        for key, margin in sorted(self.ambiguous, key=lambda item: item[1]):
            lines.append(f"  ambiguous: label[{key[0]}--{key[1]}] margin {margin:+.3f}")
        for key, margin in sorted(self.leadered, key=lambda item: item[1]):
            lines.append(
                f"  leadered: label[{key[0]}--{key[1]}] margin {margin:+.3f} "
                f"(attribution carried by its leader line)"
            )
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
            margin = nearest_foreign - own
            if placement.leader_to is not None:
                report.leadered.append((key, margin))
            else:
                report.ambiguous.append((key, margin))
        placed.append((key, rect))
    return report



@dataclass
class Extent:
    """How big the drawing is, and what is sticking out of it.

    **Nothing else here measures this**, and that is the gap it exists to close. Every other number
    in this module asks "do things overlap each other"; none asks "how big is the whole picture". So
    the placer can improve every metric it knows about while making a figure too wide to use, and
    report success -- which is exactly what happened to a `sidewaysfigure` already at the page limit:
    zero collisions, zero ambiguous, zero crossings, and a new overfull hbox.

    It will recur, and specifically because of leader lines and far placement: both work by moving
    labels **outward**, which is what grows the bounding box. The better the placer gets at avoiding
    collisions, the more often it inflates the footprint.

    **The absolute centimetres are an estimate; the comparisons are the useful part.** Node and
    label sizes come from :func:`~pathmgr.render.style.text_width`, which is fitted rather than
    exact. Measured against a direct ``\hbox`` of the same ``tikzpicture``, this read 22.246 cm
    where pdflatex measured 23.313 cm -- about 5% low on that figure. So:

    - **Do** use :attr:`overhang` and :attr:`outside` to catch a regression and to find *which*
      labels are responsible. Those are computed from the same geometry the placer used, so they
      answer "did this get wider, and because of what" exactly. On the figure that prompted this
      they named precisely the two loops the consumer had to override.
    - **Do not** treat :attr:`width` as a page-fit test. Take the authoritative number from the
      LaTeX log; a budget declared here is a tripwire, not a certificate.
    """

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    #: the extent of the node boxes alone -- the "intended" footprint of the diagram
    node_min_x: float
    node_min_y: float
    node_max_x: float
    node_max_y: float
    #: labels and leader endpoints reaching beyond the node span, worst first, as
    #: ``(label key, centimetres beyond, which side)``
    outside: tuple[tuple[tuple[str, str], float, str], ...] = ()

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def node_width(self) -> float:
        return self.node_max_x - self.node_min_x

    @property
    def node_height(self) -> float:
        return self.node_max_y - self.node_min_y

    @property
    def overhang(self) -> float:
        """How much wider the picture is than its nodes. The number that changed."""
        return self.width - self.node_width

    def summary(self) -> str:
        return (
            f"extent {self.width:.3f} x {self.height:.3f} cm "
            f"(nodes {self.node_width:.3f} x {self.node_height:.3f}, "
            f"overhang {self.overhang:+.3f})"
        )


def extent(model: Model, layout: Layout, style: DiagramStyle | None = None) -> Extent:
    """The drawing's bounding box, and which labels reach outside the node span.

    Both back ends are measured by this one function: they place labels from the same
    :func:`~pathmgr.render.placement.place_labels` result and draw the same loops, so their extents
    agree up to how TeX versus matplotlib typesets a label -- which is the estimator's error, not a
    difference between the back ends. One number, and the caveat about it, rather than two.
    """
    from .placement import (
        DEFAULT_LOOP_DIRECTION,
        DEFAULT_LOOP_LOOSENESS,
        loop_path,
    )

    style = style or DiagramStyle()
    layout = layout.completed(model)
    coding = style.coefficient_coding(model)
    edges = labelled_edges(model, style, coding)
    placements = place_labels(model, layout, style, edges)
    texts = {key: text for key, text, _bow, _kind in edges}

    node_xs: list[float] = []
    node_ys: list[float] = []
    for name in model.names:
        if name not in layout:
            continue
        rect = node_rect(name, model, layout, style)
        x, y = layout[name]
        node_xs += [x - rect.width / 2, x + rect.width / 2]
        node_ys += [y - rect.height / 2, y + rect.height / 2]
    if not node_xs:
        return Extent(0, 0, 0, 0, 0, 0, 0, 0)

    span = (min(node_xs), min(node_ys), max(node_xs), max(node_ys))
    xs, ys = list(node_xs), list(node_ys)
    outside: list[tuple[tuple[str, str], float, str]] = []

    def note(key, lo_x, lo_y, hi_x, hi_y) -> None:
        xs.extend((lo_x, hi_x))
        ys.extend((lo_y, hi_y))
        for over, side in (
            (span[0] - lo_x, "left"),
            (hi_x - span[2], "right"),
            (span[1] - lo_y, "below"),
            (hi_y - span[3], "above"),
        ):
            if over > 1e-9:
                outside.append((key, over, side))

    for key, placement in placements.items():
        text = texts.get(key)
        if not text:
            continue
        rect = label_rect(placement.point, text, style)
        left, bottom, right, top = rect.bounds
        note(key, left, bottom, right, top)
        if placement.leader_to is not None:
            lx, ly = placement.leader_to
            note(key, lx, ly, lx, ly)

    # variance loops reach beyond their node whether or not they carry a label
    loop_choices = {
        key[0]: (p.loop_direction, p.loop_looseness)
        for key, p in placements.items()
        if p.loop_direction is not None
    }
    for edge in model.bidirected_edges:
        if not edge.is_variance or edge.a not in layout:
            continue
        if not style.draws_variance(False):
            continue
        direction, looseness = loop_choices.get(
            edge.a, (DEFAULT_LOOP_DIRECTION, DEFAULT_LOOP_LOOSENESS)
        )
        arc = loop_path(layout[edge.a], node_rect(edge.a, model, layout, style), direction, looseness)
        xs.extend(point[0] for point in arc)
        ys.extend(point[1] for point in arc)

    return Extent(
        min_x=min(xs),
        min_y=min(ys),
        max_x=max(xs),
        max_y=max(ys),
        node_min_x=span[0],
        node_min_y=span[1],
        node_max_x=span[2],
        node_max_y=span[3],
        outside=tuple(sorted(outside, key=lambda item: (-item[1], item[0], item[2]))),
    )


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
    #: how big the picture is, and what sticks out of it
    extent: Extent | None = None
    #: budget overshoots, as ``(axis, limit, actual)`` -- empty when no budget was given or it holds
    over_budget: tuple[tuple[str, float, float], ...] = ()

    @property
    def ok(self) -> bool:
        """True if there is nothing outstanding. What a figure generator asserts on.

        **Advisories deliberately do not affect this.** They are not defects, and a generator that
        asserts ``.ok`` must not start failing because a figure has two identically-labelled edges
        placed differently -- which is often the *correct* outcome.
        """
        return (
            not self.collisions.collisions
            and not self.collisions.ambiguous
            and not self.crossings
            and not self.model_issues
            and not self.over_budget
        )

    def summary(self) -> str:
        size = f"  {self.extent.summary()}" if self.extent is not None else ""
        budget = f"  OVER BUDGET x{len(self.over_budget)}" if self.over_budget else ""
        return (
            f"{self.collisions.summary()}  crossings={len(self.crossings)}  "
            f"model={len(self.model_issues)}  advisories={len(self.advisories)}{size}{budget}"
        )

    def __str__(self) -> str:
        lines = [self.summary(), *str(self.collisions).splitlines()[1:]]
        for crossing in self.crossings:
            lines.append(f"  crossing: {crossing}")
        for issue in self.model_issues:
            lines.append(f"  model: {issue}")
        for advisory in self.advisories:
            lines.append(f"  advisory: {advisory}")
        for axis, limit, actual in self.over_budget:
            lines.append(
                f"  over budget: {axis} {actual:.3f}cm exceeds {limit:.3f}cm "
                f"by {actual - limit:.3f}cm"
            )
        if self.extent is not None:
            for key, over, side in self.extent.outside[:8]:
                lines.append(
                    f"  outside the node span: label[{key[0]}--{key[1]}] "
                    f"{over:.3f}cm {side}"
                )
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
    max_width: float | None = None,
    max_height: float | None = None,
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

    ``max_width`` / ``max_height`` declare a size budget in cm -- for a figure that has to fit a
    page. Exceeding it makes the report not ``ok`` and names **which** labels are responsible, so
    the caller knows which to override. The placer is deliberately **not** made to optimise under
    the budget: placing labels under a width constraint is a much larger problem and would trade
    away collision-freedom nobody asked to lose. This reports; the human decides, as with the other
    advisories. Note the extent is estimated -- see :class:`Extent`.
    """
    from .placement import edge_node_crossings

    style = style or DiagramStyle()
    layout = (layout or Layout()).completed(model)
    coding = style.coefficient_coding(model)
    edges = labelled_edges(model, style, coding)
    placements = place_labels(model, layout, style, edges)
    texts = {key: text for key, text, _bow, _kind in edges}
    size = extent(model, layout, style)
    over: list[tuple[str, float, float]] = []
    if max_width is not None and size.width > max_width:
        over.append(("width", max_width, size.width))
    if max_height is not None and size.height > max_height:
        over.append(("height", max_height, size.height))
    return Diagnosis(
        collisions=collision_report(model, layout, style),
        crossings=tuple(edge_node_crossings(model, layout, style)),
        model_issues=_model_issues(model),
        advisories=_consistency_advisories(model, layout, style, placements, texts),
        extent=size,
        over_budget=tuple(over),
    )
