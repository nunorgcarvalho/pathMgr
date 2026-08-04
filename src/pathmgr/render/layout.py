"""Node placement for a path diagram.

Deliberately scoped. **Explicit coordinates are the reliable path** and the one that matters:
pedigrees -- the main use case -- have an obvious natural layout (generations as rows,
individuals as columns) which the pedigree builder can hand over directly. The automatic
fallback exists so an arbitrary model renders at all; the bar for it is "legible and correct",
not beautiful.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.model import Model

__all__ = ["Layout", "layered_layout"]


@dataclass
class Layout:
    """Node positions in diagram coordinates, plus the spacing used to derive them.

    ``positions`` maps a variable name to ``(x, y)``. Anything missing is filled in by
    :func:`layered_layout` when the layout is completed against a model, so partial
    hand-placement works: pin the nodes you care about and let the rest fall where they may.
    """

    positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: horizontal gap between adjacent nodes in a row, in TikZ units (cm)
    x_gap: float = 2.2
    #: vertical gap between layers
    y_gap: float = 1.6

    def __getitem__(self, name: str) -> tuple[float, float]:
        return self.positions[name]

    def __contains__(self, name: object) -> bool:
        return name in self.positions

    def get(self, name: str, default=None):
        return self.positions.get(name, default)

    def completed(self, model: Model) -> "Layout":
        """This layout with any unplaced variable filled in by the automatic fallback."""
        missing = [n for n in model.names if n not in self.positions]
        if not missing:
            return self
        auto = layered_layout(model, x_gap=self.x_gap, y_gap=self.y_gap)
        merged = dict(auto.positions)
        merged.update(self.positions)  # explicit placement always wins
        return Layout(merged, x_gap=self.x_gap, y_gap=self.y_gap)

    def bounds(self) -> tuple[float, float, float, float]:
        """``(min_x, min_y, max_x, max_y)``. Zeros for an empty layout."""
        if not self.positions:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in self.positions.values()]
        ys = [p[1] for p in self.positions.values()]
        return (min(xs), min(ys), max(xs), max(ys))

    def scaled(self, factor: float) -> "Layout":
        return Layout(
            {n: (x * factor, y * factor) for n, (x, y) in self.positions.items()},
            x_gap=self.x_gap * factor,
            y_gap=self.y_gap * factor,
        )


def _depths(model: Model) -> dict[str, int]:
    """Longest directed path from any root to each node, so parents sit above children.

    Falls back to insertion order for nodes inside a cycle, which cannot have a well-defined
    depth -- a diagram of a feedback model is still worth drawing.
    """
    depth: dict[str, int] = {}
    order = model.names

    if model.is_recursive:
        # process in topological order so every parent is resolved before its child
        remaining = {n: set(model.parents(n)) for n in order}
        ready = [n for n in order if not remaining[n]]
        seen: set[str] = set()
        while ready:
            node = ready.pop(0)
            if node in seen:
                continue
            seen.add(node)
            parents = model.parents(node)
            depth[node] = 0 if not parents else 1 + max(depth[p] for p in parents)
            for child in model.children(node):
                remaining[child].discard(node)
                if not remaining[child] and child not in seen:
                    ready.append(child)
        for node in order:  # anything unreached (shouldn't happen in a DAG)
            depth.setdefault(node, 0)
        return depth

    # cyclic: approximate by iterating the same relation a bounded number of times
    depth = {n: 0 for n in order}
    for _ in range(len(order)):
        changed = False
        for node in order:
            parents = model.parents(node)
            want = 0 if not parents else 1 + max(depth[p] for p in parents)
            if want > depth[node] and want < len(order):
                depth[node] = want
                changed = True
        if not changed:
            break
    return depth


def layered_layout(model: Model, x_gap: float = 2.2, y_gap: float = 1.6) -> Layout:
    """Place nodes in rows by causal depth: roots on top, each node below its parents.

    Within a row, nodes keep the model's insertion order and rows are centred against each
    other. Latent variables are nudged slightly left of observed ones at the same depth, which
    keeps ``g``/``e`` from colliding with ``y`` in the common ``y = g + e`` motif.
    """
    depth = _depths(model)
    rows: dict[int, list[str]] = {}
    for name in model.names:
        rows.setdefault(depth[name], []).append(name)

    _order_rows_by_barycentre(model, rows)

    widest = max((len(r) for r in rows.values()), default=1)
    positions: dict[str, tuple[float, float]] = {}
    for layer, names in sorted(rows.items()):
        offset = (widest - len(names)) / 2.0
        for i, name in enumerate(names):
            nudge = -0.18 * x_gap if model.var(name).latent else 0.0
            positions[name] = ((i + offset) * x_gap + nudge, -layer * y_gap)
    return Layout(positions, x_gap=x_gap, y_gap=y_gap)


def _order_rows_by_barycentre(model: Model, rows: dict[int, list[str]], sweeps: int = 4) -> None:
    """Reorder within each row toward the mean position of each node's neighbours, in place.

    The standard cheap fix for edge crossings in a layered drawing. A few alternating sweeps
    is all this gets: the bar for automatic layout here is "legible and correct", and a pedigree
    that really matters supplies its own coordinates. Ties keep the previous order, so the result
    is deterministic.
    """
    if len(rows) < 2:
        return
    index = {name: i for layer in rows.values() for i, name in enumerate(layer)}

    def barycentre(name: str, neighbours: tuple[str, ...]) -> float:
        known = [index[n] for n in neighbours if n in index]
        return sum(known) / len(known) if known else float(index[name])

    layers = sorted(rows)
    for sweep in range(sweeps):
        downward = sweep % 2 == 0
        sequence = layers[1:] if downward else layers[-2::-1]
        for layer in sequence:
            neighbours_of = model.parents if downward else model.children
            rows[layer].sort(key=lambda n: (barycentre(n, neighbours_of(n)), n))
            for i, name in enumerate(rows[layer]):
                index[name] = i
    return None


def pedigree_layout(
    generations: dict[str, int],
    columns: dict[str, float] | None = None,
    x_gap: float = 2.2,
    y_gap: float = 1.6,
) -> Layout:
    """Build a layout from an explicit generation (row) and optional column per variable.

    The hand-off point for a pedigree builder: it knows which generation each individual is in,
    which is the only thing a pedigree diagram really needs to look right.

    >>> layout = pedigree_layout({"y_m": 0, "y_f": 0, "y_o": 1}, {"y_m": 0, "y_f": 1, "y_o": 0.5})
    >>> layout["y_o"]
    (1.1, -1.6)
    """
    columns = columns or {}
    counters: dict[int, int] = {}
    positions: dict[str, tuple[float, float]] = {}
    for name, generation in generations.items():
        if name in columns:
            column = columns[name]
        else:
            column = counters.get(generation, 0)
            counters[generation] = column + 1
        positions[name] = (column * x_gap, -generation * y_gap)
    return Layout(positions, x_gap=x_gap, y_gap=y_gap)
