"""Wright path tracing: covariance by explicit chain enumeration.

The second, independent engine. :mod:`pathmgr.core.ram` gives the right answer but hides the
reasoning; this module returns the *decomposition* -- which chains exist and what each
contributes -- because that is the part that goes into a writeup.

The standing correctness property of pathMgr is that the two engines agree symbolically on
every model. See ``tests/test_agreement.py``.

What a chain is here, and how it relates to the rules in the textbooks
----------------------------------------------------------------------
A chain from ``x`` to ``y`` is

    x <- ... <- u   <->   v -> ... -> y

a directed path traced **backward** from ``x`` to some ancestor ``u``, **exactly one**
bidirected edge ``u <-> v`` (possibly ``u == v``, which is a variance), then a directed path
traced **forward** from ``v`` to ``y``. Its value is the product of the directed coefficients
on both legs times the bidirected edge's value, and the covariance is the sum over all chains.
This is exactly the RAM identity written out term by term: ``Sigma = B S B^T`` with
``B = (I - A)^-1 = sum_k A^k``, so ``Sigma[x, y] = sum_{u,v} B[x,u] S[u,v] B[y,v]`` and
``B[x,u]`` is the sum over directed paths from ``u`` to ``x`` of their coefficient products.

**One classical rule is deliberately not enforced: "no variable may appear twice in a chain."**
It belongs to the *standardized* formulation of Wright's rules and is wrong here. In that
formulation a chain may turn around at any variable, implicitly using ``Var = 1``, and tracing
stops at exogenous variables whose correlations are read off directly. In the RAM formulation
the ``S`` entries are **disturbance** (co)variances, so a turning point must be written
explicitly as ``u <-> u``, and for an endogenous variable that captures only its residual --
the rest of its variance arrives via chains that continue back to its ancestors. Those chains
necessarily visit the turning-point variable in both legs. Concretely, for ``w`` with parents
``b`` and ``c``, and ``w -> x``, ``w -> y``::

    Cov[x, y] = q r Var[w]
              = q r (Var[eps_w] + p_b^2 V_b + p_c^2 V_c + 2 p_b p_c C_bc)

The last three terms are the chains ``x <- w <- b <-> b -> w -> y`` and friends, each of which
passes through ``w`` twice. Dropping them would lose ``Var[w]``'s ancestral part entirely. So
a node may repeat *across* the two legs; it cannot repeat *within* a leg, which in an acyclic
model is automatic. :attr:`Chain.revisits` reports the nodes that appear in both legs, since
they are the ones a reader hand-checking against a textbook will query.

Limits
------
- **Cyclic models cannot be traced.** A feedback loop has infinitely many chains; the matrix
  inverse sums that geometric series in closed form and enumeration cannot. Detected up front
  and raised as :class:`UntraceableModelError`, pointing at :class:`pathmgr.RAMEngine` -- never
  silently truncated.
- The number of chains is the product of the two legs' path counts and can grow
  combinatorially. :class:`WrightTracer` caps it with ``max_chains`` and raises
  :class:`ChainLimitError` rather than appearing to hang.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from .model import Model
from .units import Units

__all__ = [
    "Chain",
    "ChainLimitError",
    "Decomposition",
    "UntraceableModelError",
    "WrightTracer",
]

BACKWARD = " <- "
FORWARD = " -> "
BIDIRECTED = " <-> "


class UntraceableModelError(ValueError):
    """This model cannot be traced by enumeration -- use the RAM engine instead."""


class ChainLimitError(ValueError):
    """Enumeration exceeded ``max_chains``; the total would be incomplete."""


@dataclass(frozen=True)
class Chain:
    """One Wright chain: ``x <- ... <- u  <->  v -> ... -> y`` and what it contributes."""

    #: nodes from x back to the bidirected edge, i.e. ``(x, ..., u)``
    backward: tuple[str, ...]
    #: nodes from the bidirected edge forward to y, i.e. ``(v, ..., y)``
    forward: tuple[str, ...]
    #: the value of the bidirected edge ``u <-> v`` (a variance when ``u == v``)
    bidirected_value: sp.Expr
    #: every factor in chain order; their product is :attr:`contribution`
    factors: tuple[sp.Expr, ...]
    #: the chain's symbolic contribution to the covariance
    contribution: sp.Expr

    @property
    def x(self) -> str:
        return self.backward[0]

    @property
    def y(self) -> str:
        return self.forward[-1]

    @property
    def pivot(self) -> tuple[str, str]:
        """The bidirected edge ``(u, v)`` the chain passes through."""
        return (self.backward[-1], self.forward[0])

    @property
    def is_variance_pivot(self) -> bool:
        """True if the chain turns around at a single variable's variance (``u == v``)."""
        return self.backward[-1] == self.forward[0]

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every node the chain visits, in order (the pivot node once if ``u == v``)."""
        if self.is_variance_pivot:
            return self.backward + self.forward[1:]
        return self.backward + self.forward

    @property
    def revisits(self) -> tuple[str, ...]:
        """Nodes appearing in BOTH legs -- legitimate here; see the module docstring."""
        tail = self.forward[1:] if self.is_variance_pivot else self.forward
        return tuple(sorted(set(self.backward) & set(tail)))

    @property
    def length(self) -> int:
        """Number of directed edges traversed (the bidirected edge is not counted)."""
        return (len(self.backward) - 1) + (len(self.forward) - 1)

    def directed_edges(self) -> tuple[tuple[str, str], ...]:
        """The directed edges used, each as ``(src, dst)``. For diagram highlighting."""
        back = tuple(
            (self.backward[i + 1], self.backward[i]) for i in range(len(self.backward) - 1)
        )
        fwd = tuple((self.forward[i], self.forward[i + 1]) for i in range(len(self.forward) - 1))
        return back + fwd

    def path_string(self) -> str:
        """``'y_i <- g_i <-> g_j -> y_j'`` -- the chain as a readable trace."""
        left = BACKWARD.join(self.backward)
        right = FORWARD.join(self.forward)
        return f"{left}{BIDIRECTED}{right}"

    def factor_string(self) -> str:
        """``'1/2 * V_A * 1/2'`` -- the contribution before multiplying out."""
        return " * ".join(sp.sstr(f) for f in self.factors)

    def tex_path(self, labels: dict[str, str] | None = None) -> str:
        """The chain as math-mode LaTeX, using each variable's label where it has one."""
        labels = labels or {}
        left = r" \leftarrow ".join(_tex_name(n, labels) for n in self.backward)
        right = r" \rightarrow ".join(_tex_name(n, labels) for n in self.forward)
        return f"{left} \\leftrightarrow {right}"

    def __str__(self) -> str:
        return f"{self.path_string()}   =  {sp.sstr(self.contribution)}"


@dataclass
class Decomposition:
    """The itemized chains between two variables, plus their total.

    The total is what the RAM engine also computes; the itemized list is the point.
    """

    x: str
    y: str
    chains: tuple[Chain, ...]
    units: Units
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def total(self) -> sp.Expr:
        """Sum of every chain's contribution, expanded (the form the RAM engine returns)."""
        return sp.expand(sum((c.contribution for c in self.chains), sp.Integer(0)))

    def __len__(self) -> int:
        return len(self.chains)

    def __iter__(self):
        return iter(self.chains)

    def sorted_chains(self) -> tuple[Chain, ...]:
        """Chains shortest-first, then alphabetically -- a stable order for a document."""
        return tuple(sorted(self.chains, key=lambda c: (c.length, c.path_string())))

    def __str__(self) -> str:
        header = f"Cov[{self.x}, {self.y}]  ({len(self.chains)} chain"
        header += "s" if len(self.chains) != 1 else ""
        header += f", {self.units})"
        if not self.chains:
            return header + "\n  (no chains -- the covariance is 0)\n  total = 0"
        rows = []
        for chain in self.sorted_chains():
            note = ""
            if chain.revisits:
                note = f"   [revisits {', '.join(chain.revisits)}]"
            rows.append((chain.path_string(), chain.factor_string(), note))
        width = max(len(r[0]) for r in rows)
        lines = [header]
        for path, factors, note in rows:
            lines.append(f"  {path:<{width}}  =  {factors}{note}")
        lines.append(f"  {'total':<{width}}  =  {sp.sstr(self.total)}")
        return "\n".join(lines)

    # -- LaTeX ------------------------------------------------------------------------
    def to_latex(self, style: str = "align", total_form: str = "factored") -> str:
        """LaTeX for a writeup. ``style`` is ``"align"`` or ``"tabular"``.

        Uses each variable's ``Variable.label`` when it has one, so the symbols match the
        document rather than the internal names.
        """
        if style not in ("align", "tabular"):
            raise ValueError(f"style must be 'align' or 'tabular', got {style!r}")
        total = {
            "factored": sp.factor,
            "expanded": sp.expand,
            "simplified": sp.simplify,
            "raw": lambda e: e,
        }
        if total_form not in total:
            raise ValueError(f"total_form must be one of {sorted(total)}, got {total_form!r}")
        total_expr = total[total_form](self.total)

        lhs = rf"\operatorname{{Cov}}\left[{self._tex(self.x)}, {self._tex(self.y)}\right]"
        if style == "tabular":
            return self._tabular(lhs, total_expr)
        return self._align(lhs, total_expr)

    def _align(self, lhs: str, total_expr: sp.Expr) -> str:
        if not self.chains:
            return "\\begin{align*}\n" f"  {lhs} &= 0\n" "\\end{align*}"
        lines = ["\\begin{align*}", f"  {lhs}"]
        for i, chain in enumerate(self.sorted_chains()):
            sign = "&= " if i == 0 else "&\\quad + "
            lines.append(
                f"  {sign}{sp.latex(chain.contribution)}"
                f" && \\text{{[}}{chain.tex_path(self.labels)}\\text{{]}} \\\\"
            )
        lines.append(f"  &= {sp.latex(total_expr)}")
        lines.append("\\end{align*}")
        return "\n".join(lines)

    def _tabular(self, lhs: str, total_expr: sp.Expr) -> str:
        lines = [
            "\\begin{tabular}{ll}",
            "  \\hline",
            "  chain & contribution \\\\",
            "  \\hline",
        ]
        for chain in self.sorted_chains():
            lines.append(
                f"  ${chain.tex_path(self.labels)}$ & ${sp.latex(chain.contribution)}$ \\\\"
            )
        lines += [
            "  \\hline",
            f"  ${lhs}$ & ${sp.latex(total_expr)}$ \\\\",
            "  \\hline",
            "\\end{tabular}",
        ]
        return "\n".join(lines)

    def _tex(self, name: str) -> str:
        return _tex_name(name, self.labels)


def _tex_name(name: str, labels: dict[str, str]) -> str:
    """A variable's math-mode LaTeX: its label if it has one, else a sanitized name."""
    label = labels.get(name)
    if label:
        stripped = label.strip()
        if stripped.startswith("$") and stripped.endswith("$"):
            return stripped[1:-1]
        return stripped
    return sp.latex(sp.Symbol(name))


class WrightTracer:
    """Covariances by chain enumeration, returning the decomposition rather than a total.

    >>> import pathmgr as pm
    >>> m = pm.from_text("m ~ a*x\\ny ~ b*m + c*x\\nx ~~ V_x*x")
    >>> d = pm.WrightTracer(m).trace("x", "y")
    >>> len(d)
    2
    >>> d.total
    V_x*a*b + V_x*c
    """

    #: refuse to enumerate more than this many chains for one query
    DEFAULT_MAX_CHAINS = 20_000

    def __init__(self, model: Model, max_chains: int | None = None):
        self.model = model
        self.max_chains = self.DEFAULT_MAX_CHAINS if max_chains is None else max_chains

    @property
    def units(self) -> Units:
        return self.model.units

    # -- the tracer -------------------------------------------------------------------
    def trace(self, x: str, y: str) -> Decomposition:
        """Every Wright chain between ``x`` and ``y``, with its contribution."""
        self._require_traceable()
        self._require(x, y)

        back_legs = self._backward_paths(x)
        fwd_legs = self._backward_paths(y)  # same walk; reversed to become forward legs

        chains: list[Chain] = []
        for edge in self.model.bidirected_edges:
            if edge.value == 0:
                continue
            # S is symmetric, so an off-diagonal edge is traversed in both directions;
            # a variance (a == b) is traversed once.
            orientations = {(edge.a, edge.b), (edge.b, edge.a)}
            for u, v in sorted(orientations):
                for back_nodes, back_coeff in back_legs.get(u, ()):
                    for fwd_nodes, fwd_coeff in fwd_legs.get(v, ()):
                        forward = tuple(reversed(fwd_nodes))
                        factors = self._factors(back_nodes, edge.value, forward)
                        chains.append(
                            Chain(
                                backward=back_nodes,
                                forward=forward,
                                bidirected_value=edge.value,
                                factors=factors,
                                contribution=sp.expand(back_coeff * edge.value * fwd_coeff),
                            )
                        )
                        if len(chains) > self.max_chains:
                            raise ChainLimitError(
                                f"more than {self.max_chains} chains between {x!r} and {y!r}; "
                                f"the enumeration would be incomplete. Raise max_chains, or "
                                f"use RAMEngine, which does not enumerate."
                            )
        labels = {v.name: v.label for v in self.model.variables if v.label}
        return Decomposition(x=x, y=y, chains=tuple(chains), units=self.units, labels=labels)

    def cov(self, x: str, y: str, form: str = "expanded") -> sp.Expr:
        """The traced total -- directly comparable with :meth:`pathmgr.RAMEngine.cov`."""
        total = self.trace(x, y).total
        if form == "raw" or form == "expanded":
            return sp.expand(total)
        if form == "simplified":
            return sp.simplify(total)
        if form == "factored":
            return sp.factor(total)
        raise ValueError(f"unknown form {form!r}")

    def var(self, x: str, form: str = "expanded") -> sp.Expr:
        """A variable's own variance: the same enumeration with both endpoints equal."""
        return self.cov(x, x, form=form)

    # -- internals --------------------------------------------------------------------
    def _factors(
        self, backward: tuple[str, ...], bidirected: sp.Expr, forward: tuple[str, ...]
    ) -> tuple[sp.Expr, ...]:
        """Every factor in chain order, so a reader can see the product before it collapses."""
        factors: list[sp.Expr] = []
        for i in range(len(backward) - 1):
            factors.append(self.model.path_coeff(backward[i + 1], backward[i]))
        factors.append(bidirected)
        for i in range(len(forward) - 1):
            factors.append(self.model.path_coeff(forward[i], forward[i + 1]))
        return tuple(factors)

    def _backward_paths(
        self, start: str
    ) -> dict[str, tuple[tuple[tuple[str, ...], sp.Expr], ...]]:
        """All directed paths ending at ``start``, keyed by the ancestor they begin at.

        Each value is ``(nodes, coefficient product)`` with ``nodes`` running from ``start``
        back to the ancestor. The zero-length path (``start`` itself, product 1) is included:
        a chain may pivot on ``start``'s own disturbance.
        """
        found: dict[str, list[tuple[tuple[str, ...], sp.Expr]]] = {}
        count = 0

        def walk(node: str, path: list[str], product: sp.Expr) -> None:
            nonlocal count
            found.setdefault(node, []).append((tuple(path), product))
            count += 1
            if count > self.max_chains:
                raise ChainLimitError(
                    f"more than {self.max_chains} directed paths lead into {start!r}; "
                    f"enumeration would be incomplete. Raise max_chains, or use RAMEngine."
                )
            for parent in self.model.parents(node):
                if parent in path:  # impossible in a DAG; cheap insurance
                    continue
                coeff = self.model.path_coeff(parent, node)
                path.append(parent)
                walk(parent, path, product * coeff)
                path.pop()

        walk(start, [start], sp.Integer(1))
        return {node: tuple(paths) for node, paths in found.items()}

    def _require_traceable(self) -> None:
        if not self.model.is_recursive:
            cycles = "; ".join(" -> ".join(c) for c in self.model.cycles())
            raise UntraceableModelError(
                "chain enumeration cannot handle a model with a directed cycle: a feedback "
                "loop has infinitely many Wright chains, and truncating them would silently "
                "give a wrong answer. Use RAMEngine, which sums the geometric series in "
                f"closed form via (I - A)^-1. Cycles found: {cycles}"
            )

    def _require(self, *names: str) -> None:
        for name in names:
            if not self.model.has_var(name):
                raise KeyError(
                    f"unknown variable {name!r}; the model has: {', '.join(self.model.names)}"
                )

    def __repr__(self) -> str:
        return f"<WrightTracer over {self.model!r} max_chains={self.max_chains}>"
