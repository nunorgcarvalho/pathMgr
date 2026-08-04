"""The model-specification object: variables, directed paths, bidirected covariances.

This is the interface every engine (RAM, Wright tracing) and renderer consumes. It says
what the model *is*; it computes nothing beyond the RAM matrices that are a direct
transcription of the edges.

Conventions
-----------
- A **directed edge** ``a -> b`` with coefficient ``c`` means ``b`` regresses on ``a`` with
  coefficient ``c``. It becomes ``A[b, a] = c`` in the RAM asymmetric matrix.
- A **bidirected edge** ``a <-> b`` with value ``v`` is a *disturbance* covariance: it
  becomes ``S[a, b] = S[b, a] = v``. On an exogenous variable, ``a <-> a`` is its variance.
  On an endogenous variable it is the residual variance -- **not** the total variance, which
  the model implies rather than states.
- **Latent vs observed** matters twice over: the RAM filter matrix ``F`` selects observed
  rows, and diagrams draw latents as circles and observed variables as boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from .symbols import SymbolRegistry
from .units import Units


@dataclass(frozen=True)
class Variable:
    """A node in the path diagram."""

    name: str
    latent: bool = False
    label: str | None = None  # rendering label, e.g. r"$g_i$"; defaults to the name

    @property
    def observed(self) -> bool:
        return not self.latent

    def display(self) -> str:
        return self.label if self.label is not None else self.name


@dataclass(frozen=True)
class DirectedEdge:
    """``src -> dst`` with coefficient ``coeff``."""

    src: str
    dst: str
    coeff: sp.Expr

    def __str__(self) -> str:
        return f"{self.src} -> {self.dst}  [{self.coeff}]"


@dataclass(frozen=True)
class BidirectedEdge:
    """``a <-> b`` with disturbance covariance ``value``. ``a == b`` is a variance."""

    a: str
    b: str
    value: sp.Expr

    @property
    def is_variance(self) -> bool:
        return self.a == self.b

    def __str__(self) -> str:
        arrow = "<->"
        return f"{self.a} {arrow} {self.b}  [{self.value}]"


@dataclass
class ModelIssue:
    """One problem found by :meth:`Model.validate`."""

    severity: str  # "error" | "warning"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.message}"


class Model:
    """A path model: a mutable builder that engines consume.

    Mutable-with-``copy()`` rather than persistent/immutable, because the driving use case
    builds one growing graph -- a pedigree unrolled generation by generation -- by
    monotone accumulation. ``revision`` increments on every structural change so engines
    can key caches (the symbolic ``(I - A)^-1`` is expensive) without silently going stale.
    """

    def __init__(self, name: str | None = None, units: Units | None = None):
        self.name = name
        self.units = units if units is not None else Units.unstandardized()
        self._vars: dict[str, Variable] = {}
        self._directed: dict[tuple[str, str], DirectedEdge] = {}
        self._bidirected: dict[tuple[str, str], BidirectedEdge] = {}
        self.symbols = SymbolRegistry()
        self._assumptions: list[sp.Eq] = []
        self._revision = 0

    # -- symbols ----------------------------------------------------------------------
    def declare(self, name: str, **assumptions: bool) -> sp.Symbol:
        """Declare a symbol with explicit sympy assumptions before first use.

        e.g. ``m.declare("V_A", positive=True)`` so ``sqrt`` and ``simplify`` can act.
        """
        return self.symbols.declare(name, **assumptions)

    def sym(self, name: str) -> sp.Symbol:
        """The Symbol for ``name`` (created with default assumptions if new)."""
        return self.symbols.get(name)

    def expr(self, value) -> sp.Expr:
        """Parse ``value`` into a sympy expression over this model's symbols."""
        return self.symbols.parse(value)

    # -- variables --------------------------------------------------------------------
    def add_var(
        self, name: str, latent: bool = False, label: str | None = None
    ) -> "Model":
        """Add a variable. Observed by default; ``latent=True`` for unobserved nodes."""
        if name in self._vars:
            raise ValueError(f"variable {name!r} already in model")
        if not name or not isinstance(name, str):
            raise ValueError(f"variable name must be a non-empty string, got {name!r}")
        self._vars[name] = Variable(name, latent=latent, label=label)
        self._touch()
        return self

    def add_vars(self, *names: str, latent: bool = False) -> "Model":
        """Add several variables sharing the same latent/observed status."""
        for n in names:
            self.add_var(n, latent=latent)
        return self

    def has_var(self, name: str) -> bool:
        return name in self._vars

    def var(self, name: str) -> Variable:
        self._require(name)
        return self._vars[name]

    # -- edges ------------------------------------------------------------------------
    def add_path(self, src: str, dst: str, coeff=1) -> "Model":
        """Add a directed edge ``src -> dst`` with coefficient ``coeff``.

        ``coeff`` may be a number (``1``, ``sympy.Rational(1, 2)``), a symbol name
        (``"b1"``), or an expression (``"(1 + rho_g)/2"``). Defaults to 1, the common case
        for a decomposition like ``y = g + e``.
        """
        self._require(src, dst)
        if src == dst:
            raise ValueError(
                f"self-loop {src!r} -> {src!r} is not a path; a variable's own variance is "
                f"a bidirected edge: add_cov({src!r}, {src!r}, ...)"
            )
        if (src, dst) in self._directed:
            raise ValueError(
                f"path {src!r} -> {dst!r} already specified with coefficient "
                f"{self._directed[(src, dst)].coeff}; remove it first or combine the terms"
            )
        self._directed[(src, dst)] = DirectedEdge(src, dst, self.expr(coeff))
        self._touch()
        return self

    def add_cov(self, a: str, b: str, value) -> "Model":
        """Add a bidirected edge ``a <-> b`` with disturbance covariance ``value``.

        ``a == b`` states a variance. On an endogenous variable this is the *residual*
        variance, not the total.
        """
        self._require(a, b)
        key = self._cov_key(a, b)
        if key in self._bidirected:
            raise ValueError(
                f"covariance {a!r} <-> {b!r} already specified as "
                f"{self._bidirected[key].value}; remove it first"
            )
        self._bidirected[key] = BidirectedEdge(key[0], key[1], self.expr(value))
        self._touch()
        return self

    def add_variance(self, name: str, value) -> "Model":
        """Shorthand for ``add_cov(name, name, value)``."""
        return self.add_cov(name, name, value)

    def remove_path(self, src: str, dst: str) -> "Model":
        del self._directed[(src, dst)]
        self._touch()
        return self

    def remove_cov(self, a: str, b: str) -> "Model":
        del self._bidirected[self._cov_key(a, b)]
        self._touch()
        return self

    def path_coeff(self, src: str, dst: str) -> sp.Expr | None:
        edge = self._directed.get((src, dst))
        return None if edge is None else edge.coeff

    def cov_value(self, a: str, b: str) -> sp.Expr | None:
        edge = self._bidirected.get(self._cov_key(a, b))
        return None if edge is None else edge.value

    # -- side relations ---------------------------------------------------------------
    def assume(self, lhs, rhs=None) -> "Model":
        """Record a side relation that holds in this model, e.g. ``V_A + V_E = 1``.

        These are *not* edges -- they are constraints an engine may substitute into or
        solve against (``rho_g = rho_y * h2_eq``, ``V_K = V_A0 / 2``). Pass either a sympy
        ``Eq``, or a left- and right-hand side.
        """
        if rhs is None:
            eq = lhs if isinstance(lhs, sp.Eq) else sp.Eq(self.expr(lhs), 0)
        else:
            eq = sp.Eq(self.expr(lhs), self.expr(rhs))
        self._assumptions.append(eq)
        self._touch()
        return self

    @property
    def assumptions(self) -> tuple[sp.Eq, ...]:
        return tuple(self._assumptions)

    def substitutions(self) -> dict[sp.Symbol, sp.Expr]:
        """Those assumptions of the form ``Symbol = expr``, as a substitution dict."""
        subs: dict[sp.Symbol, sp.Expr] = {}
        for eq in self._assumptions:
            if isinstance(eq.lhs, sp.Symbol) and eq.lhs not in eq.rhs.free_symbols:
                subs[eq.lhs] = eq.rhs
        return subs

    # -- views ------------------------------------------------------------------------
    @property
    def variables(self) -> tuple[Variable, ...]:
        """All variables in insertion order -- the canonical order of the RAM matrices."""
        return tuple(self._vars.values())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._vars)

    @property
    def observed(self) -> tuple[str, ...]:
        return tuple(n for n, v in self._vars.items() if v.observed)

    @property
    def latent(self) -> tuple[str, ...]:
        return tuple(n for n, v in self._vars.items() if v.latent)

    @property
    def directed_edges(self) -> tuple[DirectedEdge, ...]:
        return tuple(self._directed.values())

    @property
    def bidirected_edges(self) -> tuple[BidirectedEdge, ...]:
        return tuple(self._bidirected.values())

    @property
    def revision(self) -> int:
        """Increments on every structural change; engines key their caches on it."""
        return self._revision

    def parents(self, name: str) -> tuple[str, ...]:
        self._require(name)
        return tuple(s for (s, d) in self._directed if d == name)

    def children(self, name: str) -> tuple[str, ...]:
        self._require(name)
        return tuple(d for (s, d) in self._directed if s == name)

    @property
    def exogenous(self) -> tuple[str, ...]:
        """Variables with no incoming directed edge."""
        return tuple(n for n in self._vars if not self.parents(n))

    @property
    def endogenous(self) -> tuple[str, ...]:
        return tuple(n for n in self._vars if self.parents(n))

    # -- RAM matrices (a transcription of the edges, not a solve) ----------------------
    def A_matrix(self) -> sp.Matrix:
        """Asymmetric (directed-path) matrix: ``A[dst, src] = coeff``."""
        order = self.names
        idx = {n: i for i, n in enumerate(order)}
        A = sp.zeros(len(order), len(order))
        for e in self._directed.values():
            A[idx[e.dst], idx[e.src]] = e.coeff
        return A

    def S_matrix(self) -> sp.Matrix:
        """Symmetric (disturbance covariance) matrix: ``S[a, b] = S[b, a] = value``."""
        order = self.names
        idx = {n: i for i, n in enumerate(order)}
        S = sp.zeros(len(order), len(order))
        for e in self._bidirected.values():
            S[idx[e.a], idx[e.b]] = e.value
            S[idx[e.b], idx[e.a]] = e.value
        return S

    def F_matrix(self) -> sp.Matrix:
        """Filter matrix selecting the observed rows, in insertion order."""
        order = self.names
        obs = self.observed
        F = sp.zeros(len(obs), len(order))
        idx = {n: i for i, n in enumerate(order)}
        for r, n in enumerate(obs):
            F[r, idx[n]] = 1
        return F

    def ram(self) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, tuple[str, ...]]:
        """``(A, S, F, order)`` -- everything an engine needs from the specification."""
        return self.A_matrix(), self.S_matrix(), self.F_matrix(), self.names

    # -- structure --------------------------------------------------------------------
    def cycles(self) -> list[list[str]]:
        """Directed cycles, if any. A recursive (acyclic) model has none."""
        found: list[list[str]] = []
        state: dict[str, int] = {n: 0 for n in self._vars}  # 0 new, 1 open, 2 closed

        def walk(node: str, stack: list[str]) -> None:
            state[node] = 1
            stack.append(node)
            for child in self.children(node):
                if state[child] == 1:
                    found.append(stack[stack.index(child):] + [child])
                elif state[child] == 0:
                    walk(child, stack)
            stack.pop()
            state[node] = 2

        for n in self._vars:
            if state[n] == 0:
                walk(n, [])
        return found

    @property
    def is_recursive(self) -> bool:
        """True if the directed part is acyclic (as every pedigree model is)."""
        return not self.cycles()

    def validate(self) -> list[ModelIssue]:
        """Cheap structural checks. Returns issues; does not raise."""
        issues: list[ModelIssue] = []
        if not self._vars:
            issues.append(ModelIssue("error", "model has no variables"))
        for n in self.exogenous:
            if self.cov_value(n, n) is None:
                issues.append(
                    ModelIssue(
                        "warning",
                        f"exogenous variable {n!r} has no variance; it will contribute "
                        f"nothing to any covariance. Add add_variance({n!r}, ...).",
                    )
                )
        for cyc in self.cycles():
            issues.append(
                ModelIssue("warning", "directed cycle: " + " -> ".join(cyc))
            )
        if self.units.is_standardized:
            for n in self.exogenous:
                v = self.cov_value(n, n)
                if v is not None and v == 0:
                    issues.append(
                        ModelIssue("error", f"{n!r} has zero variance in a standardized model")
                    )
        return issues

    # -- copying ----------------------------------------------------------------------
    def copy(self, name: str | None = None) -> "Model":
        """An independent copy, for branching a model without disturbing the original."""
        new = Model(name if name is not None else self.name, self.units)
        new._vars = dict(self._vars)
        new._directed = dict(self._directed)
        new._bidirected = dict(self._bidirected)
        new.symbols = self.symbols.copy()
        new._assumptions = list(self._assumptions)
        new._revision = self._revision
        return new

    # -- text front-end ---------------------------------------------------------------
    @classmethod
    def from_text(cls, text: str, name: str | None = None) -> "Model":
        """Build a model from the terse text grammar. See :mod:`pathmgr.core.text`."""
        from .text import from_text  # late import: text.py imports this module

        return from_text(text, name=name)

    def to_text(self, include_name: bool = True) -> str:
        """Render this model in the text grammar. Round-trips through :meth:`from_text`."""
        from .text import to_text

        return to_text(self, include_name=include_name)

    # -- display ----------------------------------------------------------------------
    def describe(self) -> str:
        """A human-readable dump of the specification."""
        head = f"Model({self.name!r})" if self.name else "Model"
        lines = [f"{head}  [{self.units}]  rev {self._revision}"]
        lines.append(f"  observed ({len(self.observed)}): {', '.join(self.observed) or '-'}")
        lines.append(f"  latent   ({len(self.latent)}): {', '.join(self.latent) or '-'}")
        lines.append(f"  paths ({len(self._directed)}):")
        lines += [f"    {e}" for e in self._directed.values()]
        lines.append(f"  covariances ({len(self._bidirected)}):")
        lines += [f"    {e}" for e in self._bidirected.values()]
        if self._assumptions:
            lines.append(f"  assumptions ({len(self._assumptions)}):")
            lines += [f"    {sp.pretty(eq, use_unicode=False)}" for eq in self._assumptions]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<Model {self.name!r} vars={len(self._vars)} "
            f"paths={len(self._directed)} covs={len(self._bidirected)} "
            f"units={self.units.kind} rev={self._revision}>"
        )

    # -- internals --------------------------------------------------------------------
    def _touch(self) -> None:
        self._revision += 1

    def _require(self, *names: str) -> None:
        for n in names:
            if n not in self._vars:
                raise KeyError(
                    f"unknown variable {n!r}; add it first with add_var. "
                    f"Known: {', '.join(self._vars) or '(none)'}"
                )

    @staticmethod
    def _cov_key(a: str, b: str) -> tuple[str, str]:
        """Canonical (order-independent) key for a bidirected edge."""
        return (a, b) if a <= b else (b, a)
