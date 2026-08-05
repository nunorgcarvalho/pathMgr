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


@dataclass(frozen=True)
class CoPath:
    """``a -- b``: covariance attributable to **matching**, not to a common cause.

    A third edge type, distinct from both arrows. Following Sunde et al. 2025 Nat Commun
    (Supplementary Note 1), a co-path "denotes covariance attributable to matching (e.g.
    assortative mating) where covariance is induced **without causing variance**". The
    consequence that makes it irreducible to a bidirected edge: matching "will induce
    correlations in all the causes of the variables that are matched", so the association
    propagates *backward* up the graph, which an ``S`` entry cannot do.

    ``coefficient`` is **not** the correlation. Sunde's Eq. (1) is
    ``Cov[a, b] = mu * Var[a] * Var[b]``, so a target correlation ``rho`` between unit-variance
    variables needs ``mu = rho``, but between variables of variance ``V_P`` it needs
    ``mu = rho / V_P`` -- generation-indexed under assortment, since ``V_P`` grows.

    ``process`` names the **mating process** this co-path belongs to. A chain may not use two
    co-paths from the same process (Sunde, Supplementary Note 3: "a chain cannot include
    multiple co-path coefficients stemming from the same mating process"). One couple may carry
    several co-paths -- Sunde use four per mating process for cross-trait assortment -- which is
    why the process is named separately from the endpoints.
    """

    a: str
    b: str
    coefficient: sp.Expr
    process: str

    def __str__(self) -> str:
        return f"{self.a} -- {self.b}  [{self.coefficient}]  (process {self.process})"


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
        self._copaths: dict[tuple[str, str, str], CoPath] = {}
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

    def add_copath(self, a: str, b: str, coefficient, process: str | None = None) -> "Model":
        """Add a co-path ``a -- b``: covariance from **matching**, not from a common cause.

        See :class:`CoPath`. ``coefficient`` is not the correlation -- for variables of variance
        ``V`` a target correlation ``rho`` needs ``rho / V``. ``process`` names the mating
        process; it defaults to the pair itself, which is right unless one couple carries
        several co-paths (cross-trait assortment), in which case name it explicitly so the
        "one co-path per mating process per chain" rule can see they belong together.
        """
        self._require(a, b)
        if a == b:
            raise ValueError(
                f"a co-path joins two variables; {a!r} -- {a!r} is not meaningful. A co-path "
                f"induces covariance without causing variance, so it has no self form."
            )
        first, second = (a, b) if a <= b else (b, a)
        resolved = process if process is not None else f"{first}--{second}"
        key = (first, second, resolved)
        if key in self._copaths:
            raise ValueError(
                f"co-path {a!r} -- {b!r} in process {resolved!r} already specified as "
                f"{self._copaths[key].coefficient}; remove it first"
            )
        self._copaths[key] = CoPath(first, second, self.expr(coefficient), resolved)
        self._touch()
        return self

    def copaths_between(self, a: str, b: str) -> tuple[CoPath, ...]:
        """Every co-path joining ``a`` and ``b``, whatever mating process each belongs to."""
        first, second = (a, b) if a <= b else (b, a)
        return tuple(
            edge for (x, y, _), edge in self._copaths.items() if (x, y) == (first, second)
        )

    def _one_copath(self, a: str, b: str, process: str | None) -> CoPath | None:
        """Resolve a co-path by endpoints, requiring a process only when it is ambiguous.

        Looking one up by its endpoints has to work without knowing the process name, since a
        caller who named the process at construction should not have to repeat it to find the
        edge again -- and the default name is derived from the endpoints anyway.
        """
        candidates = self.copaths_between(a, b)
        if process is not None:
            for edge in candidates:
                if edge.process == process:
                    return edge
            return None
        if not candidates:
            return None
        if len(candidates) > 1:
            raise ValueError(
                f"{a!r} and {b!r} are joined by {len(candidates)} co-paths, from processes "
                f"{sorted(c.process for c in candidates)}. Name the process to say which."
            )
        return candidates[0]

    def remove_copath(self, a: str, b: str, process: str | None = None) -> "Model":
        edge = self._one_copath(a, b, process)
        if edge is None:
            raise KeyError(
                f"no co-path between {a!r} and {b!r}"
                + (f" in process {process!r}" if process else "")
            )
        del self._copaths[(edge.a, edge.b, edge.process)]
        self._touch()
        return self

    def copath_value(self, a: str, b: str, process: str | None = None) -> sp.Expr | None:
        """The coefficient of the co-path joining ``a`` and ``b``, or None if there is none.

        ``process`` is needed only when the pair carries more than one co-path.
        """
        edge = self._one_copath(a, b, process)
        return None if edge is None else edge.coefficient

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

    def substitutions(self, solve_for: tuple[str, ...] | list[str] = ()) -> dict[sp.Symbol, sp.Expr]:
        """Assumptions as a substitution dict.

        By default this is only those assumptions already in ``Symbol = expr`` form, because
        those are the unambiguous ones -- ``rho_g = rho_y * h2_eq`` says what to replace.
        A relation like ``V_A + V_E = 1`` does not: it could be solved for either symbol, and
        picking one silently would be guessing. Name the symbols you want eliminated in
        ``solve_for`` to have such relations solved explicitly:

        >>> m = Model().assume("V_A + V_E", 1)
        >>> m.substitutions()                        # nothing unambiguous to do
        {}
        >>> m.substitutions(solve_for=["V_E"])       # doctest: +SKIP
        {V_E: 1 - V_A}
        """
        subs: dict[sp.Symbol, sp.Expr] = {}
        for eq in self._assumptions:
            if isinstance(eq.lhs, sp.Symbol) and eq.lhs not in eq.rhs.free_symbols:
                subs[eq.lhs] = eq.rhs
        for name in solve_for:
            symbol = self.sym(name)
            if symbol in subs:
                continue
            for eq in self._assumptions:
                if symbol not in eq.free_symbols:
                    continue
                solutions = sp.solve(eq, symbol)
                if len(solutions) == 1:
                    subs[symbol] = solutions[0]
                    break
            else:
                raise ValueError(
                    f"no assumption determines {name!r} uniquely; "
                    f"assumptions are: {[str(e) for e in self._assumptions]}"
                )
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
    def copaths(self) -> tuple[CoPath, ...]:
        return tuple(self._copaths.values())

    @property
    def has_copaths(self) -> bool:
        return bool(self._copaths)

    @property
    def mating_processes(self) -> tuple[str, ...]:
        """Distinct mating-process identifiers, in first-appearance order."""
        seen: dict[str, None] = {}
        for edge in self._copaths.values():
            seen.setdefault(edge.process, None)
        return tuple(seen)

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
        issues.extend(self._check_disturbance_covariances())
        issues.extend(self._check_copaths())
        if self.units.is_standardized:
            for n in self.exogenous:
                v = self.cov_value(n, n)
                if v is not None and v == 0:
                    issues.append(
                        ModelIssue("error", f"{n!r} has zero variance in a standardized model")
                    )
        return issues

    def _check_disturbance_covariances(self) -> list[ModelIssue]:
        """Flag off-diagonal bidirected edges that touch an endogenous variable.

        A bidirected edge is a covariance between *disturbances*, not between variables. On
        two exogenous variables those coincide, which is why the distinction is easy to miss.
        On an endogenous variable they do not, and there are two failure modes:

        - If the endogenous variable has **no** disturbance variance of its own, it is a
          deterministic function of its parents, so its disturbance is identically zero and
          cannot covary with anything. Stating that it does yields a Sigma that is not
          positive semi-definite -- an implied correlation above 1 -- with no other complaint.
          That is an **error**.
        - Otherwise the edge is meaningful but means the covariance of the *residual*, not
          the total covariance the notation suggests. That is a **warning**.

        The trap that motivated this: in an assortative-mating pedigree, mates' genetic values
        are correlated, but a child's genetic value is endogenous (it has parents). Writing
        that correlation as a plain bidirected edge silently produces an invalid Sigma. The
        assortment has to enter as a directed effect of the phenotype instead.
        """
        issues: list[ModelIssue] = []
        endogenous = set(self.endogenous)
        for edge in self._bidirected.values():
            if edge.is_variance or edge.value == 0:
                continue
            for name, other in ((edge.a, edge.b), (edge.b, edge.a)):
                if name not in endogenous:
                    continue
                own = self.cov_value(name, name)
                if own is None or own == 0:
                    issues.append(
                        ModelIssue(
                            "error",
                            f"{name!r} is endogenous with no disturbance variance, so its "
                            f"disturbance is identically zero and cannot covary with "
                            f"{other!r} -- yet '{edge.a} <-> {edge.b}' says it does. The "
                            f"implied covariance matrix is not positive semi-definite. "
                            f"Bidirected edges are DISTURBANCE covariances: to correlate a "
                            f"variable that has parents, add the association as a directed "
                            f"path, or give {name!r} a disturbance variance.",
                        )
                    )
                else:
                    issues.append(
                        ModelIssue(
                            "warning",
                            f"'{edge.a} <-> {edge.b}' involves the endogenous variable "
                            f"{name!r}, so it is the covariance of {name!r}'s DISTURBANCE "
                            f"with {other!r} -- not their total covariance, which the model "
                            f"implies and which will be larger.",
                        )
                    )
        return issues

    def _check_copaths(self) -> list[ModelIssue]:
        """Catch the two cheap ways a co-path and a bidirected edge get confused.

        The general case is undecidable from structure alone -- a co-path between two roots is
        exactly the legitimate founding-couple encoding, and looks identical to a bidirected
        edge that someone typed with the wrong operator. But two patterns are worth flagging:

        - **Both edge types on the same pair.** Almost always double-counting: the co-path
          already induces the covariance the bidirected edge is stating by hand. This is the
          mistake the superseded ``tests/fixtures/am_equilibrium_handwritten.pmg`` encoding
          made in reverse.
        - **A co-path onto a variable with no incoming paths and no variance.** A co-path
          contributes ``mu * Var[a] * Var[b]``, so if either endpoint has no variance of its own
          and no causes, the co-path contributes nothing at all and is silently inert.
        """
        issues: list[ModelIssue] = []
        for copath in self._copaths.values():
            if self.cov_value(copath.a, copath.b) is not None:
                issues.append(
                    ModelIssue(
                        "warning",
                        f"{copath.a!r} and {copath.b!r} are joined by BOTH a co-path and a "
                        f"bidirected edge. The co-path already induces covariance between them "
                        f"and among all their causes, so stating it again by hand almost "
                        f"certainly double-counts. Drop one.",
                    )
                )
            for name in (copath.a, copath.b):
                if not self.parents(name) and self.cov_value(name, name) is None:
                    issues.append(
                        ModelIssue(
                            "warning",
                            f"co-path '{copath.a} -- {copath.b}' touches {name!r}, which has no "
                            f"variance and no causes. A co-path contributes "
                            f"mu * Var[{copath.a}] * Var[{copath.b}], so this one contributes "
                            f"nothing.",
                        )
                    )
        return issues

    # -- copying ----------------------------------------------------------------------
    def copy(self, name: str | None = None) -> "Model":
        """An independent copy, for branching a model without disturbing the original."""
        new = Model(name if name is not None else self.name, self.units)
        new._vars = dict(self._vars)
        new._directed = dict(self._directed)
        new._bidirected = dict(self._bidirected)
        new._copaths = dict(self._copaths)
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
        if self._copaths:
            lines.append(f"  co-paths ({len(self._copaths)}):")
            lines += [f"    {e}" for e in self._copaths.values()]
        if self._assumptions:
            lines.append(f"  assumptions ({len(self._assumptions)}):")
            lines += [f"    {sp.pretty(eq, use_unicode=False)}" for eq in self._assumptions]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<Model {self.name!r} vars={len(self._vars)} "
            f"paths={len(self._directed)} covs={len(self._bidirected)} "
            f"copaths={len(self._copaths)} "
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
