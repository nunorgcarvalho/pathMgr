"""Wright path tracing: covariance by explicit chain enumeration.

The second, independent engine. :mod:`pathmgr.core.ram` gives the right answer but hides the
reasoning; this module returns the *decomposition* -- which chains exist and what each
contributes -- because that is the part that goes into a writeup.

**This module is the specification for co-path semantics.** Sunde's rules are stated as tracing
rules, so they are implemented here directly and the RAM engine is made to agree with this,
not the other way round. The standing correctness property is that the two agree symbolically
on every model in ``tests/battery.py``.

What a chain is here, and how it relates to the rules in the textbooks
----------------------------------------------------------------------
A **segment** is a standard valid chain

    x <- ... <- u   <->   v -> ... -> y

a directed path traced **backward** from ``x`` to some ancestor ``u``, **exactly one**
bidirected edge ``u <-> v`` (possibly ``u == v``, which is a variance), then a directed path
traced **forward** from ``v`` to ``y``. Its value is the product of the directed coefficients on
both legs times the bidirected edge's value.

A **chain** is one or more segments joined by **co-paths**::

    [segment] -- [segment] -- [segment]

This is exactly the RAM identity written out term by term. Without co-paths,
``Sigma = B S B^T`` with ``B = (I - A)^-1 = sum_k A^k``, so
``Sigma[x, y] = sum_{u,v} B[x,u] S[u,v] B[y,v]`` and ``B[x,u]`` is the sum over directed paths
from ``u`` to ``x`` of their coefficient products.

Co-path rules (Sunde et al. 2025 Nat Commun, Supplementary Notes 1 and 3)
-------------------------------------------------------------------------
- A co-path "denotes covariance attributable to matching ... where covariance is induced
  **without causing variance**", and "will induce correlations in all the causes of the
  variables that are matched". That backward reach up the graph is why it is not a bidirected
  edge -- see :class:`pathmgr.core.model.CoPath`.
- "A co-path connects two valid chains per standard path tracing rules into longer chains."
- "A chain cannot start or end with a co-path" -- you must always begin tracing backward.
- "A chain cannot include multiple co-path coefficients stemming from the same **mating
  process** (i.e., linking the same pair of partners)" (Supplementary Note 3, introduced for
  cross-trait assortment, where one couple carries four co-paths).

Note what that last rule does **not** say: co-paths from *different* mating processes may
appear in the same chain, and must be able to -- that is what accumulates
``((1 + rho_g)/2)^d`` across generations. Enforcing "one co-path per chain" would silently
truncate every multi-generation result to first order in ``rho_g``.

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
passes through ``w`` twice. Dropping them would lose ``Var[w]``'s ancestral part entirely. So a
node may repeat across legs; it cannot repeat *within* a leg, which in an acyclic model is
automatic. :attr:`Chain.revisits` reports repeated nodes, since they are the ones a reader
hand-checking against a textbook will query.

Limits
------
- **Cyclic models cannot be traced.** A feedback loop has infinitely many chains; the matrix
  inverse sums that geometric series in closed form and enumeration cannot. Detected up front
  and raised as :class:`UntraceableModelError`, pointing at :class:`pathmgr.RAMEngine` -- never
  silently truncated.
- Chain count is the product of the legs' path counts, multiplied again for every co-path
  crossing. :class:`WrightTracer` caps it with ``max_chains`` and raises
  :class:`ChainLimitError` rather than appearing to hang.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain as _iterchain

import sympy as sp

from .model import CoPath, CoPathVarianceError, Model, copath_mu
from .units import Units

__all__ = [
    "Chain",
    "ChainLimitError",
    "Decomposition",
    "Segment",
    "UntraceableModelError",
    "WrightTracer",
    "tex",
]

BACKWARD = " <- "
FORWARD = " -> "
BIDIRECTED = " <-> "
COPATH = " -- "


def tex(expression: sp.Expr, latex_names: dict | None = None) -> str:
    """``sp.latex``, but rendering named symbols and subexpressions the way a document names them.

    ``latex_names`` maps a sympy expression to the LaTeX that should appear in its place, e.g.
    ``{V_A0 + V_E: r"\\VPo"}`` when the surrounding writeup has been calling that sum ``\\VPo`` for
    two pages. Without it a caption renders the sum it was derived from, which is correct and
    unreadable next to prose using the macro.

    Plain symbols could be done with ``sp.latex(symbol_names=...)`` alone. **Composite**
    subexpressions cannot, and they are the case that actually came up, so both go through the same
    route: substitute a placeholder symbol, then name the placeholder. Longest expressions are
    substituted first, so a key that contains another key still matches -- ``{V_A + V_E: ...}``
    would otherwise be blocked by a ``{V_A: ...}`` substitution landing first.

    Substitution happens **after** any ``factor()``/``simplify()`` the caller has already done, on
    the expression it is about to render, so it cannot disturb the algebra: nothing is computed
    from the result.
    """
    if not latex_names:
        return sp.latex(expression)

    placeholders: dict[sp.Symbol, str] = {}
    substituted = expression
    keys = sorted(latex_names, key=lambda k: sp.count_ops(k), reverse=True)
    for index, key in enumerate(keys):
        placeholder = sp.Symbol(f"_pmName{index}")
        replaced = substituted.subs(key, placeholder)
        if replaced != substituted or substituted == key:
            placeholders[placeholder] = latex_names[key]
            substituted = replaced
    if not placeholders:
        return sp.latex(expression)
    return sp.latex(substituted, symbol_names=placeholders)


class UntraceableModelError(ValueError):
    """This model cannot be traced by enumeration -- use the RAM engine instead."""


class ChainLimitError(ValueError):
    """Enumeration exceeded ``max_chains``; the total would be incomplete."""


@dataclass(frozen=True)
class Segment:
    """One standard valid chain: ``x <- ... <- u  <->  v -> ... -> y``."""

    #: nodes from the start back to the bidirected edge, i.e. ``(x, ..., u)``
    backward: tuple[str, ...]
    #: nodes from the bidirected edge forward to the end, i.e. ``(v, ..., y)``
    forward: tuple[str, ...]
    #: the value of the bidirected edge ``u <-> v`` (a variance when ``u == v``)
    bidirected_value: sp.Expr

    @property
    def start(self) -> str:
        return self.backward[0]

    @property
    def end(self) -> str:
        return self.forward[-1]

    @property
    def pivot(self) -> tuple[str, str]:
        """The bidirected edge ``(u, v)`` this segment passes through."""
        return (self.backward[-1], self.forward[0])

    @property
    def is_variance_pivot(self) -> bool:
        """True if the segment turns around at a single variable's variance (``u == v``)."""
        return self.backward[-1] == self.forward[0]

    @property
    def nodes(self) -> tuple[str, ...]:
        """Nodes visited in order (the pivot node once if ``u == v``)."""
        if self.is_variance_pivot:
            return self.backward + self.forward[1:]
        return self.backward + self.forward

    def directed_edges(self) -> tuple[tuple[str, str], ...]:
        back = tuple(
            (self.backward[i + 1], self.backward[i]) for i in range(len(self.backward) - 1)
        )
        fwd = tuple((self.forward[i], self.forward[i + 1]) for i in range(len(self.forward) - 1))
        return back + fwd

    def path_string(self) -> str:
        left = BACKWARD.join(self.backward)
        right = FORWARD.join(self.forward)
        return f"{left}{BIDIRECTED}{right}"

    def tex_path(self, labels: dict[str, str] | None = None) -> str:
        labels = labels or {}
        left = r" \leftarrow ".join(_tex_name(n, labels) for n in self.backward)
        right = r" \rightarrow ".join(_tex_name(n, labels) for n in self.forward)
        return f"{left} \\leftrightarrow {right}"

    def __str__(self) -> str:
        return self.path_string()


@dataclass(frozen=True)
class Chain:
    """One or more :class:`Segment`s joined by co-paths, and what the whole thing contributes."""

    segments: tuple[Segment, ...]
    #: the co-paths joining consecutive segments; ``len(crossings) == len(segments) - 1``
    crossings: tuple[CoPath, ...]
    #: every factor in chain order; their product is :attr:`contribution`
    factors: tuple[sp.Expr, ...]
    #: the chain's symbolic contribution to the covariance
    contribution: sp.Expr

    # -- endpoints --------------------------------------------------------------------
    @property
    def x(self) -> str:
        return self.segments[0].start

    @property
    def y(self) -> str:
        return self.segments[-1].end

    # -- single-segment conveniences (the co-path-free case) ---------------------------
    @property
    def backward(self) -> tuple[str, ...]:
        """The first segment's backward leg."""
        return self.segments[0].backward

    @property
    def forward(self) -> tuple[str, ...]:
        """The last segment's forward leg."""
        return self.segments[-1].forward

    @property
    def bidirected_value(self) -> sp.Expr:
        """The bidirected value, for a single-segment chain. Use :attr:`pivots` otherwise."""
        self._require_single("bidirected_value")
        return self.segments[0].bidirected_value

    @property
    def pivot(self) -> tuple[str, str]:
        """The bidirected edge crossed, for a single-segment chain. See :attr:`pivots`."""
        self._require_single("pivot")
        return self.segments[0].pivot

    @property
    def is_variance_pivot(self) -> bool:
        self._require_single("is_variance_pivot")
        return self.segments[0].is_variance_pivot

    def _require_single(self, attribute: str) -> None:
        if len(self.segments) != 1:
            raise AttributeError(
                f"{attribute!r} is only defined for a single-segment chain; this chain crosses "
                f"{len(self.crossings)} co-path(s). Use .pivots / .segments instead."
            )

    # -- general ----------------------------------------------------------------------
    @property
    def pivots(self) -> tuple[tuple[str, str], ...]:
        """The bidirected edge crossed by each segment, in order."""
        return tuple(s.pivot for s in self.segments)

    @property
    def copath_processes(self) -> tuple[str, ...]:
        """The mating processes this chain draws on -- distinct by Sunde's rule."""
        return tuple(c.process for c in self.crossings)

    @property
    def crosses_copaths(self) -> bool:
        return bool(self.crossings)

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every node visited, in order, across all segments."""
        return tuple(_iterchain.from_iterable(s.nodes for s in self.segments))

    @property
    def revisits(self) -> tuple[str, ...]:
        """Nodes visited more than once. Legitimate here -- see the module docstring."""
        seen: dict[str, int] = {}
        for node in self.nodes:
            seen[node] = seen.get(node, 0) + 1
        return tuple(sorted(n for n, count in seen.items() if count > 1))

    @property
    def length(self) -> int:
        """Number of directed edges traversed (bidirected edges and co-paths not counted)."""
        return sum(len(s.backward) - 1 + len(s.forward) - 1 for s in self.segments)

    def directed_edges(self) -> tuple[tuple[str, str], ...]:
        """The directed edges used, each as ``(src, dst)``. For diagram highlighting."""
        return tuple(_iterchain.from_iterable(s.directed_edges() for s in self.segments))

    def copath_edges(self) -> tuple[tuple[str, str], ...]:
        """The co-paths crossed, each as ``(a, b)``. For diagram highlighting."""
        return tuple((c.a, c.b) for c in self.crossings)

    # -- rendering --------------------------------------------------------------------
    def path_string(self) -> str:
        """``'y_i <- g_i <-> g_j -> y_j'``, with ``--`` marking each co-path crossing."""
        return COPATH.join(s.path_string() for s in self.segments)

    def factor_string(self) -> str:
        """``'1/2 * V_A * 1/2'`` -- the contribution before multiplying out."""
        return " * ".join(sp.sstr(f) for f in self.factors)

    #: how a co-path crossing renders in math mode. An em dash in \text, not a bare "-",
    #: which would read as a minus sign and make a chain look like a subtraction.
    COPATH_TEX = r" \;\text{---}\; "

    def tex_path(self, labels: dict[str, str] | None = None) -> str:
        """The chain as math-mode LaTeX, using each variable's label where it has one."""
        return self.COPATH_TEX.join(s.tex_path(labels) for s in self.segments)

    def tex_factors(self, omit_unit: bool = True, latex_names: dict | None = None) -> str:
        """The contribution as the PRODUCT being formed, before it is multiplied out.

        ``\\frac{1}{2} \\cdot \\beta_{0} \\cdot \\mu \\cdot \\beta_{0} \\cdot \\frac{1}{2}`` --
        so a figure can show the arithmetic rather than only its answer, and a reader can check it
        against the chain edge by edge.

        Unit factors are dropped by default, matching the edge labels (which also omit a
        coefficient of 1): leaving them in gives ``\\cdot 1 \\cdot`` runs that obscure the terms
        that matter. If every factor is 1 the product is just ``1``.
        """
        factors = [f for f in self.factors if not (omit_unit and f == 1)]
        if not factors:
            return "1"
        rendered = []
        for factor in factors:
            text = tex(factor, latex_names)
            if isinstance(factor, sp.Add) or (factor.is_number and factor < 0):
                text = f"\\left({text}\\right)"
            rendered.append(text)
        return r" \cdot ".join(rendered)

    def tex_contribution(
        self, factored: bool = True, omit_unit: bool = True, latex_names: dict | None = None
    ) -> str:
        """``<product> = <value>`` -- the arithmetic and its result in one line."""
        value = sp.factor(self.contribution) if factored else self.contribution
        product = self.tex_factors(omit_unit=omit_unit, latex_names=latex_names)
        return f"{product} = {tex(value, latex_names)}"

    def tex_caption(
        self,
        labels: dict[str, str] | None = None,
        name: str | None = None,
        omit_unit: bool = True,
        latex_names: dict | None = None,
    ) -> str:
        """A two-line caption: the Wright chain, then the product it contributes.

        ``name`` prefixes the second line, e.g. ``\\operatorname{Cov}[a, b]``, when the chain is
        the whole covariance rather than one term of it.

        ``omit_unit`` and ``latex_names`` exist so the caption can be made to agree with the
        diagram above it and with the document around it. They are plain arguments rather than a
        ``DiagramStyle`` because :mod:`pathmgr.core` may not import :mod:`pathmgr.render`; the
        render layer reads them off the style and passes them in. See
        :meth:`pathmgr.render.style.DiagramStyle.caption_options`.
        """
        head = self.tex_path(labels)
        body = self.tex_contribution(omit_unit=omit_unit, latex_names=latex_names)
        if name:
            body = f"{name} = {body}"
        return f"{head}\\\\{body}"

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
        """Chains by co-path count, then length, then alphabetically -- stable for a document."""
        return tuple(
            sorted(self.chains, key=lambda c: (len(c.crossings), c.length, c.path_string()))
        )

    def copath_chains(self) -> tuple[Chain, ...]:
        """Only the chains that cross at least one co-path."""
        return tuple(c for c in self.chains if c.crosses_copaths)

    def __str__(self) -> str:
        header = f"Cov[{self.x}, {self.y}]  ({len(self.chains)} chain"
        header += "s" if len(self.chains) != 1 else ""
        crossing = len(self.copath_chains())
        if crossing:
            header += f", {crossing} via co-path"
        header += f", {self.units})"
        if not self.chains:
            return header + "\n  (no chains -- the covariance is 0)\n  total = 0"
        rows = []
        for chain in self.sorted_chains():
            notes = []
            if chain.crossings:
                notes.append("co-paths: " + ", ".join(chain.copath_processes))
            if chain.revisits:
                notes.append("revisits " + ", ".join(chain.revisits))
            note = f"   [{'; '.join(notes)}]" if notes else ""
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
        document rather than the internal names. Co-path crossings render as an en-dash-like
        plain line (``-``), matching Sunde's arrowless co-path.
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
        self._path_cache: dict[str, dict[str, tuple[tuple[tuple[str, ...], sp.Expr], ...]]] = {}
        self._mu_cache: dict[tuple[str, str, str], sp.Expr] = {}
        self._mu_cache_revision: int | None = None
        self._cached_revision: int | None = None

    @property
    def units(self) -> Units:
        return self.model.units

    # -- the tracer -------------------------------------------------------------------
    def trace(self, x: str, y: str) -> Decomposition:
        """Every Wright chain between ``x`` and ``y``, with its contribution."""
        self._require_traceable()
        self._require(x, y)
        chains = list(self._chains(x, y, frozenset()))
        labels = {v.name: v.label for v in self.model.variables if v.label}
        return Decomposition(x=x, y=y, chains=tuple(chains), units=self.units, labels=labels)

    def cov(self, x: str, y: str, form: str = "expanded") -> sp.Expr:
        """The traced total -- directly comparable with :meth:`pathmgr.RAMEngine.cov`."""
        total = self.trace(x, y).total
        if form in ("raw", "expanded"):
            return sp.expand(total)
        if form == "simplified":
            return sp.simplify(total)
        if form == "factored":
            return sp.factor(total)
        raise ValueError(f"unknown form {form!r}")

    def var(self, x: str, form: str = "expanded") -> sp.Expr:
        """A variable's own variance: the same enumeration with both endpoints equal."""
        return self.cov(x, x, form=form)

    # -- enumeration ------------------------------------------------------------------
    def _chains(self, x: str, y: str, used_processes: frozenset[str]):
        """Chains from ``x`` to ``y`` not reusing any mating process in ``used_processes``.

        Two cases, exactly mirroring Sunde's rules: a chain is either a single standard
        segment, or a standard segment ending at one end of an unused co-path, that co-path,
        and then (recursively) a chain from its other end to ``y``. Because the recursion only
        ever *appends* a co-path after a completed segment, a chain can neither start nor end
        with a co-path, which is the rule stated in Supplementary Note 1.
        """
        for segment in self._segments(x, y):
            factors = self._segment_factors(segment)
            yield self._build([segment], (), factors)

        for copath in self.model.copaths:
            if copath.process in used_processes:
                continue  # one co-path per mating process per chain (Supp. Note 3)
            mu = self._mu(copath)
            if mu == 0:
                continue
            ends = {(copath.a, copath.b), (copath.b, copath.a)}
            for near, far in sorted(ends):
                for segment in self._segments(x, near):
                    head_factors = self._segment_factors(segment)
                    for rest in self._chains(far, y, used_processes | {copath.process}):
                        yield self._build(
                            [segment, *rest.segments],
                            (copath, *rest.crossings),
                            head_factors + (mu,) + rest.factors,
                        )

    def _mu(self, copath: CoPath) -> sp.Expr:
        """This co-path's ``mu``, derived from co-path-free variances if declared by correlation.

        Cached per ``model.revision``: it is asked for once per chain otherwise, and the equal-
        variance test inside :func:`~pathmgr.core.model.copath_mu` calls ``simplify``.
        """
        if not copath.is_standardized:
            return copath.coefficient
        key = (copath.a, copath.b, copath.process)
        if self._mu_cache_revision != self.model.revision:
            self._mu_cache = {}
            self._mu_cache_revision = self.model.revision
        if key not in self._mu_cache:
            self._require_copath_free_variance(copath)
            self._mu_cache[key] = copath_mu(
                copath,
                self._copath_free_cov(copath.a, copath.a),
                self._copath_free_cov(copath.b, copath.b),
            )
        return self._mu_cache[key]

    def _require_copath_free_variance(self, copath: CoPath) -> None:
        """Mirror of the RAM engine's guard; see :class:`CoPathVarianceError` for the why.

        Deliberately duplicated rather than shared: the two engines must reach the same verdict by
        their own route, and this one is stated in terms of chains rather than matrix entries.
        """
        for endpoint in (copath.a, copath.b):
            for other in self.model.copaths:
                if (other.a, other.b, other.process) == (copath.a, copath.b, copath.process):
                    continue
                for far in (other.a, other.b):
                    if self._copath_free_cov(endpoint, far) != 0:
                        raise CoPathVarianceError(
                            f"co-path {copath.a!r} -- {copath.b!r} is declared by correlation, but "
                            f"{endpoint!r} is already correlated with {far!r} before assortment, "
                            f"so the co-path {other.a!r} -- {other.b!r} changes Var[{endpoint}] "
                            f"and the correlation cannot be resolved from the co-path-free "
                            f"variances. Give this co-path an explicit coefficient= (raw mu) "
                            f"instead. See CoPathVarianceError."
                        )

    def _copath_free_cov(self, x: str, y: str) -> sp.Expr:
        """``Cov[x, y]`` from standard chains only -- no co-paths.

        Not a shortcut for ``self.cov(x, y)``: that would recurse, since resolving a declared
        correlation is exactly what we are in the middle of doing.
        """
        total = sp.Integer(0)
        for segment in self._segments(x, y):
            term = sp.Integer(1)
            for factor in self._segment_factors(segment):
                term *= factor
            total += term
        return sp.expand(total)

    def _build(self, segments, crossings, factors) -> Chain:
        return Chain(
            segments=tuple(segments),
            crossings=tuple(crossings),
            factors=tuple(factors),
            contribution=sp.expand(sp.Mul(*factors)),
        )

    def _segments(self, x: str, y: str):
        """Every standard valid chain (backward, one bidirected edge, forward) from x to y."""
        back_legs = self._backward_paths(x)
        fwd_legs = self._backward_paths(y)
        produced = 0
        for edge in self.model.bidirected_edges:
            if edge.value == 0:
                continue
            # S is symmetric, so an off-diagonal edge is traversed in both directions;
            # a variance (a == b) is traversed once.
            for u, v in sorted({(edge.a, edge.b), (edge.b, edge.a)}):
                for back_nodes, _ in back_legs.get(u, ()):
                    for fwd_nodes, _ in fwd_legs.get(v, ()):
                        produced += 1
                        if produced > self.max_chains:
                            raise ChainLimitError(
                                f"more than {self.max_chains} chains between {x!r} and {y!r}; "
                                f"the enumeration would be incomplete. Raise max_chains, or "
                                f"use RAMEngine, which does not enumerate."
                            )
                        yield Segment(
                            backward=back_nodes,
                            forward=tuple(reversed(fwd_nodes)),
                            bidirected_value=edge.value,
                        )

    def _segment_factors(self, segment: Segment) -> tuple[sp.Expr, ...]:
        """Every factor in segment order, so a reader sees the product before it collapses."""
        factors: list[sp.Expr] = []
        backward, forward = segment.backward, segment.forward
        for i in range(len(backward) - 1):
            factors.append(self.model.path_coeff(backward[i + 1], backward[i]))
        factors.append(segment.bidirected_value)
        for i in range(len(forward) - 1):
            factors.append(self.model.path_coeff(forward[i], forward[i + 1]))
        return tuple(factors)

    def _backward_paths(
        self, start: str
    ) -> dict[str, tuple[tuple[tuple[str, ...], sp.Expr], ...]]:
        """All directed paths ending at ``start``, keyed by the ancestor they begin at.

        Each value is ``(nodes, coefficient product)`` with ``nodes`` running from ``start``
        back to the ancestor. The zero-length path (``start`` itself, product 1) is included:
        a chain may pivot on ``start``'s own disturbance. Cached per model revision, since a
        deep pedigree walks the same ancestry for many queries.
        """
        if self._cached_revision != self.model.revision:
            self._path_cache.clear()
            self._cached_revision = self.model.revision
        if start in self._path_cache:
            return self._path_cache[start]

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
        result = {node: tuple(paths) for node, paths in found.items()}
        self._path_cache[start] = result
        return result

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
        return (
            f"<WrightTracer over {self.model!r} max_chains={self.max_chains}>"
        )
