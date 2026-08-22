"""Closed-form symbolic covariance engine (RAM).

Computes the model-implied covariance matrix of a :class:`pathmgr.core.model.Model` by the
Reticular Action Model identity

    Sigma = (I - A)^-1 S (I - A)^-T          (over ALL nodes)
    Sigma_observed = F Sigma F^T             (a view, not the primary object)

The **full, unfiltered** matrix over every node is the primary object, because the point of
pathMgr is to ask for the covariance between *any* two variables -- latent and intermediate
ones included. The observed/latent filter `F` is applied only when explicitly asked for.

How it is actually computed
---------------------------
Not by inverting a symbolic matrix, in the common case. For a **recursive** (acyclic) model
the engine uses two topological sweeps and never forms `(I - A)^-1` at all. Writing
`T = S (I - A)^-T`, the two identities

    T[v, u]     = sum over parents p of u:  A[u, p] T[v, p]      + S[v, u]
    Sigma[v, u] = sum over parents p of v:  A[v, p] Sigma[p, u]  + T[v, u]

each let an entry be built from entries already computed, if `u` (resp. `v`) is visited in
topological order. Cost is O(n^2 * mean-parents) symbolic multiply-adds instead of the
O(n^3) of a matrix product on top of an inverse -- and, more importantly for sympy, every
term is generated already in expanded sum-of-products form, so like terms cancel as they
appear rather than after a blowup. See `docs/scale_ram.md` for measured timings.

For a **cyclic** model (feedback loops) there is no topological order, so the engine falls
back to the explicit inverse. This is a genuine capability difference from the chain
enumeration in :mod:`pathmgr.core.tracing`: the matrix form handles feedback loops, whose
Wright chains are infinite in number, by summing the geometric series implicitly. A model
with a cycle whose `(I - A)` is structurally singular is rejected with
:class:`SingularModelError`.

Simplification policy
---------------------
Deliberate, and documented, rather than reflexive. `sympy.simplify` is expensive and is
never called automatically. The only automatic step is `expand` on each entry as it is
built, which is cheap and is what makes cancellation happen. Beyond that the caller chooses:
``form="expanded"`` (the default for :meth:`RAMEngine.cov` -- canonical, and the form to
compare term-by-term against the tracer), ``"raw"``, ``"simplified"``, or ``"factored"``.
:meth:`RAMEngine.corr` defaults to ``"simplified"`` because an unsimplified ratio of
expanded polynomials is unreadable.

Units
-----
The engine never assumes unit variance. `corr` always divides by the *model-implied*
standard deviations, which are themselves symbolic expressions. A model's
:class:`pathmgr.core.units.Units` is carried on the engine and reported by
:meth:`RAMEngine.explain`, so a returned expression is never scale-ambiguous; a standardized
model can be audited with :meth:`RAMEngine.check_standardization`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from .model import (
    CoPathVarianceError,
    Model,
    copath_mu,
    copath_resolution_order,
    node_depths,
)
from .units import Units

__all__ = [
    "CoPathLimitError",
    "CoPathVarianceError",
    "CovarianceReport",
    "CyclicModelError",
    "RAMEngine",
    "SingularModelError",
]

#: post-processing modes accepted by the query methods
FORMS = ("raw", "expanded", "simplified", "factored")


class SingularModelError(ValueError):
    """``(I - A)`` is not invertible, so the model implies no covariance matrix."""


class CyclicModelError(ValueError):
    """A topological order was requested for a model with a directed cycle."""


class CoPathLimitError(ValueError):
    """Too many distinct co-path sequences to enumerate; the result would be incomplete."""


@dataclass
class CovarianceReport:
    """A covariance with everything needed to interpret it -- above all, its units."""

    x: str
    y: str
    cov: sp.Expr
    var_x: sp.Expr
    var_y: sp.Expr
    corr: sp.Expr
    units: Units
    latent: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        tag = f" (latent: {', '.join(self.latent)})" if self.latent else ""
        return (
            f"Cov[{self.x}, {self.y}]{tag}\n"
            f"  = {self.cov}\n"
            f"  units: {self.units}\n"
            f"  Var[{self.x}] = {self.var_x}\n"
            f"  Var[{self.y}] = {self.var_y}\n"
            f"  Corr[{self.x}, {self.y}] = {self.corr}"
        )


class RAMEngine:
    """Model-implied covariances, computed once and cached against ``model.revision``.

    >>> import pathmgr as pm
    >>> m = pm.from_text("y ~ b*x\\nx ~~ V_x*x\\ny ~~ V_r*y")
    >>> eng = pm.RAMEngine(m)
    >>> eng.cov("x", "y")
    V_x*b
    >>> eng.var("y")
    V_r + V_x*b**2
    """

    #: refuse to enumerate more than this many co-path sequences when building Sigma
    DEFAULT_MAX_COPATH_SEQUENCES = 20_000

    def __init__(
        self,
        model: Model,
        auto_expand: bool = True,
        max_copath_sequences: int | None = None,
    ):
        self.model = model
        #: expand each entry as it is built. Off only for profiling comparisons.
        self.auto_expand = auto_expand
        self.max_copath_sequences = (
            self.DEFAULT_MAX_COPATH_SEQUENCES
            if max_copath_sequences is None
            else max_copath_sequences
        )
        self._copath_sequences = 0
        self._revision: int | None = None
        self._sigma: sp.Matrix | None = None
        #: the co-path-free Sigma; the full one is built from it only when actually asked for
        self._sigma0: sp.Matrix | None = None
        self._sigma_full: sp.Matrix | None = None
        self._order: tuple[str, ...] = ()
        self._index: dict[str, int] = {}
        self._used_inverse: bool = False

    # -- properties -------------------------------------------------------------------
    @property
    def units(self) -> Units:
        """The scale the results are on -- taken from the model, never assumed."""
        return self.model.units

    @property
    def order(self) -> tuple[str, ...]:
        """All node names, in the row/column order of :meth:`sigma`."""
        self._refresh()
        return self._order

    @property
    def used_inverse(self) -> bool:
        """True if the last computation needed the explicit inverse (a cyclic model)."""
        self._refresh()
        return self._used_inverse

    # -- the matrices -----------------------------------------------------------------
    def sigma(self) -> sp.Matrix:
        """The full model-implied covariance matrix over **all** nodes, latents included.

        With co-paths this is the expensive object: every co-path sequence contributes an outer
        product over the whole matrix. :meth:`cov` does **not** go through it -- see
        :meth:`_copath_entry` -- so asking for one covariance of a deep pedigree is cheap even
        when materialising all of Sigma would not be.
        """
        self._refresh()
        assert self._sigma0 is not None
        if not self.model.has_copaths or self._sigma0.rows == 0:
            return self._sigma0
        if self._sigma_full is None:
            self._sigma_full = self._apply_copaths(self._sigma0)
        return self._sigma_full

    def sigma_copath_free(self) -> sp.Matrix:
        """Sigma with every co-path removed -- the object co-path chains are assembled from."""
        self._refresh()
        assert self._sigma0 is not None
        return self._sigma0

    def sigma_observed(self) -> tuple[sp.Matrix, tuple[str, ...]]:
        """``(F Sigma F^T, observed names)`` -- a view on :meth:`sigma`, not the primary object."""
        full = self.sigma()
        observed = self.model.observed
        rows = [self._index[n] for n in observed]
        return full[rows, rows], observed

    def inverse_IA(self) -> sp.Matrix:
        """``(I - A)^-1``, the total-effects matrix. Computed on demand, not to build Sigma."""
        A = self.model.A_matrix()
        return self._invert(sp.eye(A.rows) - A)

    # -- queries ----------------------------------------------------------------------
    def cov(
        self,
        x: str,
        y: str,
        form: str = "expanded",
        apply_assumptions: bool | tuple[str, ...] | list[str] = False,
    ) -> sp.Expr:
        """Model-implied covariance of ``x`` and ``y``. Either may be latent or intermediate.

        ``form`` is one of ``"raw"``, ``"expanded"`` (default), ``"simplified"``,
        ``"factored"`` -- see the module docstring on why the default is not ``simplified``.
        ``apply_assumptions`` substitutes the model's side relations; it is opt-in so nothing
        is ever silently rewritten. Pass ``True`` for the unambiguous ``Symbol = expr``
        relations only, or a sequence of symbol names to also solve relations like
        ``V_A + V_E = 1`` for those symbols (see :meth:`Model.substitutions`).
        """
        self._refresh()
        self._require(x, y)
        if self.model.has_copaths and self._sigma_full is None:
            expr = self._copath_entry(self._index[x], self._index[y])
        else:
            expr = self.sigma()[self._index[x], self._index[y]]
        return self._finish(expr, form, apply_assumptions)

    def var(
        self, x: str, form: str = "expanded", apply_assumptions: bool | tuple[str, ...] | list[str] = False
    ) -> sp.Expr:
        """Model-implied variance of ``x`` -- the natural special case of :meth:`cov`."""
        return self.cov(x, x, form=form, apply_assumptions=apply_assumptions)

    def corr(
        self,
        x: str,
        y: str,
        form: str = "simplified",
        apply_assumptions: bool | tuple[str, ...] | list[str] = False,
    ) -> sp.Expr:
        """Model-implied correlation of ``x`` and ``y``.

        Divides by the **model-implied** standard deviations, which are themselves symbolic
        expressions -- unit variance is never assumed, whatever the model's units say.
        """
        self._refresh()
        self._require(x, y)
        covariance = self.cov(x, y, form="raw")
        var_x = self.cov(x, x, form="raw")
        var_y = self.cov(y, y, form="raw")
        for name, value in ((x, var_x), (y, var_y)):
            if value == 0:
                raise ValueError(
                    f"Var[{name}] is identically zero, so no correlation is defined. "
                    f"An exogenous variable with no bidirected self-edge is the usual cause; "
                    f"model.validate() reports those."
                )
        expr = covariance / sp.sqrt(var_x * var_y)
        return self._finish(expr, form, apply_assumptions, ratio=True)

    def explain(self, x: str, y: str, apply_assumptions: bool | tuple[str, ...] | list[str] = False) -> CovarianceReport:
        """A covariance together with its units and both variances -- never scale-ambiguous."""
        self._require(x, y)
        latent = tuple(n for n in (x, y) if self.model.var(n).latent)
        return CovarianceReport(
            x=x,
            y=y,
            cov=self.cov(x, y, apply_assumptions=apply_assumptions),
            var_x=self.var(x, apply_assumptions=apply_assumptions),
            var_y=self.var(y, apply_assumptions=apply_assumptions),
            corr=self.corr(x, y, apply_assumptions=apply_assumptions),
            units=self.units,
            latent=latent,
        )

    # -- units audit ------------------------------------------------------------------
    def check_standardization(self, apply_assumptions: bool = True) -> list[str]:
        """For a standardized model, the variables whose implied variance is not 1.

        Gives the units declaration teeth: a model that claims to be standardized to some
        reference population should imply unit variance for its variables there, and this
        says which ones do not. Returns an empty list for an unstandardized model.
        """
        if not self.units.is_standardized:
            return []
        offenders = []
        for name in self.model.names:
            implied = self.var(name, form="simplified", apply_assumptions=apply_assumptions)
            if sp.simplify(implied - 1) != 0:
                offenders.append(name)
        return offenders

    # -- internals --------------------------------------------------------------------
    def _refresh(self) -> None:
        if self._revision == self.model.revision and self._sigma is not None:
            return
        A, S, _F, order = self.model.ram()
        self._order = order
        self._index = {name: i for i, name in enumerate(order)}
        if not order:
            self._sigma = sp.zeros(0, 0)
            self._used_inverse = False
        elif self.model.is_recursive:
            self._sigma = self._sigma_recursive(A, S, order)
            self._used_inverse = False
        else:
            self._sigma = self._sigma_via_inverse(A, S)
            self._used_inverse = True
        self._sigma0 = self._sigma
        self._sigma_full = None
        self._revision = self.model.revision

    def _oriented_copaths(self, index: dict[str, int], sigma0: sp.Matrix | None = None):
        """Each co-path in both orientations, as ``(process, near, far, mu)``.

        A co-path declared by its **correlation** has its ``mu`` derived here, and the order matters:
        resolving ``mu = rho / (sd_a * sd_b)`` needs the endpoints' *true* variances, and under
        assortment those depend on the co-paths **upstream** of them -- a co-path does not change the
        variance of the pair it matches, but it does change their descendants', which is the entire
        content of the AM dynamics.

        So co-paths are resolved in :func:`~pathmgr.core.model.copath_resolution_order` -- increasing
        depth of the deeper endpoint -- and each one's variances are computed with only the
        already-resolved co-paths in play. That is exact rather than approximate: every dependency of
        a co-path at depth *d* lies at depth *< d*, so nothing needed is missing.
        """
        if sigma0 is None:
            sigma0 = self.sigma_copath_free()
        ordered = copath_resolution_order(self.model)
        depths = node_depths(self.model)
        if not depths and any(c.is_standardized for c in self.model.copaths):
            raise CoPathVarianceError(
                "this model has a directed cycle, so no co-path is upstream of another and a "
                "declared correlation cannot be resolved in dependency order. Give the co-paths an "
                "explicit coefficient= (raw mu)."
            )

        oriented: list[tuple[str, int, int, sp.Expr]] = []
        for copath in ordered:
            if copath.is_standardized:
                # variances computed against the co-paths resolved SO FAR, which are exactly the
                # ones upstream of this pair
                var_a = self._copath_entry(
                    index[copath.a], index[copath.a], oriented, sigma0
                )
                var_b = self._copath_entry(
                    index[copath.b], index[copath.b], oriented, sigma0
                )
                mu = copath_mu(copath, var_a, var_b, self.model.substitutions())
            else:
                mu = copath.coefficient
            if mu == 0:
                continue
            for a, b in sorted({(copath.a, copath.b), (copath.b, copath.a)}):
                oriented.append((copath.process, index[a], index[b], mu))
        return oriented

    def _copath_entry(self, row: int, column: int, oriented=None, sigma0=None) -> sp.Expr:
        """One entry of Sigma, without materialising the whole matrix.

        The same sum over sequences of distinct-process co-paths as :meth:`_apply_copaths`, but
        for a single ``(row, column)``. Each sequence contributes the **scalar**

            sigma0[row, u_1] * mu_1 * sigma0[v_1, u_2] * ... * mu_k * sigma0[v_k, column]

        where the full-matrix version has to form an outer product instead. That is an ``n^2``
        saving per sequence, and on a deep pedigree it is the difference between usable and not:
        the number of sequences grows with pedigree depth, so paying ``n^2`` for each is what made
        a five-generation unroll take minutes. Asking for one covariance is the common case.
        """
        if sigma0 is None:
            sigma0 = self.sigma_copath_free()
        if oriented is None:
            oriented = self._oriented_copaths(self._index, sigma0)
        total = sigma0[row, column]
        if not oriented:
            return total
        self._copath_sequences = 0

        def extend(far: int, used: frozenset[str], scalar: sp.Expr) -> None:
            nonlocal total
            self._copath_sequences += 1
            if self._copath_sequences > self.max_copath_sequences:
                raise CoPathLimitError(
                    f"more than {self.max_copath_sequences} co-path sequences for one entry; "
                    f"the result would be incomplete. The count grows with the number of mating "
                    f"processes linked by nonzero covariance. Raise max_copath_sequences if you "
                    f"can afford it."
                )
            tail = sigma0[far, column]
            if tail != 0:
                total = self._norm(total + scalar * tail)
            for process, near2, far2, mu2 in oriented:
                if process in used:
                    continue
                link = sigma0[far, near2]
                if link == 0:
                    continue
                extend(far2, used | {process}, self._norm(scalar * link * mu2))

        for process, near, far, mu in oriented:
            head = sigma0[row, near]
            if head == 0:
                continue
            extend(far, frozenset({process}), self._norm(head * mu))
        return total

    def _apply_copaths(self, sigma0: sp.Matrix) -> sp.Matrix:
        """Add every co-path chain's contribution to the co-path-free ``sigma0``.

        Derivation. A chain crossing co-paths ``c_1 ... c_k`` in order is a sequence of standard
        segments joined by those co-paths. Summing over all segment choices bundles each leg
        into a ``sigma0`` entry, so the whole family of chains crossing that particular co-path
        sequence contributes

            sigma0[:, u_1] * (mu_1 sigma0[v_1, u_2] mu_2 ... mu_k) * sigma0[v_k, :]

        -- a scalar times an **outer product** of one ``sigma0`` column and one ``sigma0`` row.
        Being built from ``sigma0`` columns is exactly why a co-path reaches a matched
        variable's *causes*, which a bidirected edge cannot: an ``S`` entry contributes
        ``B[:, u] s B[:, v]^T``, using path columns rather than covariance columns.

        Each co-path is traversed in either orientation, and by Sunde's rule the sequence may
        use **at most one co-path per mating process**. So the sum runs over ordered sequences
        of distinct-process co-paths, which is finite. It is enumerated depth-first and pruned
        wherever the connecting ``sigma0[v_i, u_{i+1}]`` is zero -- in practice that kills the
        overwhelming majority of sequences, since consecutive co-paths must be linked by actual
        covariance in the co-path-free model.

        **Why there is no clean geometric-series closed form.** Allowing repeats would give
        ``sigma0 (I - C sigma0)^-1`` with ``C`` holding the co-path coefficients, and that
        overcounts. On a single mated pair with one co-path, expanding the series term by term
        at the ``(y_m, y_p)`` entry gives

            k=1:  rho_y V_P     <- the only Sunde-legal term
            k=3:  rho_y^3 V_P   <- traverses the SAME co-path three times
            k=5:  rho_y^5 V_P   <- ... and five times

        summing to ``rho_y V_P / (1 - rho_y^2)`` instead of ``rho_y V_P`` (numerically 0.3495
        against the correct 0.3180 at ``V_A = 0.46``, ``V_E = 0.6``, ``rho_y = 0.3``). The even
        powers vanish here only because ``sigma0`` is block diagonal across the two partners;
        that is incidental, not protective. Restricted to distinct co-paths the sum is a
        simple-walk enumeration over the co-path graph, which has no closed form in general --
        hence the explicit enumeration and the :class:`CoPathLimitError` guard.
        """
        index = self._index
        oriented = self._oriented_copaths(index, sigma0)
        if not oriented:
            return sigma0

        total = sigma0.as_mutable()
        self._copath_sequences = 0

        def extend(left: int, far: int, used: frozenset[str], scalar: sp.Expr) -> None:
            self._copath_sequences += 1
            if self._copath_sequences > self.max_copath_sequences:
                raise CoPathLimitError(
                    f"more than {self.max_copath_sequences} co-path sequences while building "
                    f"Sigma; the result would be incomplete. The count grows with the number "
                    f"of mating processes that are linked by nonzero covariance. Raise "
                    f"max_copath_sequences if you can afford it."
                )
            column = sigma0[:, left]
            row = sigma0[far, :]
            for i in range(total.rows):
                if column[i] == 0:
                    continue
                weight = self._norm(scalar * column[i])
                if weight == 0:
                    continue
                for j in range(total.cols):
                    if row[j] == 0:
                        continue
                    total[i, j] = self._norm(total[i, j] + weight * row[j])
            for process, near2, far2, mu2 in oriented:
                if process in used:
                    continue  # one co-path per mating process per chain
                link = sigma0[far, near2]
                if link == 0:
                    continue
                extend(left, far2, used | {process}, self._norm(scalar * link * mu2))

        for process, near, far, mu in oriented:
            extend(near, far, frozenset({process}), mu)
        return total.as_immutable()

    def _sigma_recursive(self, A: sp.Matrix, S: sp.Matrix, order: tuple[str, ...]) -> sp.Matrix:
        """Two topological sweeps; no matrix inverse is ever formed. See module docstring."""
        n = len(order)
        topo = self.topological_order()
        parents = {
            i: [(self._index[p], A[i, self._index[p]]) for p in self.model.parents(order[i])]
            for i in range(n)
        }

        # T = S (I - A)^-T, built row by row with u in topological order
        T = sp.zeros(n, n)
        for v in range(n):
            for u in topo:
                acc = S[v, u]
                for p, coeff in parents[u]:
                    if T[v, p] != 0:
                        acc += coeff * T[v, p]
                T[v, u] = self._norm(acc)

        # Sigma = (I - A)^-1 T, built with v in topological order. Sigma is symmetric, so
        # once row u is complete every entry Sigma[v, u] with u earlier in the topological
        # order can be copied from Sigma[u, v] instead of recomputed -- halving the work.
        sigma = sp.zeros(n, n)
        done: set[int] = set()
        for v in topo:
            for u in range(n):
                if u in done:
                    sigma[v, u] = sigma[u, v]
                    continue
                acc = T[v, u]
                for p, coeff in parents[v]:
                    if sigma[p, u] != 0:
                        acc += coeff * sigma[p, u]
                sigma[v, u] = self._norm(acc)
            done.add(v)
        return sigma

    def _sigma_via_inverse(self, A: sp.Matrix, S: sp.Matrix) -> sp.Matrix:
        """The general form, needed when the model has a directed cycle."""
        B = self._invert(sp.eye(A.rows) - A)
        return (B * S * B.T).applyfunc(self._norm)

    def _invert(self, IA: sp.Matrix) -> sp.Matrix:
        # A recursive model is triangular under a topological permutation with 1s on the
        # diagonal, so det(I - A) == 1 identically. Skip the determinant: computing it
        # symbolically is expensive and, for a large pedigree, the dominant cost here.
        if self.model.is_recursive:
            return IA.inv()
        det = IA.det()
        if sp.simplify(det) == 0:
            raise SingularModelError(
                "(I - A) is singular, so this model implies no covariance matrix. "
                "A feedback loop with unit total gain is the usual cause "
                f"(cycles: {['-> '.join(c) for c in self.model.cycles()]})."
            )
        try:
            return IA.inv()
        except sp.matrices.exceptions.NonInvertibleMatrixError as exc:  # pragma: no cover
            raise SingularModelError(f"(I - A) is not invertible: {exc}") from exc

    def topological_order(self) -> list[int]:
        """Node indices in topological order (parents before children). Kahn's algorithm.

        Indices are into ``model.names``, which is also the row order of :meth:`sigma`. Read
        from the model directly rather than from the cache, so this is correct even when
        called before or independently of a Sigma build.
        """
        order = self.model.names
        index = {name: i for i, name in enumerate(order)}
        indegree = {i: len(self.model.parents(order[i])) for i in range(len(order))}
        queue = [i for i in range(len(order)) if indegree[i] == 0]
        out: list[int] = []
        while queue:
            i = queue.pop(0)
            out.append(i)
            for child in self.model.children(order[i]):
                j = index[child]
                indegree[j] -= 1
                if indegree[j] == 0:
                    queue.append(j)
        if len(out) != len(order):
            raise CyclicModelError(
                "model has a directed cycle, so it has no topological order: "
                + "; ".join(" -> ".join(c) for c in self.model.cycles())
            )
        return out

    def _norm(self, expr: sp.Expr) -> sp.Expr:
        return sp.expand(expr) if self.auto_expand else expr

    def _finish(
        self,
        expr: sp.Expr,
        form: str,
        apply_assumptions: bool | tuple[str, ...] | list[str],
        ratio: bool = False,
    ) -> sp.Expr:
        if form not in FORMS:
            raise ValueError(f"form must be one of {FORMS}, got {form!r}")
        if apply_assumptions:
            solve_for = () if apply_assumptions is True else tuple(apply_assumptions)
            subs = self.model.substitutions(solve_for=solve_for)
            # side relations may reference one another; iterate to a fixed point
            for _ in range(len(subs) + 1):
                new = expr.subs(subs)
                if new == expr:
                    break
                expr = new
        if form == "raw":
            return expr
        if form == "expanded":
            # a ratio does not benefit from expand; cancel is the right canonical form
            return sp.cancel(expr) if ratio else sp.expand(expr)
        if form == "factored":
            return sp.factor(expr)
        return sp.simplify(sp.cancel(expr) if ratio else expr)

    def _require(self, *names: str) -> None:
        self._refresh()
        for name in names:
            if name not in self._index:
                raise KeyError(
                    f"unknown variable {name!r}; the model has: {', '.join(self._order)}"
                )

    def __repr__(self) -> str:
        return (
            f"<RAMEngine over {self.model!r} "
            f"[{self.units}] cached_rev={self._revision}>"
        )
