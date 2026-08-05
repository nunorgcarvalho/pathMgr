"""The allele-level transmission motif: per-variant alleles, Mendelian transmission, segregation.

One randomly mating base generation and one offspring generation. No equilibrium, no deep
pedigree -- this is the level at which the interesting facts about assortative mating become
visible, and it is the sharpest available test of the co-path, because here the co-path's whole
job is to reach the *causes* of the matched phenotypes and every allele covariance must come out
**without being specified by hand**.

Alleles are indexed by PARENTAL ORIGIN, not by transmission
------------------------------------------------------------
``z_mat[i, k]`` is the allele individual ``i`` inherited from their mother at variant ``k``;
``z_pat[i, k]`` the paternal one. This is the design decision that took the longest to settle, and
the reason for it is that "transmitted" is defined relative to a *chosen descendant*, so it is not
a property of an individual and cannot be carried up a pedigree. Maternal/paternal origin is
intrinsic, which makes this motif reusable at every parent-child pair rather than only at the last
meiosis. (A transmitted/non-transmitted split works for exactly one child per couple, because the
"which allele" coin flip is absorbed into which node you *name* transmitted, and a second child
needs a second flip. The two schemes are the same model with the segregation randomness placed
differently and they agree on all second moments. The transmitted/non-transmitted split remains
the right tool for *within-family* questions -- direct vs indirect effects, non-transmitted-allele
designs -- which are out of scope here.)

Both of the mother's alleles feed the child's maternal allele with coefficient **1/2**, not
``sqrt(1/2)``: these are regression coefficients and ``E[child's allele | mother] = (A + B)/2``.
Because every allele node carries the same variance the standardized coefficient is also 1/2, so
there is no ambiguity.

Why the segregation residual exists
-----------------------------------
It looks like it appears from nowhere, so: the child's genotype is **not determined** by the
parents' genotypes -- only its conditional distribution is. Meiosis is genuine new randomness.
``Var(z_mat_parent / 2 + z_pat_parent / 2) = 1/4``, but the child's allele must have variance
``1/2``, and the gap is ``E[Var(allele | parent)] = E[(A - B)^2] / 4 = 1/4``. Omit it and you have
written down *blending inheritance*, which halves the genetic variance every generation. Weighted
by effects, these residuals are exactly the recombination variance
``V_K = sum_k beta_k^2 / 2 = V_A(0) / 2``.

Two caveats, recorded rather than fixed
---------------------------------------
**The segregation variance is 1/4 only while the transmitting parent's two alleles are
uncorrelated.** In general it is ``1/4 - c/2`` with ``c = Cov[z_mat, z_pat]`` in that parent,
because the predictable part ``Var(x/2) = (1 + 2c)/4`` grows and the residual shrinks. Founders
have ``c = 0``, so **this motif is exact**. But the offspring generation acquires
``c = beta_k^2 rho_y / (4 V_P)``, so a *third* generation would need ``1/4 - beta_k^2 rho_y/(8 V_P)``.
Weighted by effects that is ``O(rho_y V_A^2 / M)`` -- the same negligible order as the per-variant
inflation below -- which is why the aggregate ``V_K`` is treated as constant. This is where it
would first bite for the pedigree unroller (task-20260804-151350).

**The segregation residual is uncorrelated with, but not independent of, the parental alleles**:
``Var(s | A, B) = (A - B)^2 / 4``. Uncorrelatedness is all a linear path model needs, so every
covariance here is exact -- but it is not a homoscedastic Gaussian residual, which matters to
anyone later simulating from this model rather than reading covariances off it.

Scale
-----
Node count grows as ``2M + 1`` per individual per variant plus segregation residuals, so keep
``M`` small: ``M = 2`` shows everything including the cross-variant results, and ``M = 3``
matches the coordinator's oracle. See ``docs/scale_alleles.md`` for the measured limit. The
aggregate results come from the ``g``-level model; this motif is not meant to scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from ..core.model import Model
from ..core.units import Units

__all__ = ["AlleleMotif", "allele_motif"]


@dataclass
class AlleleMotif:
    """An allele-level model plus the names and symbols needed to ask questions of it.

    The naming helpers exist so a test or a writeup reads as the algebra does:
    ``motif.z("m", "mat", 0)`` rather than a hand-built ``"z_mat_m_0"``.
    """

    model: Model
    #: per-variant effect symbols, in variant order
    betas: tuple[sp.Expr, ...]
    rho_y: sp.Expr
    V_E: sp.Expr
    founders: tuple[str, str] = ("m", "f")
    children: tuple[str, ...] = ("o",)

    @property
    def n_variants(self) -> int:
        return len(self.betas)

    @property
    def V_A(self) -> sp.Expr:
        """Base-population additive variance, ``sum_k beta_k^2``.

        Exact, not an approximation: founder genotypes have variance 1 by construction
        (two alleles of variance 1/2, uncorrelated), and the founders are in linkage equilibrium.
        """
        return sp.Add(*[b**2 for b in self.betas])

    @property
    def V_P(self) -> sp.Expr:
        """Base-population phenotypic variance. The co-path coefficient is ``rho_y / V_P``."""
        return self.V_A + self.V_E

    @property
    def rho_g(self) -> sp.Expr:
        """``rho_y V_A / V_P`` -- the induced genetic correlation between partners."""
        return self.rho_y * self.V_A / self.V_P

    @property
    def mu(self) -> sp.Expr:
        """The co-path coefficient. NOT the correlation -- see :class:`pathmgr.CoPath`."""
        return self.rho_y / self.V_P

    # -- node names --------------------------------------------------------------------
    def z(self, who: str, origin: str, k: int) -> str:
        """An allele node: ``origin`` is ``"mat"`` or ``"pat"``, by PARENTAL ORIGIN."""
        if origin not in ("mat", "pat"):
            raise ValueError(f"origin must be 'mat' or 'pat', got {origin!r}")
        return f"z_{origin}_{who}_{k}"

    def s(self, who: str, origin: str, k: int) -> str:
        """A segregation residual on the ``origin`` allele of ``who`` at variant ``k``."""
        return f"s_{origin}_{who}_{k}"

    def x(self, who: str, k: int) -> str:
        """The diploid genotype at variant ``k``: ``z_mat + z_pat``."""
        return f"x_{who}_{k}"

    def g(self, who: str) -> str:
        return f"g_{who}"

    def e(self, who: str) -> str:
        return f"e_{who}"

    def y(self, who: str) -> str:
        return f"y_{who}"

    @property
    def individuals(self) -> tuple[str, ...]:
        return self.founders + self.children

    def describe(self) -> str:
        return (
            f"AlleleMotif: {self.n_variants} variant(s), founders {self.founders}, "
            f"children {self.children}, {len(self.model.names)} nodes, "
            f"{len(self.model.copaths)} co-path"
        )


def allele_motif(
    n_variants: int = 2,
    n_children: int = 1,
    effects=None,
    rho_y="rho_y",
    V_E="V_E",
    name: str | None = None,
) -> AlleleMotif:
    """Build the allele-level motif: two founders assorting on ``y``, plus ``n_children``.

    ``effects`` may be a sequence of symbols/expressions/numbers, one per variant; by default
    they are the symbols ``beta_0 ... beta_{M-1}``.

    Everything about the assortment beyond the single co-path between the founders' phenotypes is
    **derived, not specified**: no allele covariance, no genetic correlation between the partners,
    and no linkage disequilibrium is written into the model. All exogenous covariances are zero.

    >>> motif = allele_motif(n_variants=1)
    >>> len(motif.model.copaths)
    1
    >>> motif.model.copath_value("y_m", "y_f") == motif.mu
    True
    """
    if n_variants < 1:
        raise ValueError(f"need at least one variant, got {n_variants}")
    if n_children < 1:
        raise ValueError(f"need at least one child, got {n_children}")

    model = Model(
        name or f"allele motif ({n_variants} variant(s), {n_children} child(ren))",
        units=Units.unstandardized(),
    )
    model.declare("V_E", positive=True)
    if effects is None:
        effects = [model.declare(f"beta_{k}", real=True) for k in range(n_variants)]
    else:
        effects = [model.expr(b) for b in effects]
        if len(effects) != n_variants:
            raise ValueError(
                f"got {len(effects)} effects for {n_variants} variants; they must match"
            )

    founders = ("m", "f")
    children = tuple(f"o{i + 1}" for i in range(n_children)) if n_children > 1 else ("o",)
    motif = AlleleMotif(
        model=model,
        betas=tuple(effects),
        rho_y=model.expr(rho_y),
        V_E=model.expr(V_E),
        founders=founders,
        children=children,
    )

    half = sp.Rational(1, 2)
    quarter = sp.Rational(1, 4)

    def add_individual(who: str) -> None:
        """Phenotype layer: y = g + e, g = sum_k beta_k x_k, x_k = z_mat_k + z_pat_k."""
        model.add_var(motif.g(who), latent=True, label=f"$g_{{{who}}}$")
        model.add_var(motif.e(who), latent=True, label=f"$e_{{{who}}}$")
        model.add_var(motif.y(who), label=f"$y_{{{who}}}$")
        model.add_path(motif.g(who), motif.y(who), 1)
        model.add_path(motif.e(who), motif.y(who), 1)
        model.add_variance(motif.e(who), motif.V_E)
        for k, beta in enumerate(motif.betas):
            # the genotype is observed; the phased alleles behind it are not
            model.add_var(motif.x(who, k), label=f"$x_{{{who},{k}}}$")
            model.add_path(motif.x(who, k), motif.g(who), beta)
            for origin in ("mat", "pat"):
                model.add_var(
                    motif.z(who, origin, k),
                    latent=True,
                    label=f"$z^{{({origin[0]})}}_{{{who},{k}}}$",
                )
                model.add_path(motif.z(who, origin, k), motif.x(who, k), 1)

    for who in founders:
        add_individual(who)
        for k in range(n_variants):
            for origin in ("mat", "pat"):
                # 1/2 each, so a founder's diploid genotype has variance 1 in the base
                # population and the two alleles are uncorrelated (Hardy-Weinberg)
                model.add_variance(motif.z(who, origin, k), half)

    mother, father = founders
    for child in children:
        add_individual(child)
        for k in range(n_variants):
            # the child's maternal allele comes from BOTH of the mother's alleles at 1/2 each,
            # plus genuine meiotic randomness; likewise the paternal one from the father
            for origin, parent in (("mat", mother), ("pat", father)):
                model.add_var(
                    motif.s(child, origin, k),
                    latent=True,
                    label=f"$s^{{({origin[0]})}}_{{{child},{k}}}$",
                )
                model.add_variance(motif.s(child, origin, k), quarter)
                model.add_path(motif.s(child, origin, k), motif.z(child, origin, k), 1)
                for parent_origin in ("mat", "pat"):
                    model.add_path(
                        motif.z(parent, parent_origin, k),
                        motif.z(child, origin, k),
                        half,
                    )

    # THE ONLY cross-couple link. Everything else about the assortment is derived.
    model.add_copath(motif.y(mother), motif.y(father), motif.mu, process="founding couple")

    model.assume("V_A", motif.V_A)
    model.assume("V_P", motif.V_P)
    model.assume("rho_g", motif.rho_g)
    return motif
