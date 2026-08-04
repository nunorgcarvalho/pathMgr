"""Pedigree construction: unroll generations into a path model.

**Not yet implemented -- this is task-20260804-151350.**

Builds a :class:`pathmgr.core.model.Model` forward from a randomly mating base population.
Per individual: latent ``g`` and ``e``, observed ``y = g + e``. Per mating: transmission
``g_o = (g_m + g_f)/2 + s_o`` with segregation deviation ``s_o`` of variance ``V_K``, plus
the assortment covariance between mates' genetic values -- and, critically, the cross term
``Cov[e_m, g_f] = rho_g * V_E``, which is what makes lineal and collateral relatives differ
under assortative mating.

The unrolled model is finite by construction and therefore safe to trace; equilibrium is a
separate fixed-point solve in :mod:`pathmgr.genetics.am`, never "unroll a lot".

READ THIS FIRST: how assortment must be encoded
-----------------------------------------------
Found the hard way while profiling the RAM engine (task-20260804-151347); full write-up in
``docs/profile_ram.md``, and a working reference implementation in ``scripts/profile_ram.py``.

Mates' genetic values are correlated, so the obvious encoding is a bidirected edge
``g_mother <-> g_father``. **That is correct only while both mates are exogenous**, as in a
founding couple. The moment a mate is themselves a child in the pedigree, their genetic value
is endogenous, and a bidirected edge is a covariance between *disturbances*, not between
variables. A child's genetic value is fully determined by its parents plus its segregation
term, so its disturbance is identically zero and cannot covary with anything -- asserting that
it does produces a ``Sigma`` that is **not positive semi-definite**, with an implied
correlation above 1 and nothing else to signal the error. The first draft of the profiling
lineage did exactly this: the covariances decayed as ``2^-d`` with no ``(1 + rho_g)``
accumulation at all, silently disagreeing with the writeup. ``Model.validate()`` now flags it.

The correct encoding makes assortment a **directed path from the focal individual's phenotype
to the partner's components** (a "copath", in Cloninger's terminology)::

    y_focal -> g_partner    coefficient  rho_g
    y_focal -> e_partner    coefficient  rho_y * V_E / V_P

with the partner's residual variances reduced to preserve ``Var[g] = V_A_eq`` and
``Var[e] = V_E``, and a disturbance covariance ``g_partner <-> e_partner`` of
``-rho_g * (rho_y V_E / V_P) * V_P`` to cancel the spurious within-individual g-e covariance
that the two shared loadings would otherwise induce (GE-indep requires it to be zero).

This encoding reproduces ``relative_covariance.tex`` exactly, keeps the equilibrium
self-consistent (``Var[g]`` preserved generation to generation), and makes
``Cov[e_partner, g_focal] = rho_g V_E`` -- the term behind the lineal/collateral asymmetry --
fall out automatically instead of needing an edge of its own.
"""

__all__: list[str] = []
