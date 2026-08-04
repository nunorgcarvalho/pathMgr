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
"""

__all__: list[str] = []
