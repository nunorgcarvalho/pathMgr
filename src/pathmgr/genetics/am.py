"""Assortative-mating dynamics and the equilibrium fixed point.

**Not yet implemented -- this is task-20260804-151351.**

Two separate things, deliberately:

- **Generation-by-generation dynamics.** Iterate the recursion
  ``V_A(t+1) = V_A(0)/2 + V_A(t) (1 + rho_g(t))/2`` and report how V_A, rho_g, h^2 and
  relative correlations evolve over the first few generations.
- **The equilibrium fixed point**, solved explicitly -- never approached by unrolling.
  ``rho_g = rho_y h^2_eq`` and ``V_A_eq = V_A(0)/(1 - rho_g)`` are coupled; substituting
  gives a quadratic in ``rho_g``, and the root taken is the one with ``rho_g -> 0`` as
  ``rho_y -> 0``.

Validation targets (hand-derived, in popstatgenwriteups'
``writeups/statistical_genetics/relative_covariance/relative_covariance.tex``): the two
boxed results above, the collateral result ``V_A_eq ((1 + rho_g)/2)^d``, and the lineal
result ``V_A_eq ((1 + rho_y)/2) ((1 + rho_g)/2)^(d-1)``. If pathMgr and the writeup
disagree, the disagreement is a finding -- surface it, do not conform to either.
"""

__all__: list[str] = []
