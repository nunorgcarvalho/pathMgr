"""Closed-form symbolic covariance engine (RAM).

**Not yet implemented -- this is task-20260804-151347.**

Will compute the model-implied covariance matrix by the RAM identity

    Sigma = F (I - A)^-1 S (I - A)^-T F^T

over a :class:`pathmgr.core.model.Model`, and expose a query API for the covariance or
correlation between any two variables (including latent and intermediate ones).

Design notes carried forward from the specification task:

- Cache ``(I - A)^-1`` keyed on ``model.revision``; the symbolic inverse is the expensive
  step and models are mutable builders.
- For a recursive (acyclic) model, ``(I - A)^-1`` is the sum of powers of ``A`` and can be
  had by forward substitution in topological order -- much cheaper than a general inverse.
- Do not assume ``simplify()`` is cheap. Simplify at defined points only.
- Results must carry ``model.units`` so a returned expression is never scale-ambiguous.
- Honour ``model.substitutions()`` / ``model.assumptions`` as an opt-in, not silently.
"""

from __future__ import annotations

__all__: list[str] = []
