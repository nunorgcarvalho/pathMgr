"""Wright path-tracing engine: covariance by explicit chain enumeration.

**Not yet implemented -- this is task-20260804-151348.**

The second, independent engine. Enumerates the admissible Wright chains between two
variables and returns the decomposition -- which chains exist and what each contributes --
because the *enumeration* is the insight the matrix identity hides.

The standing correctness property of this package is that this engine and
:mod:`pathmgr.core.ram` agree symbolically on every model. That agreement is the main
defense against subtle tracing bugs.

Design notes carried forward from the specification task:

- Wright's rules in their classic form assume standardized variables; the genetics is
  written unstandardized. The enumeration must be stated in a form valid for
  unstandardized models (each chain: product of directed coefficients times the one
  bidirected value it passes through), and must consult ``model.units``.
- At assortative-mating equilibrium the ancestral graph extends back forever. A naive
  tracer will not terminate or will silently truncate. Finite-generation unrolling
  terminates by construction; equilibrium must come from an explicit fixed-point solve --
  never from "unroll a lot". Enumeration must fail loudly, not truncate, on a graph it
  cannot finish.
"""

from __future__ import annotations

__all__: list[str] = []
