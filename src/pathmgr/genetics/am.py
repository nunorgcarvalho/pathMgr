"""Assortative-mating dynamics and the equilibrium fixed point.

Two separate things, deliberately kept apart:

- **Generation-by-generation dynamics.** Iterate the recursion forward from a randomly mating base
  population and watch ``V_A``, ``rho_g``, ``h^2`` and relative-pair correlations move. This is what
  the user asked to *see* -- the initial generations, not just the endpoint.
- **The equilibrium fixed point**, solved **explicitly and symbolically**. Never approached by
  unrolling until things stop moving. The system is coupled -- ``rho_g`` depends on ``h^2_eq``,
  which depends on ``V_A^(eq)``, which depends back on ``rho_g`` -- and the elimination is done here
  with sympy rather than quoted.

The recursion is not asserted here
----------------------------------
:func:`recursion_from_model` reads it off the **engine**, applied to a one-generation pedigree:
``Var[g_child]`` comes out as ``V_K + V_A(t)(1 + rho_g(t))/2`` because that is what the path model
implies, not because it was written down. Everything below is built on that, so the boxed results
are reproductions rather than restatements.

The fixed-point quadratic is NOT the writeup's form
---------------------------------------------------
``relative_covariance.tex`` gives ``(1 - V_A0) rho_g^2 - rho_g + rho_y V_A0 = 0``. That is correct
*there*, because the writeup fixes ``V_P^(0) = 1`` as its standardization convention and uses it to
eliminate ``V_P``. It is **wrong for any other scale** -- 20% off at ``V_P^(0) = 1.2``. The general
form, which this module derives and which reduces to the writeup's exactly when
``V_A^(0) + V_E = 1``, is

    V_E rho_g^2  -  rho_g V_P^(0)  +  rho_y V_A^(0)  =  0

This is the third instance in this project of a result that is right at ``V_P = 1`` and wrong
elsewhere, so the standing rule applies to anything added here: **never choose test parameters that
sum to 1.**
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sympy as sp

from ..core.ram import RAMEngine
from .pedigree import AMParameters, am_pedigree, g_level_model

__all__ = [
    "AMDynamics",
    "reduce_radicals",
    "Equilibrium",
    "equilibrium",
    "plot_trajectories",
    "recursion_from_model",
]


def reduce_radicals(expression: sp.Expr) -> sp.Expr:
    """Factor inside every square root, then simplify.

    sympy will not reduce ``sqrt(V_A0**2 + 2*V_A0*V_E + V_E**2)`` to ``V_A0 + V_E`` even with both
    symbols declared positive -- it does not factor the radicand on its own. Since the equilibrium
    root's discriminant is exactly ``V_P^(0)**2 - 4 V_E rho_y V_A^(0)``, which collapses to a
    perfect square at ``rho_y = 0``, the root-selection test needs this or it silently fails to
    recognise the root that vanishes. Same limitation bit the mate-correlation check in
    task-20260804-151347.
    """
    replaced = expression.replace(
        lambda node: node.is_Pow and node.exp == sp.Rational(1, 2),
        lambda node: sp.sqrt(sp.factor(node.base)),
    )
    return sp.simplify(replaced)


def recursion_from_model(
    V_A_t: sp.Expr, V_E: sp.Expr, rho_y: sp.Expr, V_A0: sp.Expr
) -> sp.Expr:
    """``V_A(t+1)``, read off the engine rather than asserted.

    Builds a one-generation pedigree, asks the RAM engine for the implied ``Var[g_child]``, and
    returns it with the parents' generation values substituted in. The returned expression *is*
    ``V_K + V_A(t)(1 + rho_g(t))/2`` -- but derived, so the boxed recursion in
    ``relative_covariance.tex`` is reproduced rather than restated.
    """
    # the parents' additive variance and the segregation variance are DIFFERENT quantities here:
    # V_A(t) moves, V_K stays at V_A(0)/2. Building with a distinct base symbol and an explicit
    # V_K keeps them apart -- sharing one symbol would silently tie them together.
    pedigree = am_pedigree(1, children_per_couple=1)
    unrolled = g_level_model(
        pedigree,
        AMParameters(
            base_additive_variance="V_A_parents",
            segregation_variance=sp.Symbol("V_K_fixed"),
        ),
    )
    child = pedigree.children_of(pedigree.couples[0])[0]
    implied = RAMEngine(unrolled.model).var(f"g_{child}")
    # substitute using the MODEL'S OWN symbols. A symbol built here would carry different sympy
    # assumptions from the one the model registered, and would silently fail to match.
    return sp.together(
        implied.subs(
            {
                unrolled.V_A[0]: V_A_t,
                unrolled.V_K: V_A0 / 2,
                unrolled.V_E: V_E,
                unrolled.rho_y: rho_y,
            }
        )
    )


@dataclass(frozen=True)
class Equilibrium:
    """The fixed point, solved symbolically. Every field is an expression, not a number."""

    rho_g: sp.Expr
    V_A: sp.Expr
    V_P: sp.Expr
    h2: sp.Expr
    #: the polynomial in rho_g whose root this is, as an ``Eq``
    quadratic: sp.Eq
    #: the writeup's ``V_P(0) = 1`` form, for comparison -- NOT what was solved
    writeup_quadratic: sp.Eq
    #: the exact symbols these expressions are written in. Carried rather than left to the caller
    #: to reconstruct: a symbol rebuilt with different sympy assumptions does not compare equal,
    #: so `subs` against it silently does nothing.
    symbols: dict = None

    def _resolve(self, values: dict) -> dict:
        """Accept ``{"V_A0": 0.4}`` as well as ``{symbol: value}``, and rationalise numbers."""
        resolved = {}
        for key, value in values.items():
            symbol = self.symbols[key] if isinstance(key, str) else key
            resolved[symbol] = (
                sp.nsimplify(value, rational=True) if isinstance(value, (int, float)) else value
            )
        return resolved

    def substitute(self, values: dict) -> "Equilibrium":
        resolved = self._resolve(values)
        return Equilibrium(
            rho_g=self.rho_g.subs(resolved),
            V_A=self.V_A.subs(resolved),
            V_P=self.V_P.subs(resolved),
            h2=self.h2.subs(resolved),
            quadratic=self.quadratic.subs(resolved),
            writeup_quadratic=self.writeup_quadratic.subs(resolved),
            symbols=self.symbols,
        )

    def evaluate(self, values: dict) -> dict[str, float]:
        substituted = self.substitute(values)
        return {
            "rho_g": float(sp.N(substituted.rho_g)),
            "V_A": float(sp.N(substituted.V_A)),
            "V_P": float(sp.N(substituted.V_P)),
            "h2": float(sp.N(substituted.h2)),
        }

    def __str__(self) -> str:
        return (
            f"equilibrium:\n"
            f"  quadratic  {sp.sstr(self.quadratic)}\n"
            f"  rho_g    = {sp.sstr(self.rho_g)}\n"
            f"  V_A^(eq) = {sp.sstr(self.V_A)}\n"
            f"  h2_eq    = {sp.sstr(self.h2)}"
        )


def equilibrium(
    V_A0: sp.Expr | str = "V_A0",
    V_E: sp.Expr | str = "V_E",
    rho_y: sp.Expr | str = "rho_y",
) -> Equilibrium:
    """Solve the coupled fixed point symbolically, deriving the quadratic rather than quoting it.

    The elimination, done with sympy and not by hand:

    1. Set ``V_A(t+1) = V_A(t) = V_A_eq`` in the recursion the engine implies, and solve. That
       gives ``V_A_eq = V_A(0) / (1 - rho_g)`` -- the first boxed result, derived.
    2. Substitute it into ``rho_g = rho_y h^2_eq = rho_y V_A_eq / (V_A_eq + V_E)``. Clearing
       denominators leaves a quadratic in ``rho_g``.
    3. Take the root with ``rho_g -> 0`` as ``rho_y -> 0``.

    >>> eq = equilibrium()
    >>> sp.simplify(eq.V_A - sp.Symbol("V_A0", positive=True) / (1 - eq.rho_g))
    0
    """
    V_A0 = sp.Symbol(V_A0, positive=True) if isinstance(V_A0, str) else V_A0
    V_E = sp.Symbol(V_E, positive=True) if isinstance(V_E, str) else V_E
    rho_y = sp.Symbol(rho_y, real=True) if isinstance(rho_y, str) else rho_y
    rho_g, V_A_eq = sp.symbols("rho_g V_A_eq", real=True)

    # 1. the recursion, from the engine, at its fixed point.
    next_V_A = recursion_from_model(V_A_eq, V_E, rho_y, V_A0)
    # Rewrite it in terms of rho_g by substituting the DEFINITION of rho_g, rearranged:
    # rho_g = rho_y V_A_eq/(V_A_eq + V_E)  <=>  rho_y = rho_g (V_A_eq + V_E)/V_A_eq.
    # An exact identity, and it makes the fixed point LINEAR in V_A_eq -- which is what lets the
    # elimination follow the writeup's route instead of producing a quartic.
    in_rho_g = sp.simplify(next_V_A.subs(rho_y, rho_g * (V_A_eq + V_E) / V_A_eq))
    solutions = sp.solve(sp.Eq(V_A_eq, in_rho_g), V_A_eq)
    if not solutions:  # pragma: no cover - the linear solve always succeeds
        raise RuntimeError("could not solve the variance fixed point")
    V_A_solution = sp.simplify(solutions[0])

    # 2. close the loop through rho_g = rho_y h2_eq, and clear denominators
    closure = sp.Eq(rho_g, rho_y * V_A_solution / (V_A_solution + V_E))
    polynomial = sp.Poly(sp.numer(sp.together(closure.lhs - closure.rhs)), rho_g)
    quadratic = sp.Eq(sp.expand(polynomial.as_expr()), 0)

    # 3. the root that vanishes with rho_y
    roots = sp.solve(quadratic, rho_g)
    chosen = None
    for candidate in roots:
        if reduce_radicals(candidate.subs(rho_y, 0)) == 0:
            chosen = sp.simplify(candidate)
            break
    if chosen is None:  # pragma: no cover
        raise RuntimeError(f"no root vanishes as rho_y -> 0; candidates {roots}")

    V_A_at_equilibrium = sp.simplify(V_A_solution.subs(rho_g, chosen))
    V_P_at_equilibrium = sp.simplify(V_A_at_equilibrium + V_E)
    writeup = sp.Eq(
        sp.expand((1 - V_A0) * rho_g**2 - rho_g + rho_y * V_A0), 0
    )
    return Equilibrium(
        rho_g=chosen,
        V_A=V_A_at_equilibrium,
        V_P=V_P_at_equilibrium,
        h2=sp.simplify(V_A_at_equilibrium / V_P_at_equilibrium),
        quadratic=quadratic,
        writeup_quadratic=writeup,
        symbols={"V_A0": V_A0, "V_E": V_E, "rho_y": rho_y, "rho_g": rho_g, "V_A_eq": V_A_eq},
    )


@dataclass
class AMDynamics:
    """Generation-by-generation trajectories from a randomly mating base population.

    Every quantity is indexed to the generation it belongs to. The relative-pair covariances take
    the **parents'** generation, which is the index that matters: using the offspring's ``V_A`` is
    wrong by ~1.6e-2 in the first generation.
    """

    V_A0: float = 0.4
    V_E: float = 0.6
    rho_y: float = 0.3

    def __post_init__(self) -> None:
        if abs((self.V_A0 + self.V_E) - 1.0) < 1e-12:
            # not an error -- the writeup's own convention -- but the masking that has now caught
            # three separate results in this project, so it should never be silent
            self.unit_phenotypic_variance = True
        else:
            self.unit_phenotypic_variance = False

    @property
    def V_K(self) -> float:
        """Segregation variance, held constant at ``V_A(0)/2``. See :mod:`pathmgr.genetics.pedigree`."""
        return self.V_A0 / 2

    def V_A(self, n_generations: int) -> list[float]:
        """``V_A(0..n)`` by iterating the recursion the engine implies."""
        values = [float(self.V_A0)]
        for _ in range(n_generations):
            current = values[-1]
            rho_g = self.rho_y * current / (current + self.V_E)
            values.append(self.V_K + current * (1 + rho_g) / 2)
        return values

    def trajectory(self, n_generations: int = 10) -> dict[str, list[float]]:
        """Every tracked quantity over ``0..n``, as plain lists ready to plot.

        ``sibling`` and ``parent_offspring`` are **correlations** (covariance over ``V_P(t)``), so
        they are comparable across generations even though the scale is moving underneath them.
        """
        V_A = self.V_A(n_generations)
        V_P = [v + self.V_E for v in V_A]
        rho_g = [self.rho_y * a / p for a, p in zip(V_A, V_P)]
        h2 = [a / p for a, p in zip(V_A, V_P)]
        # indexed to the PARENTS' generation t; the pair itself lives in generation t+1
        sibling = [a * (1 + g) / 2 for a, g in zip(V_A, rho_g)]
        parent_offspring = [a * (1 + self.rho_y) / 2 for a in V_A]
        return {
            "generation": list(range(n_generations + 1)),
            "V_A": V_A,
            "V_P": V_P,
            "rho_g": rho_g,
            "h2": h2,
            "sibling_cov": sibling,
            "parent_offspring_cov": parent_offspring,
            "sibling_corr": [c / p for c, p in zip(sibling, V_P)],
            "parent_offspring_corr": [c / p for c, p in zip(parent_offspring, V_P)],
            "partner_corr": [self.rho_y] * (n_generations + 1),
        }

    def equilibrium(self) -> dict[str, float]:
        """The analytic fixed point at these parameters -- solved, never unrolled."""
        return equilibrium().evaluate(
            {"V_A0": self.V_A0, "V_E": self.V_E, "rho_y": self.rho_y}
        )

    def generations_to_converge(self, tolerance: float = 0.01, limit: int = 200) -> int:
        """First generation at which ``V_A(t)`` is within ``tolerance`` *relative* of equilibrium.

        Sunde et al. 2024 put this at roughly six to ten generations; reproducing that number
        independently is the point of the test that uses this.
        """
        target = self.equilibrium()["V_A"]
        values = self.V_A(limit)
        for t, value in enumerate(values):
            if abs(value - target) / target <= tolerance:
                return t
        raise RuntimeError(f"not within {tolerance} of {target} after {limit} generations")


def plot_trajectories(
    dynamics: AMDynamics,
    path: str | Path,
    n_generations: int = 10,
    dpi: int = 180,
) -> Path:
    """Plot the dynamics over the first generations, with the equilibrium marked.

    A first-class deliverable, not a nicety: the user asked to *see* the initial generations. Each
    panel draws the analytic equilibrium as a dashed line, so convergence is visible rather than
    asserted. matplotlib is imported lazily, as everywhere else in this package.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib: pip install 'pathmgr[render]'"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = dynamics.trajectory(n_generations)
    fixed = dynamics.equilibrium()
    generations = data["generation"]

    panels = [
        ("V_A", "$V_A(t)$", fixed["V_A"]),
        ("rho_g", r"$\rho_g(t)$", fixed["rho_g"]),
        ("h2", "$h^2(t)$", fixed["h2"]),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
    flat = axes.ravel()
    for axis, (key, label, target) in zip(flat, panels):
        axis.plot(generations, data[key], marker="o", markersize=3.5, color="#1F77B4")
        axis.axhline(target, linestyle="--", linewidth=1, color="#B03A2E",
                     label=f"equilibrium = {target:.5f}")
        axis.set_title(label, fontsize=11)
        axis.set_xlabel("generation $t$")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)

    relatives = flat[3]
    relatives.plot(generations, data["sibling_corr"], marker="o", markersize=3.5,
                   label="full siblings", color="#1F77B4")
    relatives.plot(generations, data["parent_offspring_corr"], marker="s", markersize=3.5,
                   label="parent-offspring", color="#2E8B57")
    relatives.plot(generations, data["partner_corr"], linestyle=":", linewidth=1.6,
                   label=r"partners ($\rho_y$, held fixed)", color="#B03A2E")
    relatives.set_title("relative-pair correlations", fontsize=11)
    relatives.set_xlabel("generation $t$ (the PARENTS' generation)")
    relatives.legend(fontsize=7)
    relatives.grid(alpha=0.25)

    scale = f"$V_P(0)={dynamics.V_A0 + dynamics.V_E:g}$"
    figure.suptitle(
        f"assortative mating from a randomly mating base "
        f"($V_A(0)={dynamics.V_A0:g}$, $V_E={dynamics.V_E:g}$, "
        rf"$\rho_y={dynamics.rho_y:g}$; {scale})",
        fontsize=11,
    )
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path
