"""The assortative-mating lineage: the directed encoding of assortment, checked substantively.

These assertions previously lived in ``check()`` inside ``scripts/profile_ram.py``, so they only
ran when a developer happened to regenerate the timing report. The lineage is a *reference
implementation* of the trickiest encoding in the project -- see
``docs/assortment_representation_trap.md`` -- and a reference implementation that CI never checks
is not one. Nothing here is new: the model, the expected values and the reasoning are moved
verbatim from that script, which now imports the model from ``tests/battery.py``.

Battery membership already gives the lineage two-engine agreement coverage. Agreement says the
two engines compute the same thing; it cannot say they compute the *right* thing. That is what
these three properties are for.
"""

from __future__ import annotations

import pytest
import sympy as sp

import pathmgr as pm

from battery import lineage

DEPTHS = [1, 2, 3, 4, 5]


@pytest.fixture(scope="module")
def engines():
    """One model and engine per depth, built once -- Sigma is the expensive part."""
    return {d: (lambda m: (m, pm.RAMEngine(m)))(lineage(d)) for d in DEPTHS}


def _symbols(model):
    V_A, V_E, rho_g, rho_y = (model.sym(s) for s in ("V_A_eq", "V_E", "rho_g", "rho_y"))
    return V_A, V_E, rho_g, rho_y, {rho_g: rho_y * V_A / (V_A + V_E)}


@pytest.mark.parametrize("depth", DEPTHS)
def test_the_lineage_encoding_validates(depth, engines):
    """No errors: in particular, no bidirected edge on an endogenous genetic value."""
    model, _ = engines[depth]
    errors = [i for i in model.validate() if i.severity == "error"]
    assert not errors, f"depth {depth}: {errors}"


@pytest.mark.parametrize("depth", DEPTHS)
def test_equilibrium_is_self_consistent(depth, engines):
    """The child's genetic variance is still V_A_eq -- otherwise it is not an equilibrium."""
    model, engine = engines[depth]
    V_A, _, _, _, _ = _symbols(model)
    assert sp.simplify(engine.var(f"g_{depth}c") - V_A) == 0, f"depth {depth}: Var[g] drifted"


@pytest.mark.parametrize("depth", DEPTHS)
def test_partners_correlate_at_rho_y(depth, engines):
    """By construction, so this checks the construction.

    Compared squared: the correlation carries a ``sqrt((V_A_eq + V_E)**2)`` that ``simplify()``
    will not reduce even though both symbols are declared positive -- a small reminder that sympy
    needs help at the points we choose, which is why simplification here is explicit.
    """
    model, engine = engines[depth]
    _, _, _, rho_y, fixed_point = _symbols(model)
    mate_corr = engine.corr("y_0m", "y_0p").subs(fixed_point)
    assert sp.simplify((mate_corr / rho_y) ** 2 - 1) == 0, (
        f"depth {depth}: mate corr {sp.simplify(mate_corr)}"
    )


@pytest.mark.parametrize("depth", DEPTHS)
def test_lineal_relative_result_of_the_writeup(depth, engines):
    """``eq:am-level3-lin``: Cov[y_a, y_b] = V_A_eq (1+rho_y)/2 ((1+rho_g)/2)^(d-1).

    The reason the directed encoding is trusted at all. The first version of this model used a
    bidirected edge and its covariances decayed as ``2**-d`` with no ``(1 + rho_g)`` accumulation
    whatsoever -- this assertion is what caught that.
    """
    model, engine = engines[depth]
    V_A, _, rho_g, rho_y, fixed_point = _symbols(model)
    got = sp.simplify(engine.cov("y_0m", f"y_{depth}c").subs(fixed_point))
    want = sp.simplify(
        V_A * (1 + rho_y) / 2 * ((1 + rho_g.subs(fixed_point)) / 2) ** (depth - 1)
    )
    assert sp.simplify(got - want) == 0, f"depth {depth}: lineal mismatch"
