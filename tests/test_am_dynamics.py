"""Assortative-mating dynamics and the equilibrium fixed point (task-20260804-151351).

The acceptance milestone for the package: trajectories over the first generations, the equilibrium
solved symbolically rather than approached by unrolling, and every boxed result of Section 2 of
``relative_covariance.tex`` reproduced.

**The fixed-point quadratic is tested at V_P(0) != 1.** The writeup's form
``(1 - V_A0) rho_g^2 - rho_g + rho_y V_A0 = 0`` uses ``V_P^(0) = 1`` to eliminate ``V_P`` and is 20%
wrong at ``V_P^(0) = 1.2``. The package derives the general form and is checked at three scales,
only one of which is 1. This is the third result in this project that is right at ``V_P = 1`` and
wrong elsewhere, so: **never choose test parameters that sum to 1** — or at least never *only* those.
"""

import sympy as sp

import pathmgr as pm
from pathmgr.genetics import AMDynamics, equilibrium, recursion_from_model
from pathmgr.genetics.am import reduce_radicals

# three scales; only the first has V_P(0) = 1
SCALES = [(0.4, 0.6, 0.3), (0.5, 0.7, 0.3), (0.3, 1.2, 0.3)]
#: the coordinator's independently computed rho_g at each, from a t=40 unroll
TRUE_RHO_G = [0.1301659, 0.1357496, 0.0631949]


# ======================================================================================
# the recursion, derived rather than asserted
# ======================================================================================
def test_the_recursion_comes_out_of_the_engine():
    """Boxed: V_A(t+1) = V_A(0)/2 + V_A(t)(1 + rho_g(t))/2."""
    V_A0, V_E, rho_y = sp.symbols("V_A0 V_E rho_y", positive=True)
    V_A_t = sp.Symbol("V_A_t", positive=True)
    derived = recursion_from_model(V_A_t, V_E, rho_y, V_A0)
    rho_g_t = rho_y * V_A_t / (V_A_t + V_E)
    assert sp.simplify(derived - (V_A0 / 2 + V_A_t * (1 + rho_g_t) / 2)) == 0
    # general at every scale, not just V_P = 1
    for V_A0_value, V_E_value, rho_y_value in SCALES:
        values = {V_A0: sp.nsimplify(V_A0_value), V_E: sp.nsimplify(V_E_value),
                  rho_y: sp.nsimplify(rho_y_value), V_A_t: sp.nsimplify(V_A0_value)}
        got = float(sp.N(derived.subs(values)))
        expected = AMDynamics(V_A0_value, V_E_value, rho_y_value).V_A(1)[1]
        assert abs(got - expected) < 1e-12


# ======================================================================================
# the equilibrium, solved not quoted
# ======================================================================================
def test_the_boxed_equilibrium_results_are_derived():
    eq = equilibrium()
    V_A0, V_E, rho_y = eq.symbols["V_A0"], eq.symbols["V_E"], eq.symbols["rho_y"]

    # rho_g = rho_y h2_eq
    assert sp.simplify(eq.rho_g - rho_y * eq.h2) == 0
    # V_A^(eq) = V_A(0) / (1 - rho_g)
    assert sp.simplify(eq.V_A - V_A0 / (1 - eq.rho_g)) == 0
    # and the root really solves the quadratic
    assert sp.simplify(eq.quadratic.lhs.subs(eq.symbols["rho_g"], eq.rho_g)) == 0


def test_the_derived_quadratic_is_the_general_form_not_the_writeup_form():
    """The writeup's form eliminates V_P using V_P(0) = 1; the package must not inherit that."""
    eq = equilibrium()
    V_A0, V_E, rho_y, rho_g = (eq.symbols[k] for k in ("V_A0", "V_E", "rho_y", "rho_g"))

    general = V_E * rho_g**2 - rho_g * (V_A0 + V_E) + rho_y * V_A0
    assert sp.simplify(sp.expand(eq.quadratic.lhs) - sp.expand(general)) == 0

    # it reduces to the writeup's form exactly when V_P(0) = 1, and only then
    at_unit = {V_E: 1 - V_A0}
    assert sp.simplify(
        (eq.quadratic.lhs - eq.writeup_quadratic.lhs).subs(at_unit)
    ) == 0
    assert sp.simplify(eq.quadratic.lhs - eq.writeup_quadratic.lhs) != 0


def test_the_root_vanishes_as_rho_y_goes_to_zero():
    eq = equilibrium()
    rho_y = eq.symbols["rho_y"]
    assert reduce_radicals(eq.rho_g.subs(rho_y, 0)) == 0
    # and under random mating the equilibrium is the base population
    V_A0 = eq.symbols["V_A0"]
    assert reduce_radicals(eq.V_A.subs(rho_y, 0) - V_A0) == 0


def test_equilibrium_matches_an_independent_unroll_at_three_scales():
    """Including two with V_P(0) != 1, where the writeup's form would be 20% out."""
    eq = equilibrium()
    for (V_A0, V_E, rho_y), expected in zip(SCALES, TRUE_RHO_G):
        got = eq.evaluate({"V_A0": V_A0, "V_E": V_E, "rho_y": rho_y})["rho_g"]
        assert abs(got - expected) < 1e-6, f"V_P(0)={V_A0 + V_E}: {got} vs {expected}"


def test_the_writeup_form_is_wrong_away_from_unit_phenotypic_variance():
    """Pin the trap itself, so nobody 'simplifies' the general form back to the writeup's."""
    eq = equilibrium()
    rho_g = eq.symbols["rho_g"]
    for (V_A0, V_E, rho_y), expected in zip(SCALES, TRUE_RHO_G):
        values = {"V_A0": V_A0, "V_E": V_E, "rho_y": rho_y}
        writeup = eq.substitute(values).writeup_quadratic
        roots = [r for r in sp.solve(writeup, rho_g) if abs(float(sp.N(r))) < 1]
        writeup_value = float(sp.N(min(roots, key=lambda r: abs(float(sp.N(r))))))
        if abs((V_A0 + V_E) - 1.0) < 1e-12:
            assert abs(writeup_value - expected) < 1e-6, "correct at V_P(0) = 1"
        else:
            assert abs(writeup_value - expected) > 1e-3, "and wrong elsewhere"


def test_the_worked_numeric_example_from_section_2():
    """h2(0) = 0.40, rho_y = 0.30 -> rho_g = 0.130, V_A_eq = 0.460, h2_eq = 0.434."""
    eq = equilibrium().evaluate({"V_A0": 0.4, "V_E": 0.6, "rho_y": 0.3})
    assert round(eq["rho_g"], 3) == 0.130
    assert round(eq["V_A"], 3) == 0.460
    assert round(eq["h2"], 3) == 0.434
    # the first-order approximation rho_y h2(0) understates it, as the writeup says
    assert 0.3 * 0.4 < eq["rho_g"]


# ======================================================================================
# trajectories and convergence
# ======================================================================================
def test_trajectory_reproduces_the_independent_eight_generation_unroll():
    dynamics = AMDynamics(0.4, 0.6, 0.3)
    trajectory = [round(v, 5) for v in dynamics.V_A(8)]
    assert trajectory == [0.4, 0.424, 0.43833, 0.44692, 0.45208, 0.45518, 0.45704, 0.45816, 0.45884]
    # rho_g(t) is rho_y V_A(t)/V_P(t) at every generation, by construction
    data = dynamics.trajectory(8)
    for V_A, V_P, rho_g in zip(data["V_A"], data["V_P"], data["rho_g"]):
        assert abs(rho_g - 0.3 * V_A / V_P) < 1e-15
    # approaching, but not yet at, the analytic 0.1301659 -- still short at t = 8
    assert 0.1299 < data["rho_g"][-1] < 0.1301659


def test_trajectory_carries_every_quantity_the_task_asks_for():
    data = AMDynamics(0.5, 0.7, 0.3).trajectory(6)  # V_P(0) = 1.2, deliberately not 1
    for key in ("V_A", "rho_g", "h2", "sibling_corr", "parent_offspring_corr", "partner_corr"):
        assert len(data[key]) == 7, key
    # partners are held at rho_y by construction; everything else moves
    assert data["partner_corr"] == [0.3] * 7
    assert data["V_A"][0] < data["V_A"][-1]
    assert data["rho_g"][0] < data["rho_g"][-1]
    assert data["h2"][0] < data["h2"][-1]
    # parent-offspring exceeds sibling at every generation once assortment is on
    assert all(
        p > s for p, s in zip(data["parent_offspring_corr"], data["sibling_corr"])
    )


def test_trajectories_converge_to_the_solved_equilibrium():
    """Convergence as a TEST, not just a figure -- and at three scales."""
    for V_A0, V_E, rho_y in SCALES:
        dynamics = AMDynamics(V_A0, V_E, rho_y)
        target = dynamics.equilibrium()
        late = dynamics.trajectory(60)
        assert abs(late["V_A"][-1] - target["V_A"]) < 1e-9
        assert abs(late["rho_g"][-1] - target["rho_g"]) < 1e-9
        assert abs(late["h2"][-1] - target["h2"]) < 1e-9
        # monotone approach from below
        assert all(a <= b + 1e-15 for a, b in zip(late["V_A"], late["V_A"][1:]))


def test_convergence_takes_six_to_ten_generations():
    """An independent reproduction of Sunde et al.'s reported range."""
    dynamics = AMDynamics(0.4, 0.6, 0.3)
    assert dynamics.generations_to_converge(tolerance=0.01) == 6
    assert 6 <= dynamics.generations_to_converge(tolerance=0.001) <= 12
    # and it is still short of equilibrium at 8, as the coordinator measured
    assert abs(dynamics.V_A(8)[-1] - dynamics.equilibrium()["V_A"]) > 1e-4


def test_random_mating_is_a_fixed_point():
    dynamics = AMDynamics(0.4, 0.6, 0.0)
    assert dynamics.V_A(10) == [0.4] * 11
    assert dynamics.equilibrium()["rho_g"] == 0.0
    assert abs(dynamics.equilibrium()["V_A"] - 0.4) < 1e-12


def test_unit_phenotypic_variance_is_flagged():
    """It is the writeup's convention, but it masks scale errors, so it is never silent."""
    assert AMDynamics(0.4, 0.6, 0.3).unit_phenotypic_variance is True
    assert AMDynamics(0.5, 0.7, 0.3).unit_phenotypic_variance is False


def test_the_plot_is_produced(tmp_path):
    from pathmgr.genetics import plot_trajectories

    out = plot_trajectories(AMDynamics(0.4, 0.6, 0.3), tmp_path / "traj.png", n_generations=10)
    assert out.exists() and out.stat().st_size > 5000
