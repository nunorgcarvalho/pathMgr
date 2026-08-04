"""The two validation models of task-20260804-151346 §3.

Each is encoded by hand in the specification API and checked to round-trip into a sane
internal representation: correct latent/observed split, correct symbolic RAM matrices.

There is no engine yet (that is task-20260804-151347). ``ram_sigma`` from conftest is a
deliberate four-line **spike**, not the engine: it exists only to confirm that what the API
*says* is what we meant, and to leave a known-good target for 151347.
"""

import sympy as sp

import pathmgr as pm

from conftest import ram_sigma


# ======================================================================================
# Model 1 -- plain bivariate regression: y ~ b1*x1 + b2*x2, x1 ~~ x2
# The standard SEM smoke test. All variables observed; y's bidirected self-edge is its
# *residual* variance, not its total variance.
# ======================================================================================
def bivariate_regression() -> pm.Model:
    m = pm.Model("bivariate regression")
    for name in ("V_1", "V_2", "V_r"):
        m.declare(name, positive=True)
    m.add_vars("x1", "x2", "y")
    m.add_path("x1", "y", "b1")
    m.add_path("x2", "y", "b2")
    m.add_variance("x1", "V_1")
    m.add_variance("x2", "V_2")
    m.add_variance("y", "V_r")  # residual
    m.add_cov("x1", "x2", "c12")
    return m


def test_bivariate_regression_roundtrips():
    m = bivariate_regression()

    assert m.observed == ("x1", "x2", "y")
    assert m.latent == ()
    assert set(m.exogenous) == {"x1", "x2"}
    assert m.endogenous == ("y",)
    assert m.is_recursive
    assert m.validate() == []

    b1, b2, V_1, V_2, V_r, c12 = (
        m.sym(s) for s in ("b1", "b2", "V_1", "V_2", "V_r", "c12")
    )
    A, S, F, order = m.ram()
    assert order == ("x1", "x2", "y")
    assert A == sp.Matrix([[0, 0, 0], [0, 0, 0], [b1, b2, 0]])
    assert S == sp.Matrix([[V_1, c12, 0], [c12, V_2, 0], [0, 0, V_r]])
    assert F == sp.eye(3)


def test_bivariate_regression_implies_the_textbook_covariances():
    """Spike, not the engine: confirms the spec says what the textbook says."""
    m = bivariate_regression()
    b1, b2, V_1, V_2, V_r, c12 = (
        m.sym(s) for s in ("b1", "b2", "V_1", "V_2", "V_r", "c12")
    )
    Sigma, i = ram_sigma(m)

    assert sp.simplify(Sigma[i["x1"], i["y"]] - (b1 * V_1 + b2 * c12)) == 0
    assert sp.simplify(Sigma[i["x2"], i["y"]] - (b2 * V_2 + b1 * c12)) == 0
    expected_Vy = b1**2 * V_1 + b2**2 * V_2 + 2 * b1 * b2 * c12 + V_r
    assert sp.simplify(Sigma[i["y"], i["y"]] - expected_Vy) == 0


# ======================================================================================
# Model 2 -- Section 1 of relative_covariance.tex
# y_i = g_i + e_i for two individuals, with a genetic covariance between them.
# g and e are LATENT (never observed); only y is observed. Written unstandardized in the
# components V_A and V_E, with the base-population standardization V_A + V_E = 1 recorded
# as a side relation rather than baked into the edges.
# ======================================================================================
def relative_covariance_section1() -> pm.Model:
    m = pm.Model(
        "relative covariance, Section 1 (random mating, independent environment)",
        units=pm.Units.unstandardized(),
    )
    for name in ("V_A", "V_E"):
        m.declare(name, positive=True)

    for i in ("i", "j"):
        m.add_var(f"g_{i}", latent=True, label=rf"$g_{i}$")
        m.add_var(f"e_{i}", latent=True, label=rf"$e_{i}$")
        m.add_var(f"y_{i}", label=rf"$y_{i}$")
        m.add_path(f"g_{i}", f"y_{i}", 1)  # y = g + e
        m.add_path(f"e_{i}", f"y_{i}", 1)
        m.add_variance(f"g_{i}", "V_A")
        m.add_variance(f"e_{i}", "V_E")

    # Level 2: Cov[g_i, g_j | pi_ij] = V_A * pi_ij. 'pi' would be 3.14159... under plain
    # sympify; the registry keeps it a symbol.
    m.add_cov("g_i", "g_j", "V_A * pi_ij")

    # GE-indep, No-IGE, Env-indep: every g-e covariance is absent, within and between
    # individuals. Absence of an edge IS the assumption -- but record it so the diagram and
    # a later reader can see it was a choice.
    m.assume("V_A + V_E", 1)  # V_P = 1 in the base population, so h2 = V_A
    return m


def test_relative_covariance_section1_roundtrips():
    m = relative_covariance_section1()

    assert m.observed == ("y_i", "y_j")
    assert set(m.latent) == {"g_i", "e_i", "g_j", "e_j"}
    assert set(m.exogenous) == {"g_i", "e_i", "g_j", "e_j"}
    assert m.endogenous == ("y_i", "y_j")
    assert m.is_recursive
    assert m.validate() == []

    V_A, V_E, pi_ij = (m.sym(s) for s in ("V_A", "V_E", "pi_ij"))
    A, S, F, order = m.ram()
    assert order == ("g_i", "e_i", "y_i", "g_j", "e_j", "y_j")

    # y = g + e, both coefficients exactly 1, and no cross-individual directed paths
    expected_A = sp.zeros(6, 6)
    expected_A[2, 0] = expected_A[2, 1] = 1  # y_i <- g_i, e_i
    expected_A[5, 3] = expected_A[5, 4] = 1  # y_j <- g_j, e_j
    assert A == expected_A

    # the only off-block entry is the genetic covariance; every g-e term is zero
    expected_S = sp.diag(V_A, V_E, 0, V_A, V_E, 0)
    expected_S[0, 3] = expected_S[3, 0] = V_A * pi_ij
    assert S == expected_S

    assert F == sp.Matrix([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]])

    # the model can say "standardized in the base population" without baking it in
    assert m.assumptions == (sp.Eq(V_A + V_E, 1),)
    assert m.units == pm.Units.unstandardized()


def test_relative_covariance_section1_implies_the_boxed_results():
    """Spike, not the engine: the API can express Section 1's boxed Level-2 result."""
    m = relative_covariance_section1()
    V_A, V_E, pi_ij = (m.sym(s) for s in ("V_A", "V_E", "pi_ij"))
    Sigma, _ = ram_sigma(m)

    # Var[y] = V_A + V_E  (= 1 under the recorded assumption, giving h2 = V_A)
    assert sp.simplify(Sigma[0, 0] - (V_A + V_E)) == 0
    assert sp.simplify(Sigma[0, 0].subs(m.substitutions()).subs(V_E, 1 - V_A) - 1) == 0

    # boxed eq:level2 -- E[y_i y_j | pi_ij] = V_A * pi_ij
    assert sp.simplify(Sigma[0, 1] - V_A * pi_ij) == 0


def test_latent_variables_are_queryable_targets_in_principle():
    """The point of tracking latents explicitly: g_i is a legitimate query target later."""
    m = relative_covariance_section1()
    assert "g_i" in m.names and m.var("g_i").latent
    # F drops it from the observed block, but the full RAM order retains it
    A, S, F, order = m.ram()
    assert "g_i" in order
    assert F.shape == (2, 6)
