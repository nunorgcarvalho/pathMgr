"""Proof of life: Section 1 of `relative_covariance.tex`, derived by the engine.

Task-20260804-151347 asks specifically for this. The claim of Section 1 is that under the
additive model with independent environments, the *phenotypic* covariance between two
individuals reduces to the *genetic* one:

    C_ij = E[(g_i + e_i)(g_j + e_j) | I] = E[g_i g_j | I]                    (eq:reduce)

and at Level 2, conditioning on realized relatedness,

    E[y_i y_j | pi_ij] = V_A pi_ij                                           (eq:level2)

Getting these out of the engine rather than by hand is the first real check that pathMgr
computes rather than merely records. The point is that nothing here tells the engine that the
environmental cross terms vanish -- that is a *consequence* of the assumptions being encoded
as the absence of the corresponding edges, and the engine has to produce it.

The model is the same one built for task-20260804-151346 §3; it is imported rather than
rebuilt so the two tasks cannot drift apart.
"""

import sympy as sp

import pathmgr as pm

from test_validation_models import relative_covariance_section1


def test_phenotypic_covariance_reduces_to_the_genetic_covariance():
    """eq:reduce -- every cross term involving e vanishes, and the engine finds that itself."""
    m = relative_covariance_section1()
    e = pm.RAMEngine(m)

    # the headline: Cov[y_i, y_j] equals Cov[g_i, g_j] exactly
    assert sp.simplify(e.cov("y_i", "y_j") - e.cov("g_i", "g_j")) == 0

    # and it does so because each environmental cross term is individually zero --
    # No-IGE (g_i vs e_j) and Env-indep (e_i vs e_j), encoded as absent edges
    assert e.cov("e_i", "e_j") == 0  # Env-indep
    assert e.cov("g_i", "e_j") == 0  # No-IGE
    assert e.cov("g_j", "e_i") == 0  # No-IGE, the other way
    assert e.cov("g_i", "e_i") == 0  # GE-indep, within an individual


def test_level_2_boxed_result():
    """eq:level2 -- E[y_i y_j | pi_ij] = V_A pi_ij."""
    m = relative_covariance_section1()
    e = pm.RAMEngine(m)
    V_A, pi_ij = m.sym("V_A"), m.sym("pi_ij")
    assert sp.simplify(e.cov("y_i", "y_j") - V_A * pi_ij) == 0


def test_variance_decomposition_and_heritability():
    """Var[y] = V_A + V_E, so h2 = V_A once the base population is standardized."""
    m = relative_covariance_section1()
    e = pm.RAMEngine(m)
    V_A, V_E = m.sym("V_A"), m.sym("V_E")

    assert sp.simplify(e.var("y_i") - (V_A + V_E)) == 0
    # The model records V_A + V_E = 1 as a side relation. That is not in `Symbol = expr`
    # form, so it is deliberately NOT applied by `apply_assumptions=True` -- it could be
    # solved for either symbol, and choosing one silently would be guessing. Naming the
    # symbol to eliminate makes it explicit, and then Var[y] = 1.
    assert sp.simplify(e.var("y_i", apply_assumptions=True) - (V_A + V_E)) == 0
    assert sp.simplify(e.var("y_i", apply_assumptions=["V_E"]) - 1) == 0
    # so h2 = V_A / Var[y] = V_A
    h2 = e.cov("g_i", "y_i") ** 2 / (e.var("g_i") * e.var("y_i"))  # Corr[g, y]^2
    assert sp.simplify(h2 - V_A / (V_A + V_E)) == 0
    assert sp.simplify(h2.subs(m.substitutions(solve_for=["V_E"])) - V_A) == 0


def test_relatives_correlation_is_pi_when_the_trait_is_fully_heritable():
    """Corr[y_i, y_j] = h2 * pi_ij -- and the engine normalizes by implied SDs, not by 1."""
    m = relative_covariance_section1()
    e = pm.RAMEngine(m)
    V_A, V_E, pi_ij = (m.sym(s) for s in ("V_A", "V_E", "pi_ij"))

    corr = e.corr("y_i", "y_j")
    assert sp.simplify(corr - V_A * pi_ij / (V_A + V_E)) == 0
    # with V_E -> 0 the phenotypic correlation is exactly the relatedness
    assert sp.simplify(corr.subs({V_E: 0}) - pi_ij) == 0


def test_latent_genetic_values_are_directly_queryable():
    """The user wants covariances between latent variables too, not just observed ones."""
    m = relative_covariance_section1()
    e = pm.RAMEngine(m)
    V_A, V_E, pi_ij = (m.sym(s) for s in ("V_A", "V_E", "pi_ij"))

    assert sp.simplify(e.cov("g_i", "g_j") - V_A * pi_ij) == 0
    assert sp.simplify(e.var("g_i") - V_A) == 0
    assert sp.simplify(e.corr("g_i", "g_j") - pi_ij) == 0  # genetic correlation IS pi
    # latent-to-observed: Cov[g_i, y_i] = V_A, so Corr[g_i, y_i] = h (the path coefficient)
    assert sp.simplify(e.cov("g_i", "y_i") - V_A) == 0
    assert sp.simplify(e.corr("g_i", "y_i") ** 2 - V_A / (V_A + V_E)) == 0


def test_report_states_its_units():
    """A returned covariance must never be scale-ambiguous."""
    m = relative_covariance_section1()
    report = pm.RAMEngine(m).explain("y_i", "y_j")
    assert report.units == pm.Units.unstandardized()
    assert "unstandardized" in str(report)
    assert sp.simplify(report.cov - m.sym("V_A") * m.sym("pi_ij")) == 0
