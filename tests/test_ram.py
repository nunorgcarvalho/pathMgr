"""Tests for the RAM engine: the required battery, the query API, robustness, and units.

The battery of task-20260804-151347 is the four models at the top -- plain regression, a
mediation chain (so indirect effects appear), a model with a latent variable, and a model
with a bidirected edge between exogenous variables. Each is checked against a covariance
worked out by hand, not merely against itself.

``test_recursion_matches_the_brute_force_inverse`` is the load-bearing one: the engine
computes Sigma for a recursive model by two topological sweeps and never forms
``(I - A)^-1``, so that shortcut is checked against the textbook expression on a spread of
randomly generated models with latents.
"""

import random

import pytest
import sympy as sp

import pathmgr as pm


# ======================================================================================
# the required battery
# ======================================================================================
def test_plain_regression():
    m = pm.from_text(
        """
        positive: V_1, V_2, V_r
        y ~ b1*x1 + b2*x2
        x1 ~~ V_1*x1
        x2 ~~ V_2*x2
        y  ~~ V_r*y
        """
    )
    e = pm.RAMEngine(m)
    b1, b2, V_1, V_2, V_r = (m.sym(s) for s in ("b1", "b2", "V_1", "V_2", "V_r"))

    assert sp.simplify(e.cov("x1", "y") - b1 * V_1) == 0
    assert sp.simplify(e.cov("x2", "y") - b2 * V_2) == 0
    assert sp.simplify(e.cov("x1", "x2")) == 0
    assert sp.simplify(e.var("y") - (b1**2 * V_1 + b2**2 * V_2 + V_r)) == 0
    assert not e.used_inverse


def test_mediation_chain_shows_the_indirect_effect():
    """x -> m -> y with a direct x -> y: Cov[x, y] must contain BOTH routes."""
    mod = pm.from_text(
        """
        positive: V_x, V_m, V_y
        m ~ a*x
        y ~ b*m + c*x
        x ~~ V_x*x
        m ~~ V_m*m
        y ~~ V_y*y
        """
    )
    e = pm.RAMEngine(mod)
    a, b, c, V_x, V_m, V_y = (mod.sym(s) for s in ("a", "b", "c", "V_x", "V_m", "V_y"))

    # indirect (a*b) + direct (c), each scaled by Var[x]
    assert sp.simplify(e.cov("x", "y") - V_x * (a * b + c)) == 0
    assert sp.simplify(e.cov("x", "m") - a * V_x) == 0
    # the mediator's covariance with y picks up its own disturbance through b
    assert sp.simplify(e.cov("m", "y") - (b * (a**2 * V_x + V_m) + c * a * V_x)) == 0
    assert sp.simplify(
        e.var("y") - (b**2 * (a**2 * V_x + V_m) + c**2 * V_x + 2 * a * b * c * V_x + V_y)
    ) == 0


def test_latent_common_factor():
    """One latent factor with two indicators -- the indicators covary only through it."""
    m = pm.from_text(
        """
        latent: f
        positive: V_f, V_1, V_2
        y1 ~ l1*f
        y2 ~ l2*f
        f  ~~ V_f*f
        y1 ~~ V_1*y1
        y2 ~~ V_2*y2
        """
    )
    e = pm.RAMEngine(m)
    l1, l2, V_f, V_1, V_2 = (m.sym(s) for s in ("l1", "l2", "V_f", "V_1", "V_2"))

    assert sp.simplify(e.cov("y1", "y2") - l1 * l2 * V_f) == 0
    assert sp.simplify(e.var("y1") - (l1**2 * V_f + V_1)) == 0
    # the latent is a first-class query target, not filtered away
    assert sp.simplify(e.cov("f", "y1") - l1 * V_f) == 0
    assert sp.simplify(e.var("f") - V_f) == 0


def test_bidirected_edge_between_exogenous_variables():
    """x1 <-> x2 must reach y through both paths -- the classic confounded-predictor case."""
    m = pm.from_text(
        """
        positive: V_1, V_2, V_r
        y ~ b1*x1 + b2*x2
        x1 ~~ V_1*x1
        x2 ~~ V_2*x2
        x1 ~~ c12*x2
        y  ~~ V_r*y
        """
    )
    e = pm.RAMEngine(m)
    b1, b2, V_1, V_2, V_r, c12 = (m.sym(s) for s in ("b1", "b2", "V_1", "V_2", "V_r", "c12"))

    assert sp.simplify(e.cov("x1", "x2") - c12) == 0
    assert sp.simplify(e.cov("x1", "y") - (b1 * V_1 + b2 * c12)) == 0
    assert sp.simplify(e.cov("x2", "y") - (b2 * V_2 + b1 * c12)) == 0
    assert sp.simplify(
        e.var("y") - (b1**2 * V_1 + b2**2 * V_2 + 2 * b1 * b2 * c12 + V_r)
    ) == 0
    # and the edge really is load-bearing: zeroing it removes exactly those terms
    assert sp.simplify(e.cov("x1", "y").subs({c12: 0}) - b1 * V_1) == 0


# ======================================================================================
# the shortcut is the real risk -- check it against the textbook expression
# ======================================================================================
def _random_recursive_model(rng: random.Random, n: int = 7) -> pm.Model:
    m = pm.Model("random")
    names = [f"v{i}" for i in range(n)]
    for i, name in enumerate(names):
        m.add_var(name, latent=(i % 3 == 0))
    for j in range(n):  # only i < j, so acyclic by construction
        for i in range(j):
            if rng.random() < 0.35:
                m.add_path(names[i], names[j], f"a{i}_{j}")
    for i, name in enumerate(names):
        m.add_variance(name, f"S{i}")
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.15:
                m.add_cov(names[i], names[j], f"C{i}_{j}")
    return m


@pytest.mark.parametrize("seed", range(12))
def test_recursion_matches_the_brute_force_inverse(seed):
    """Two topological sweeps must equal F (I-A)^-1 S (I-A)^-T, on models with latents."""
    m = _random_recursive_model(random.Random(seed))
    e = pm.RAMEngine(m)
    A, S, _F, _order = m.ram()

    brute = sp.expand((sp.eye(A.rows) - A).inv() * S * ((sp.eye(A.rows) - A).inv()).T)
    assert sp.simplify(e.sigma() - brute) == sp.zeros(A.rows, A.rows)
    assert not e.used_inverse  # the shortcut really was taken


@pytest.mark.parametrize("seed", range(6))
def test_sigma_is_symmetric(seed):
    m = _random_recursive_model(random.Random(100 + seed))
    sigma = pm.RAMEngine(m).sigma()
    assert sp.simplify(sigma - sigma.T) == sp.zeros(sigma.rows, sigma.rows)


# ======================================================================================
# the full matrix is primary; the observed filter is a view
# ======================================================================================
def test_full_sigma_covers_every_node_and_observed_is_a_view():
    m = pm.from_text(
        """
        latent: f
        positive: V_f, V_1
        y1 ~ l1*f
        f  ~~ V_f*f
        y1 ~~ V_1*y1
        """
    )
    e = pm.RAMEngine(m)
    # every node, latents included -- in model insertion order, which for a parsed model puts
    # directive-declared variables (`latent: f`) before those inferred from the equations
    assert e.order == ("f", "y1")
    assert e.sigma().shape == (2, 2)

    observed_sigma, observed_names = e.sigma_observed()
    assert observed_names == ("y1",)
    assert observed_sigma.shape == (1, 1)
    assert observed_sigma[0, 0] == e.var("y1")


@pytest.mark.parametrize(
    "x,y",
    [("f", "g"), ("f", "y1"), ("y1", "y2")],
    ids=["latent-latent", "latent-observed", "observed-observed"],
)
def test_queries_work_for_every_pairing(x, y):
    m = pm.from_text(
        """
        latent: f, g
        positive: V_f, V_g, V_1, V_2
        y1 ~ l1*f + l2*g
        y2 ~ l3*g
        f ~~ V_f*f
        g ~~ V_g*g
        f ~~ c_fg*g
        y1 ~~ V_1*y1
        y2 ~~ V_2*y2
        """
    )
    e = pm.RAMEngine(m)
    for expr in (e.cov(x, y), e.var(x), e.corr(x, y)):
        assert isinstance(expr, sp.Expr)
    assert sp.simplify(e.cov(x, y) - e.cov(y, x)) == 0
    assert sp.simplify(e.cov("f", "g") - m.sym("c_fg")) == 0


def test_unknown_variable_is_a_clear_error():
    e = pm.RAMEngine(pm.from_text("y ~ b*x\nx ~~ V*x"))
    with pytest.raises(KeyError, match="unknown variable 'nope'"):
        e.cov("nope", "y")


# ======================================================================================
# units: correlations never assume unit variance
# ======================================================================================
def test_corr_divides_by_model_implied_standard_deviations():
    m = pm.from_text("positive: V_x, V_r\ny ~ b*x\nx ~~ V_x*x\ny ~~ V_r*y")
    e = pm.RAMEngine(m)
    b, V_x, V_r = (m.sym(s) for s in ("b", "V_x", "V_r"))

    expected = b * V_x / sp.sqrt(V_x * (b**2 * V_x + V_r))
    assert sp.simplify(e.corr("x", "y") - expected) == 0
    # scaling x's variance changes cov but NOT the correlation -- unit variance is nowhere assumed
    doubled = e.corr("x", "y").subs({V_x: 4 * V_x})
    assert sp.simplify(doubled - expected.subs({V_x: 4 * V_x})) == 0


def test_corr_is_one_on_the_diagonal():
    m = pm.from_text("positive: V_x, V_r\ny ~ b*x\nx ~~ V_x*x\ny ~~ V_r*y")
    e = pm.RAMEngine(m)
    assert sp.simplify(e.corr("y", "y") - 1) == 0


def test_corr_refuses_a_zero_variance_variable():
    m = pm.Model()
    m.add_vars("x", "y")
    m.add_path("x", "y", "b")  # x has no variance at all
    e = pm.RAMEngine(m)
    with pytest.raises(ValueError, match="identically zero"):
        e.corr("x", "y")


def test_units_are_carried_and_reported():
    m = pm.from_text(
        "units: standardized to gen 0\npositive: V_r\ny ~ b*x\nx ~~ x\ny ~~ V_r*y"
    )
    e = pm.RAMEngine(m)
    assert e.units == pm.Units.standardized("gen 0")
    assert "standardized to gen 0" in str(e.explain("x", "y"))


def test_check_standardization_finds_variables_that_are_not_unit_variance():
    m = pm.from_text(
        """
        units: standardized to gen 0
        y ~ b*x
        x ~~ x
        y ~~ V_r*y
        """
    )
    e = pm.RAMEngine(m)
    # Var[x] = 1 by construction, but Var[y] = b^2 + V_r is not 1 for free
    assert e.check_standardization() == ["y"]
    # the standard SEM residual choice makes it so
    m2 = m.copy()
    m2.remove_cov("y", "y")
    m2.add_variance("y", "1 - b**2")
    assert pm.RAMEngine(m2).check_standardization() == []


def test_check_standardization_is_empty_for_unstandardized_models():
    m = pm.from_text("y ~ b*x\nx ~~ V_x*x\ny ~~ V_r*y")
    assert pm.RAMEngine(m).check_standardization() == []


# ======================================================================================
# robustness: cycles, singularity, caching, forms
# ======================================================================================
def test_cyclic_model_uses_the_inverse_and_sums_the_feedback_loop():
    """A capability the chain tracer cannot match: infinitely many chains, summed in closed form."""
    m = pm.Model("feedback")
    m.add_vars("x", "y", "z")
    m.add_path("x", "y", "a")
    m.add_path("y", "z", "b")
    m.add_path("z", "y", "d")  # y <-> z feedback loop
    for v in ("x", "y", "z"):
        m.add_variance(v, f"S_{v}")
    e = pm.RAMEngine(m)
    a, b, d, S_x = (m.sym(s) for s in ("a", "b", "d", "S_x"))

    assert not m.is_recursive
    assert e.used_inverse
    # geometric series in the loop gain b*d
    assert sp.simplify(e.cov("x", "y") - a * S_x / (1 - b * d)) == 0
    assert sp.simplify(e.sigma() - e.sigma().T) == sp.zeros(3, 3)


def test_topological_order_refuses_a_cyclic_model():
    m = pm.Model()
    m.add_vars("x", "y")
    m.add_path("x", "y", "a")
    m.add_path("y", "x", "b")
    with pytest.raises(pm.CyclicModelError, match="no topological order"):
        pm.RAMEngine(m).topological_order()


def test_singular_model_is_rejected_with_a_clear_error():
    """A feedback loop of unit total gain: (I - A) is singular, so there is no Sigma."""
    m = pm.Model("unit loop")
    m.add_vars("x", "y")
    m.add_path("x", "y", 1)
    m.add_path("y", "x", 1)
    m.add_variance("x", "S_x")
    with pytest.raises(pm.SingularModelError, match="singular"):
        pm.RAMEngine(m).sigma()


def test_sigma_is_cached_and_invalidated_by_a_model_change():
    m = pm.from_text("positive: V_x\ny ~ b*x\nx ~~ V_x*x")
    e = pm.RAMEngine(m)
    first = e.sigma()
    assert e.sigma() is first  # cached: same object, no recompute

    m.add_var("z")
    m.add_path("y", "z", "c")
    second = e.sigma()
    assert second is not first
    assert second.shape == (3, 3)
    assert sp.simplify(e.cov("x", "z") - m.sym("b") * m.sym("c") * m.sym("V_x")) == 0


@pytest.mark.parametrize("form", ["raw", "expanded", "simplified", "factored"])
def test_all_forms_agree_mathematically(form):
    m = pm.from_text(
        """
        positive: V_1, V_2, V_r
        y ~ b1*x1 + b2*x2
        x1 ~~ V_1*x1
        x2 ~~ V_2*x2
        x1 ~~ c12*x2
        y ~~ V_r*y
        """
    )
    e = pm.RAMEngine(m)
    reference = e.var("y", form="raw")
    assert sp.simplify(e.var("y", form=form) - reference) == 0


def test_expanded_is_the_default_for_cov_and_is_actually_expanded():
    m = pm.from_text("positive: V_1\ny ~ b1*x1\nx1 ~~ V_1*x1\ny ~~ V_r*y")
    e = pm.RAMEngine(m)
    var_y = e.var("y")
    assert var_y == sp.expand(var_y)


def test_bad_form_is_rejected():
    e = pm.RAMEngine(pm.from_text("y ~ b*x\nx ~~ V*x"))
    with pytest.raises(ValueError, match="form must be one of"):
        e.cov("x", "y", form="pretty")


def test_assumptions_are_opt_in_and_resolved_to_a_fixed_point():
    m = pm.from_text(
        """
        latent: g, e_
        positive: V_A, V_E
        y ~ g + e_
        g ~~ V_A*g
        e_ ~~ V_E*e_
        assume: V_E = 1 - V_A
        """
    )
    eng = pm.RAMEngine(m)
    assert sp.simplify(eng.var("y") - (m.sym("V_A") + m.sym("V_E"))) == 0  # untouched
    assert sp.simplify(eng.var("y", apply_assumptions=True) - 1) == 0

    # chained relations resolve, not just one level
    m2 = pm.from_text(
        """
        y ~ g
        g ~~ V_A*g
        assume: V_A = 2*V_B
        assume: V_B = 3*V_C
        """
    )
    got = pm.RAMEngine(m2).var("y", apply_assumptions=True)
    assert sp.simplify(got - 6 * m2.sym("V_C")) == 0


def test_inverse_IA_is_the_total_effects_matrix():
    m = pm.from_text("m ~ a*x\ny ~ b*m\nx ~~ V_x*x")
    e = pm.RAMEngine(m)
    B = e.inverse_IA()
    i = {n: k for k, n in enumerate(e.order)}
    # total effect of x on y is a*b, via the mediator
    assert sp.simplify(B[i["y"], i["x"]] - m.sym("a") * m.sym("b")) == 0
    assert sp.simplify(B[i["m"], i["x"]] - m.sym("a")) == 0


def test_empty_model_does_not_crash():
    e = pm.RAMEngine(pm.Model("empty"))
    assert e.sigma().shape == (0, 0)
    assert e.order == ()
