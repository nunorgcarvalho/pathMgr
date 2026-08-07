"""The pedigree unroller (task-20260804-151350).

Two things here are easy to get wrong quietly, and both are pinned deliberately.

**Generation indexing.** The relative-pair formulas index to the **parents'** generation. Using the
offspring's ``V_A`` instead is wrong by ~4e-4 near equilibrium -- small enough to read as numerical
noise -- but by 1.6e-2 at the first generation. So every indexing test runs at **low t**, where the
error is largest; a test near equilibrium would pass either way and prove nothing.

**``((1+rho_g)/2)^d`` is equilibrium-only.** On a finite unroll from a randomly mating base, a
lineal pair spanning several generations is a **chained product using each generation's own**
``rho_g``, not a power of a single one. A mismatch must not be reconciled by adjusting the formula.
"""

import pytest
import sympy as sp

import pathmgr as pm
from pathmgr.genetics import AMParameters, am_pedigree, g_level_model

# V_A(0) = 0.4, V_E = 0.6 -- the coordinator's oracle parameters, so the numbers here can be
# compared directly against their independent 8-generation numeric unroll.
BASE = {"V_A0": sp.Rational(4, 10), "V_E": sp.Rational(6, 10), "rho_y": sp.Rational(3, 10)}


def numbers(unrolled):
    return {unrolled.V_A0: BASE["V_A0"], unrolled.V_E: BASE["V_E"], unrolled.rho_y: BASE["rho_y"]}


def evaluate(unrolled, expression):
    """A per-generation-symbol expression as a float, resolving the recursion to V_A(0)."""
    resolved = expression.subs(unrolled.recursion_substitutions()).subs(numbers(unrolled))
    return float(sp.N(resolved))


# ======================================================================================
# scaffolding
# ======================================================================================
def test_pedigree_is_structure_only():
    """The scaffolding must know nothing about genetics -- that is the layer boundary.

    Checked against the CODE of `Pedigree` and `am_pedigree`, not the module text: the module
    docstring legitimately explains the genetics the builders below it implement.
    """
    import inspect

    from pathmgr.genetics.pedigree import Pedigree, am_pedigree

    for obj in (Pedigree, am_pedigree):
        source = inspect.getsource(obj)
        for genetics_word in ("V_A", "V_E", "rho_g", "rho_y", "copath", "beta", "add_path"):
            assert genetics_word not in source, f"{obj.__name__} mentions {genetics_word}"
    # and the scaffolding builds no model at all
    assert "Model(" not in inspect.getsource(Pedigree)


def test_pedigree_shape():
    pedigree = am_pedigree(2, children_per_couple=2, breeding_children=1)
    assert pedigree.n_generations == 3
    assert len(pedigree.generation(0)) == 2  # the founding couple
    assert len(pedigree.couples) == 2
    children = pedigree.children_of(pedigree.couples[0])
    assert len(children) == 2


def test_relationships_are_derived_from_structure():
    """Degree alone is not sufficient, so the distinctions come from structural tests."""
    pedigree = am_pedigree(2, children_per_couple=2, breeding_children=1)
    maternal, paternal = pedigree.couples[0].maternal, pedigree.couples[0].paternal
    sib_a, sib_b = sorted(pedigree.children_of(pedigree.couples[0]))
    grandchild = sorted(pedigree.children_of(pedigree.couples[1]))[0]

    assert pedigree.relationship(maternal, paternal) == "partners"
    assert pedigree.relationship(maternal, sib_a) == "lineal"
    assert pedigree.relationship(sib_a, sib_b) == "full siblings"
    assert pedigree.relationship(maternal, grandchild) == "lineal"       # degree 2
    assert pedigree.relationship(sib_b, grandchild) == "collateral"    # ALSO degree 2, different
    assert pedigree.relationship(sib_a, sib_a) == "self"


def test_half_siblings_are_recognised_as_a_third_case():
    pedigree = am_pedigree(1, children_per_couple=2, half_sib_at=0)
    full = sorted(pedigree.children_of(pedigree.couples[0]))
    half = sorted(pedigree.children_of(pedigree.couples[1]))[0]
    assert pedigree.relationship(full[0], full[1]) == "full siblings"
    assert pedigree.relationship(full[0], half) == "half siblings"
    # the two outer parents share a partner but no ancestor at all
    outer_a = pedigree.couples[0].paternal
    outer_b = pedigree.couples[1].paternal
    assert pedigree.relationship(outer_a, outer_b) == "co-parents-in-law"
    assert not (pedigree.ancestors_of(outer_a) & pedigree.ancestors_of(outer_b))


def test_layout_puts_generations_in_rows():
    pedigree = am_pedigree(2)
    layout = pedigree.layout()
    rows = {}
    for key, (_, y) in layout.positions.items():
        rows.setdefault(pedigree.individuals[key].generation, set()).add(y)
    assert all(len(ys) == 1 for ys in rows.values()), "one row per generation"
    assert rows[0].pop() > rows[1].pop() > rows[2].pop(), "later generations lower"


# ======================================================================================
# the model
# ======================================================================================
def test_assortment_is_one_copath_per_couple_and_nothing_else():
    pedigree = am_pedigree(2)
    unrolled = g_level_model(pedigree)
    model = unrolled.model

    assert len(model.copaths) == len(pedigree.couples)
    assert len(model.mating_processes) == len(pedigree.couples)
    # no hand-written induced covariance anywhere
    assert [e for e in model.bidirected_edges if not e.is_variance] == []
    assert [i for i in model.validate() if i.severity == "error"] == []


def test_the_copath_coefficient_is_generation_indexed():
    """mu_t = rho_y / V_P(t), recomputed per generation because V_P grows."""
    pedigree = am_pedigree(2)
    unrolled = g_level_model(pedigree)
    assert unrolled.mu[0] != unrolled.mu[1], "a fixed mu would be a different model"
    for t, mu in enumerate(unrolled.mu):
        assert sp.simplify(mu - unrolled.rho_y / unrolled.V_P[t]) == 0
    for couple in pedigree.couples:
        value = unrolled.model.copath_value(
            f"y_{couple.maternal}", f"y_{couple.paternal}", process=couple.key
        )
        assert sp.simplify(value - unrolled.mu[couple.generation]) == 0


def test_holding_mu_constant_is_available_and_is_a_different_model():
    pedigree = am_pedigree(2)
    varying = g_level_model(pedigree)
    fixed = g_level_model(pedigree, AMParameters(hold="mu"))
    assert fixed.mu[0] == fixed.mu[1]
    assert varying.mu[0] != varying.mu[1]
    with pytest.raises(ValueError, match="hold must be"):
        AMParameters(hold="something else")


def test_segregation_variance_is_an_explicit_constant():
    """V_K = V_A(0)/2 held constant -- the choice is a parameter, not an assumption."""
    pedigree = am_pedigree(1)
    unrolled = g_level_model(pedigree)
    assert sp.simplify(unrolled.V_K - unrolled.V_A0 / 2) == 0
    child = sorted(pedigree.children_of(pedigree.couples[0]))[0]
    assert unrolled.model.cov_value(f"s_{child}", f"s_{child}") == unrolled.V_K

    custom = g_level_model(pedigree, AMParameters(segregation_variance=sp.Symbol("V_K_custom")))
    # `V_K` comes back as the symbol the MODEL registered, not the one handed in: a caller's
    # symbol carries its own sympy assumptions and would not compare equal, so substituting
    # against it would silently do nothing.
    assert custom.V_K == custom.model.sym("V_K_custom")
    assert custom.V_K.name == "V_K_custom"
    child = sorted(pedigree.children_of(pedigree.couples[0]))[0]
    assert custom.model.cov_value(f"s_{child}", f"s_{child}") == custom.V_K
    # and it is usable as a substitution key, which is the point
    assert pm.RAMEngine(custom.model).var(f"g_{child}").subs({custom.V_K: 0}).has(
        custom.V_K
    ) is False


# ======================================================================================
# hand-checkable covariances -- all at the PARENTS' generation index
# ======================================================================================
@pytest.fixture
def one_generation():
    pedigree = am_pedigree(1, children_per_couple=2)
    unrolled = g_level_model(pedigree)
    return pedigree, unrolled, pm.RAMEngine(unrolled.model)


def test_partners(one_generation):
    pedigree, unrolled, engine = one_generation
    maternal, paternal = pedigree.couples[0].maternal, pedigree.couples[0].paternal
    assert sp.simplify(
        engine.cov(f"y_{maternal}", f"y_{paternal}") - unrolled.rho_y * unrolled.V_P[0]
    ) == 0
    # rho_g is DERIVED, never asserted
    assert sp.simplify(
        engine.cov(f"g_{maternal}", f"g_{paternal}") - unrolled.rho_g[0] * unrolled.V_A[0]
    ) == 0


def test_parent_offspring_and_full_siblings(one_generation):
    pedigree, unrolled, engine = one_generation
    maternal = pedigree.couples[0].maternal
    sib_a, sib_b = sorted(pedigree.children_of(pedigree.couples[0]))
    V_A, rho_g, rho_y = unrolled.V_A[0], unrolled.rho_g[0], unrolled.rho_y

    assert sp.simplify(
        engine.cov(f"g_{maternal}", f"g_{sib_a}") - V_A * (1 + rho_g) / 2
    ) == 0
    assert sp.simplify(engine.cov(f"g_{sib_a}", f"g_{sib_b}") - V_A * (1 + rho_g) / 2) == 0
    assert sp.simplify(
        engine.cov(f"y_{maternal}", f"y_{sib_a}") - V_A * (1 + rho_y) / 2
    ) == 0

    # parent-offspring exceeds full-sib, and only because of the environmental cross term
    excess = sp.simplify(
        engine.cov(f"y_{maternal}", f"y_{sib_a}") - engine.cov(f"y_{sib_a}", f"y_{sib_b}")
    )
    assert sp.simplify(excess - V_A * (rho_y - rho_g) / 2) == 0


def test_the_recursion_is_derived_not_assumed(one_generation):
    pedigree, unrolled, engine = one_generation
    child = sorted(pedigree.children_of(pedigree.couples[0]))[0]
    implied = engine.var(f"g_{child}")
    assert sp.simplify(implied - unrolled.recursion(0).rhs) == 0
    assert sp.simplify(
        implied - (unrolled.V_K + unrolled.V_A[0] * (1 + unrolled.rho_g[0]) / 2)
    ) == 0


def test_generation_indexing_is_pinned_at_low_t(one_generation):
    """THE indexing test. At low t the error is largest; near equilibrium it would vanish."""
    pedigree, unrolled, engine = one_generation
    maternal = pedigree.couples[0].maternal
    child = sorted(pedigree.children_of(pedigree.couples[0]))[0]
    rho_y = unrolled.rho_y

    got = engine.cov(f"y_{maternal}", f"y_{child}")
    parents_index = unrolled.V_A[0] * (1 + rho_y) / 2       # correct
    offspring_index = unrolled.V_A[1] * (1 + rho_y) / 2      # the trap

    assert sp.simplify(got - parents_index) == 0
    assert sp.simplify(got - offspring_index) != 0
    # and the gap is large at t = 0 -> 1, which is why the test lives here
    gap = abs(evaluate(unrolled, parents_index) - evaluate(unrolled, offspring_index))
    assert gap > 1e-2, f"expected a visible gap at low t, got {gap:.2e}"


def test_lineal_over_two_generations_is_a_chained_product_not_a_power():
    """``((1+rho_g)/2)^d`` is equilibrium-only. On a finite unroll it is a chained product."""
    pedigree = am_pedigree(2, children_per_couple=2, breeding_children=1)
    unrolled = g_level_model(pedigree)
    engine = pm.RAMEngine(unrolled.model)
    grandparent = pedigree.couples[0].maternal
    grandchild = sorted(pedigree.children_of(pedigree.couples[1]))[0]
    assert pedigree.relationship(grandparent, grandchild) == "lineal"

    got = engine.cov(f"g_{grandparent}", f"g_{grandchild}")
    chained = unrolled.V_A[0] * (1 + unrolled.rho_g[0]) / 2 * (1 + unrolled.rho_g[1]) / 2
    power = unrolled.V_A[0] * ((1 + unrolled.rho_g[0]) / 2) ** 2

    assert sp.simplify(got - chained) == 0, "each generation contributes its OWN rho_g"
    assert sp.simplify(got - power) != 0, "a single power would be the equilibrium form"
    # the difference is ~4e-4: small enough to be mistaken for numerical noise, hence this test
    gap = abs(evaluate(unrolled, chained) - evaluate(unrolled, power))
    assert 1e-5 < gap < 1e-2, f"{gap:.2e}"


def test_half_siblings_follow_neither_formula():
    """The sharpest check that a chain can cross co-paths from two different couples."""
    pedigree = am_pedigree(1, children_per_couple=2, half_sib_at=0)
    unrolled = g_level_model(pedigree)
    engine = pm.RAMEngine(unrolled.model)
    V_A, V_P, V_E, rho_g = unrolled.V_A[0], unrolled.V_P[0], unrolled.V_E, unrolled.rho_g[0]

    full = sorted(pedigree.children_of(pedigree.couples[0]))
    half = sorted(pedigree.children_of(pedigree.couples[1]))[0]

    got = engine.cov(f"g_{full[0]}", f"g_{half}")
    assert sp.simplify(got - (V_A * (1 + 2 * rho_g) + rho_g**2 * V_P) / 4) == 0
    # exceeds the collateral degree-2 form by exactly rho_g^2 V_E / 4
    collateral = V_A * (1 + rho_g) ** 2 / 4
    assert sp.simplify(got - collateral - rho_g**2 * V_E / 4) == 0


def test_individuals_with_no_common_ancestor_can_be_correlated():
    """Two people who each had children with the same third person. Zero under random mating."""
    pedigree = am_pedigree(1, children_per_couple=2, half_sib_at=0)
    unrolled = g_level_model(pedigree)
    engine = pm.RAMEngine(unrolled.model)
    outer_a, outer_b = pedigree.couples[0].paternal, pedigree.couples[1].paternal
    assert not (pedigree.ancestors_of(outer_a) & pedigree.ancestors_of(outer_b))

    got = engine.cov(f"g_{outer_a}", f"g_{outer_b}")
    assert sp.simplify(got - unrolled.rho_g[0] ** 2 * unrolled.V_P[0]) == 0
    assert sp.simplify(got.subs({unrolled.rho_y: 0})) == 0


# ======================================================================================
# regression against the random-mating results of Section 1
# ======================================================================================
def test_rho_y_zero_reduces_to_the_random_mating_results():
    pedigree = am_pedigree(2, children_per_couple=2, breeding_children=1)
    unrolled = g_level_model(pedigree)
    engine = pm.RAMEngine(unrolled.model)
    zero = {unrolled.rho_y: 0}
    # under random mating every generation has the base additive variance
    flat = {v: unrolled.V_A0 for v in unrolled.V_A}

    maternal = pedigree.couples[0].maternal
    sib_a, sib_b = sorted(pedigree.children_of(pedigree.couples[0]))
    grandchild = sorted(pedigree.children_of(pedigree.couples[1]))[0]
    V_A0 = unrolled.V_A0

    def value(expression):
        return sp.simplify(expression.subs(zero).subs(flat))

    assert value(engine.cov(f"g_{maternal}", f"g_{pedigree.couples[0].paternal}")) == 0  # partners
    assert value(engine.var(f"g_{sib_a}")) == V_A0                                    # no inflation
    assert value(engine.cov(f"g_{maternal}", f"g_{sib_a}") - V_A0 / 2) == 0             # PO
    assert value(engine.cov(f"g_{sib_a}", f"g_{sib_b}") - V_A0 / 2) == 0              # FS
    assert value(engine.cov(f"g_{maternal}", f"g_{grandchild}") - V_A0 / 4) == 0        # 2^-d
    # PO and FS are EQUAL under random mating -- the asymmetry is assortment's doing
    assert value(
        engine.cov(f"y_{maternal}", f"y_{sib_a}") - engine.cov(f"y_{sib_a}", f"y_{sib_b}")
    ) == 0


def test_both_engines_agree_on_an_unrolled_pedigree():
    pedigree = am_pedigree(2, children_per_couple=2, breeding_children=1)
    unrolled = g_level_model(pedigree)
    engine = pm.RAMEngine(unrolled.model)
    tracer = pm.WrightTracer(unrolled.model, max_chains=500_000)
    maternal = pedigree.couples[0].maternal
    grandchild = sorted(pedigree.children_of(pedigree.couples[1]))[0]
    for x, y in [
        (f"g_{maternal}", f"g_{grandchild}"),
        (f"y_{maternal}", f"y_{grandchild}"),
        (f"g_{maternal}", f"g_{pedigree.couples[0].paternal}"),
    ]:
        assert sp.simplify(tracer.cov(x, y) - engine.cov(x, y)) == 0, f"Cov[{x}, {y}]"


# ======================================================================================
# rendering
# ======================================================================================
def test_the_unrolled_pedigree_renders():
    from pathmgr.render import DiagramStyle, to_tikz

    pedigree = am_pedigree(2, children_per_couple=2, breeding_children=1)
    unrolled = g_level_model(pedigree)
    tex = to_tikz(unrolled.model, layout=unrolled.layout(), style=DiagramStyle(show_variances=False))
    assert tex.rstrip().endswith("\\end{tikzpicture}")
    assert tex.count("pmCopath,") == len(pedigree.couples)
    for key in pedigree.individuals:
        assert f"(y_{key})" in tex


# ======================================================================================
# numeric mode -- what makes deep unrolls tractable
# ======================================================================================
def test_numeric_mode_reproduces_the_analytic_trajectory():
    """V_A(t) must match the recursion, and the coordinator's independent numeric unroll."""
    unrolled = g_level_model(
        am_pedigree(8), AMParameters(values={"V_A0": 0.4, "V_E": 0.6, "rho_y": 0.3})
    )
    trajectory = [round(float(v), 5) for v in unrolled.V_A]
    assert trajectory == [0.4, 0.424, 0.43833, 0.44692, 0.45208, 0.45518, 0.45704, 0.45816, 0.45884]
    # still short of the analytic equilibrium 0.45986 at t=8 -- six to ten generations, as reported
    assert trajectory[-1] < 0.45986
    assert abs(trajectory[6] - 0.45986) / 0.45986 < 0.01, "within 1% by generation 6"
    assert round(float(unrolled.rho_g[-1]), 4) == 0.13


def test_numeric_mode_agrees_with_the_symbolic_model():
    """Same model, two routes to the same number."""
    values = {"V_A0": 0.4, "V_E": 0.6, "rho_y": 0.3}
    pedigree = am_pedigree(3)
    symbolic = g_level_model(pedigree)
    numeric = g_level_model(pedigree, AMParameters(values=values))

    deep = sorted(pedigree.generation(3))[0]
    from_symbols = symbolic.cov_value = pm.RAMEngine(symbolic.model).cov("g_i0_0", f"g_{deep}")
    resolved = from_symbols.subs(symbolic.recursion_substitutions()).subs(
        {symbolic.V_A0: sp.Rational(4, 10), symbolic.V_E: sp.Rational(6, 10),
         symbolic.rho_y: sp.Rational(3, 10)}
    )
    from_numbers = pm.RAMEngine(numeric.model).cov("g_i0_0", f"g_{deep}")
    assert abs(float(sp.N(resolved)) - float(from_numbers)) < 1e-12


def test_numeric_mode_needs_no_recursion_substitutions():
    unrolled = g_level_model(am_pedigree(2), AMParameters(values={"V_A0": 0.4, "V_E": 0.6, "rho_y": 0.3}))
    assert unrolled.recursion_substitutions() == {}
    assert all(v.is_number for v in unrolled.V_A)
    assert all(m.is_number for m in unrolled.mu)


def test_cov_does_not_materialise_sigma_when_copaths_are_present():
    """The targeted entry path is what makes a deep pedigree usable at all."""
    unrolled = g_level_model(am_pedigree(2))
    engine = pm.RAMEngine(unrolled.model)
    engine.cov("g_i0_0", "g_i1_0")
    assert engine._sigma_full is None, "cov() must not have built the full Sigma"
    # and it agrees with the full-matrix route when that IS built
    entry = engine.cov("g_i0_0", "g_i1_0")
    from_matrix = engine.sigma()[engine.order.index("g_i0_0"), engine.order.index("g_i1_0")]
    assert sp.simplify(entry - from_matrix) == 0


def test_layout_places_partners_adjacent():
    """Otherwise a sibling sits between a couple and the co-path is drawn through them."""
    pedigree = am_pedigree(2, children_per_couple=2, breeding_children=1)
    for couple in pedigree.couples:
        order = pedigree.generation_order(couple.generation)
        gap = abs(order.index(couple.maternal) - order.index(couple.paternal))
        assert gap == 1, f"{couple.maternal} and {couple.paternal} are {gap} apart"
    # and it stays deterministic
    assert pedigree.generation_order(1) == am_pedigree(
        2, children_per_couple=2, breeding_children=1
    ).generation_order(1)


# ======================================================================================
# compaction: results in the notation a reader expects (task-20260805-161349)
# ======================================================================================
def _compaction_cases():
    """A spread of queries over an unrolled pedigree, with and without the unit assumption."""
    pedigree = am_pedigree(2, children_per_couple=2, breeding_children=1)
    unrolled = g_level_model(pedigree)
    engine = pm.RAMEngine(unrolled.model)
    founder = pedigree.couples[0].maternal
    children = sorted(pedigree.children_of(pedigree.couples[0]))
    grandchild = sorted(pedigree.children_of(pedigree.couples[1]))[0]
    pairs = [
        (f"y_{founder}", f"y_{children[0]}"),
        (f"y_{children[0]}", f"y_{children[1]}"),
        (f"y_{founder}", f"y_{grandchild}"),
        (f"y_{children[1]}", f"y_{grandchild}"),
        (f"g_{founder}", f"g_{grandchild}"),
        (f"g_{children[0]}", f"g_{children[1]}"),
    ]
    return unrolled, engine, pairs


def test_compaction_never_changes_the_value():
    """THE test for this feature. Compaction may decline to shorten; it may not alter."""
    unrolled, engine, pairs = _compaction_cases()
    definitions = unrolled.compact_definitions()
    for x, y in pairs:
        raw = engine.cov(x, y)
        compacted = unrolled.compact(raw)
        assert sp.simplify(compacted.subs(definitions) - raw) == 0, (x, y, compacted)


def test_compaction_actually_shortens_the_results_it_is_for():
    """Guards against a change that quietly turns compaction into a no-op."""
    unrolled, engine, pairs = _compaction_cases()
    shortened = 0
    for x, y in pairs:
        raw = engine.cov(x, y)
        if sp.count_ops(unrolled.compact(raw)) < sp.count_ops(sp.factor(raw)):
            shortened += 1
    assert shortened >= 4, f"only {shortened} of {len(pairs)} got shorter"


def test_compaction_produces_the_textbook_forms():
    """The point of the notation: these are the expressions the writeup states."""
    pedigree = am_pedigree(2, children_per_couple=2, breeding_children=1)
    unrolled = g_level_model(pedigree)
    engine = pm.RAMEngine(unrolled.model)
    children = sorted(pedigree.children_of(pedigree.couples[0]))
    grandchild = sorted(pedigree.children_of(pedigree.couples[1]))[0]
    V_A0 = unrolled.V_A0
    rho_g_0, rho_g_1 = unrolled.rho_g_symbols[0], unrolled.rho_g_symbols[1]

    sibs = unrolled.compact(engine.cov(f"y_{children[0]}", f"y_{children[1]}"))
    assert sp.simplify(sibs - V_A0 * (1 + rho_g_0) / 2) == 0, sibs

    avuncular = unrolled.compact(engine.cov(f"y_{children[1]}", f"y_{grandchild}"))
    assert sp.simplify(avuncular - V_A0 * (1 + rho_g_0) * (1 + rho_g_1) / 4) == 0, avuncular


def test_unit_phenotypic_variance_collapses_only_when_the_model_says_so():
    """Point 5, and the guard on it: an unstated unit variance is never assumed."""
    def build(assert_unit):
        pedigree = am_pedigree(1, children_per_couple=2, half_sib_at=0)
        unrolled = g_level_model(pedigree)
        if assert_unit:
            unrolled.model.assume(unrolled.V_A[0] + unrolled.V_E, 1)
        return unrolled, pm.RAMEngine(unrolled.model), pedigree

    # the no-common-ancestor covariance is rho_g^2 V_P, so V_P is visible in it
    general, engine, pedigree = build(False)
    a, b = pedigree.couples[0].paternal, pedigree.couples[1].paternal
    raw = engine.cov(f"g_{a}", f"g_{b}")
    assert general.V_P_symbols[0] in general.compact(raw).free_symbols, (
        "V_P must survive when the model never claimed it was 1"
    )

    stated, engine_unit, pedigree_unit = build(True)
    a2, b2 = pedigree_unit.couples[0].paternal, pedigree_unit.couples[1].paternal
    compacted = stated.compact(engine_unit.cov(f"g_{a2}", f"g_{b2}"))
    assert stated.V_P_symbols[0] not in compacted.free_symbols, compacted
    assert stated._unit_phenotypic_generations() == (0,)
    assert "V_P(0) = 1 by assumption" in stated.explain_compaction()


def test_compaction_does_not_change_what_cov_returns():
    """A display step must stay a display step."""
    unrolled, engine, pairs = _compaction_cases()
    before = [engine.cov(x, y) for x, y in pairs]
    for x, y in pairs:
        unrolled.compact(engine.cov(x, y))
    after = [engine.cov(x, y) for x, y in pairs]
    assert before == after


def test_asking_for_display_symbols_does_not_invalidate_the_engine_cache():
    """`declare` must not bump model.revision, or compaction would cost a full Sigma rebuild."""
    unrolled, engine, _ = _compaction_cases()
    engine.sigma_copath_free()
    revision = unrolled.model.revision
    unrolled.V_P_symbols, unrolled.rho_g_symbols, unrolled.compact_definitions()
    assert unrolled.model.revision == revision


def test_the_equilibrium_can_be_reported_compactly():
    from pathmgr.genetics import equilibrium

    eq = equilibrium()
    compact = eq.compact()
    V_A0, V_E = eq.symbols["V_A0"], eq.symbols["V_E"]
    V_P0 = compact.symbols["V_P0"]

    assert V_P0 in compact.rho_g.free_symbols, compact.rho_g
    assert sp.count_ops(compact.rho_g) < sp.count_ops(eq.rho_g)
    # and it is the same fixed point
    assert sp.simplify(compact.rho_g.subs(V_P0, V_A0 + V_E) - eq.rho_g) == 0
    # the numbers are untouched
    assert abs(compact.evaluate({"V_A0": 0.4, "V_E": 0.6, "rho_y": 0.3})["rho_g"]
               - eq.evaluate({"V_A0": 0.4, "V_E": 0.6, "rho_y": 0.3})["rho_g"]) < 1e-12
