"""The allele-level transmission motif (task-20260804-173344).

Every expected value is the **general form**, carrying the ``1/V_P`` that the task's first draft
omitted, and the numeric parameters below are chosen so that **V_P != 1**. Both of the coordinator's
errors on this table were invisible at ``V_P = 1``, and ``V_A = 0.4, V_E = 0.6`` or
``V_A = 0.5, V_E = 0.5`` both land there. Using ``V_E = 0.7`` with ``V_A = 0.5`` is the cheapest
possible guard against that entire class of error, so it is what these tests use.

The headline property: the only cross-couple statement in the model is **one co-path between the
founders' phenotypes**. Every allele covariance, the offspring's linkage disequilibrium, and the
departure from Hardy-Weinberg are *derived*, never specified.
"""

import pytest
import sympy as sp

import pathmgr as pm
from pathmgr.genetics import AlleleMotif, allele_motif

# V_A = 0.09 + 0.16 + 0.25 = 0.5, V_E = 0.7, so V_P = 1.2 -- deliberately NOT 1
BETAS = (sp.Rational(3, 10), sp.Rational(4, 10), sp.Rational(5, 10))
V_E_VALUE = sp.Rational(7, 10)
RHO_Y_VALUE = sp.Rational(3, 10)


def numbers(motif: AlleleMotif) -> dict:
    values = {b: v for b, v in zip(motif.betas, BETAS)}
    values[motif.V_E] = V_E_VALUE
    values[motif.rho_y] = RHO_Y_VALUE
    return values


def test_the_oracle_parameters_do_not_have_unit_phenotypic_variance():
    """A guard on the guard: at V_P = 1 seven of these rows would pass while being wrong."""
    motif = allele_motif(n_variants=3)
    v_p = float(sp.N(motif.V_P.subs(numbers(motif))))
    assert abs(v_p - 1.2) < 1e-12
    assert abs(v_p - 1.0) > 0.1


# ======================================================================================
# structure
# ======================================================================================
def test_the_model_is_structurally_sound():
    motif = allele_motif(n_variants=2)
    model = motif.model
    assert [i for i in model.validate() if i.severity == "error"] == []
    assert model.is_recursive
    assert len(model.copaths) == 1
    assert model.copath_value("y_m", "y_p") == motif.mu


def test_alleles_are_indexed_by_parental_origin_not_transmission():
    """Origin is intrinsic; 'transmitted' is defined relative to a chosen descendant."""
    motif = allele_motif(n_variants=1, n_children=2)
    model = motif.model
    for who in motif.individuals:
        assert model.has_var(motif.z(who, "mat", 0))
        assert model.has_var(motif.z(who, "pat", 0))
    # both children draw from BOTH of each parent's alleles -- no "transmitted" node exists
    for child in motif.children:
        assert set(model.parents(motif.z(child, "mat", 0))) == {
            motif.z("m", "mat", 0),
            motif.z("m", "pat", 0),
            motif.s(child, "mat", 0),
        }
    assert not any("_T" in name or "trans" in name for name in model.names)


def test_transmission_coefficients_are_one_half_not_root_one_half():
    """Regression coefficients: E[child's allele | maternal] = (A + B)/2."""
    motif = allele_motif(n_variants=1)
    model = motif.model
    for parent_origin in ("mat", "pat"):
        assert model.path_coeff(
            motif.z("m", parent_origin, 0), motif.z("o", "mat", 0)
        ) == sp.Rational(1, 2)
        assert model.path_coeff(
            motif.z("p", parent_origin, 0), motif.z("o", "pat", 0)
        ) == sp.Rational(1, 2)


def test_every_exogenous_covariance_is_zero_except_the_one_copath():
    """The whole point: nothing about the assortment is written in beyond the co-path."""
    motif = allele_motif(n_variants=2)
    model = motif.model
    off_diagonal = [e for e in model.bidirected_edges if not e.is_variance]
    assert off_diagonal == [], [str(e) for e in off_diagonal]
    assert len(model.copaths) == 1


def test_exogenous_variances_are_as_specified():
    motif = allele_motif(n_variants=2)
    model = motif.model
    for origin in ("mat", "pat"):
        assert model.cov_value(motif.z("m", origin, 0), motif.z("m", origin, 0)) == sp.Rational(1, 2)
        assert model.cov_value(motif.s("o", origin, 0), motif.s("o", origin, 0)) == sp.Rational(1, 4)
    assert model.cov_value(motif.e("m"), motif.e("m")) == motif.V_E


# ======================================================================================
# the validation table -- GENERAL forms, at V_P != 1
# ======================================================================================
def test_validation_table():
    motif = allele_motif(n_variants=3, n_children=2)
    engine = pm.RAMEngine(motif.model)
    b, V_A, V_P, rho_y, rho_g = (
        motif.betas, motif.V_A, motif.V_P, motif.rho_y, motif.rho_g
    )
    o1, o2 = motif.children
    half = sp.Rational(1, 2)

    expected = {
        # the co-path reaches the causes -- NOT specified anywhere
        "Cov[z_mat[m,0], z_mat[f,0]]": (
            engine.cov(motif.z("m", "mat", 0), motif.z("p", "mat", 0)),
            b[0] ** 2 * rho_y / (4 * V_P),
        ),
        # cross-variant coupling
        "Cov[z_mat[m,0], z_mat[f,1]]": (
            engine.cov(motif.z("m", "mat", 0), motif.z("p", "mat", 1)),
            b[0] * b[1] * rho_y / (4 * V_P),
        ),
        # per-variant inflation
        "Var[x[o,0]]": (engine.var(motif.x(o1, 0)), 1 + b[0] ** 2 * rho_y / (2 * V_P)),
        # LINKAGE DISEQUILIBRIUM in the offspring, absent in the parents
        "Cov[x[o,0], x[o,1]]": (
            engine.cov(motif.x(o1, 0), motif.x(o1, 1)),
            b[0] * b[1] * rho_y / (2 * V_P),
        ),
        # the offspring is not in Hardy-Weinberg
        "Cov[z_mat[o,0], z_pat[o,0]]": (
            engine.cov(motif.z(o1, "mat", 0), motif.z(o1, "pat", 0)),
            b[0] ** 2 * rho_y / (4 * V_P),
        ),
        "Cov[x[m,0], x[o,0]]": (
            engine.cov(motif.x("m", 0), motif.x(o1, 0)),
            half + b[0] ** 2 * rho_y / (2 * V_P),
        ),
        "Var[g_o]": (engine.var(motif.g(o1)), V_A + rho_g * V_A / 2),
        # the three V_P-free rows
        "Cov[g_m, g_o]": (engine.cov(motif.g("m"), motif.g(o1)), V_A * (1 + rho_g) / 2),
        "Cov[y_m, y_o]": (engine.cov(motif.y("m"), motif.y(o1)), V_A * (1 + rho_y) / 2),
        "Cov[g_o1, g_o2]": (engine.cov(motif.g(o1), motif.g(o2)), V_A * (1 + rho_g) / 2),
    }
    for label, (got, want) in expected.items():
        assert sp.simplify(got - want) == 0, f"{label}: {sp.simplify(got)} != {sp.simplify(want)}"

    # and the defining property of the mating model
    assert sp.simplify(engine.cov("y_m", "y_p") - rho_y * V_P) == 0


def test_the_V_P_free_rows_really_are_V_P_free():
    """Three rows survive as written because rho_g absorbs the V_P; the others do not."""
    motif = allele_motif(n_variants=2)
    engine = pm.RAMEngine(motif.model)
    V_A, rho_y, rho_g = motif.V_A, motif.rho_y, motif.rho_g
    for got, want in [
        (engine.cov(motif.g("m"), motif.g("o")), V_A * (1 + rho_g) / 2),
        (engine.cov(motif.y("m"), motif.y("o")), V_A * (1 + rho_y) / 2),
    ]:
        assert sp.simplify(got - want) == 0
        assert motif.V_E not in sp.simplify(want).free_symbols or want.has(rho_g)


def test_the_V_P_one_forms_would_have_passed_wrongly():
    """Pin the trap itself: the omitted-1/V_P forms agree only when V_P = 1."""
    motif = allele_motif(n_variants=2)
    engine = pm.RAMEngine(motif.model)
    b, V_P, rho_y = motif.betas, motif.V_P, motif.rho_y

    got = engine.cov(motif.z("m", "mat", 0), motif.z("p", "mat", 0))
    general = b[0] ** 2 * rho_y / (4 * V_P)
    omitted = b[0] ** 2 * rho_y / 4

    assert sp.simplify(got - general) == 0
    assert sp.simplify(got - omitted) != 0  # wrong in general...
    at_unit_v_p = {motif.V_E: 1 - motif.V_A}  # ...but indistinguishable at V_P = 1
    assert sp.simplify((got - omitted).subs(at_unit_v_p)) == 0


# ======================================================================================
# the two substantive points
# ======================================================================================
def test_var_g_o_is_an_exact_perfect_square_not_an_approximation():
    """Diagonal and off-diagonal contributions combine into (sum beta_k^2)^2, exactly."""
    motif = allele_motif(n_variants=3)
    engine = pm.RAMEngine(motif.model)
    b, V_A, V_P, rho_y, rho_g = (
        motif.betas, motif.V_A, motif.V_P, motif.rho_y, motif.rho_g
    )
    M = motif.n_variants

    diagonal = sp.Add(*[b[k] ** 2 * (engine.var(motif.x("o", k)) - 1) for k in range(M)])
    off_diagonal = sp.Add(
        *[
            b[k] * b[j] * engine.cov(motif.x("o", k), motif.x("o", j))
            for k in range(M)
            for j in range(M)
            if k != j
        ]
    )
    assert sp.simplify(engine.var(motif.g("o")) - (V_A + diagonal + off_diagonal)) == 0
    assert sp.simplify(diagonal - rho_y / (2 * V_P) * sp.Add(*[b[k] ** 4 for k in range(M)])) == 0

    # the perfect square, asserted EXACTLY -- no polygenic approximation anywhere
    squares = sp.Add(*[b[k] ** 4 for k in range(M)])
    crosses = sp.Add(*[b[k] ** 2 * b[j] ** 2 for k in range(M) for j in range(M) if k != j])
    assert sp.expand(squares + crosses - V_A**2) == 0
    assert sp.simplify(engine.var(motif.g("o")) - (V_A + rho_g * V_A / 2)) == 0


@pytest.mark.parametrize("M", [2, 3, 4])
def test_per_variant_inflation_is_one_over_M_of_the_total(M):
    """The diagonal's SHARE of the total is 1/M -- the rigorous version of the hand-wave.

    Note which ratio: ``diagonal / (diagonal + off_diagonal) = 1/M``, whereas
    ``diagonal / off_diagonal = 1/(M-1)``. The share is the one that says the per-variant
    effect is negligible, and it is why the g-level model is legitimate.
    """
    motif = allele_motif(n_variants=M)
    engine = pm.RAMEngine(motif.model)
    b, V_P, rho_y = motif.betas, motif.V_P, motif.rho_y

    diagonal = sp.Add(*[b[k] ** 2 * (engine.var(motif.x("o", k)) - 1) for k in range(M)])
    off_diagonal = sp.Add(
        *[
            b[k] * b[j] * engine.cov(motif.x("o", k), motif.x("o", j))
            for k in range(M)
            for j in range(M)
            if k != j
        ]
    )
    equal_effects = {b[k]: sp.Symbol("b", positive=True) for k in range(M)}
    share = sp.simplify((diagonal / (diagonal + off_diagonal)).subs(equal_effects))
    assert sp.simplify(share - sp.Rational(1, M)) == 0

    ratio = sp.simplify((diagonal / off_diagonal).subs(equal_effects))
    assert sp.simplify(ratio - sp.Rational(1, M - 1)) == 0


def test_parent_offspring_exceeds_full_sibling_and_the_source_is_the_environment():
    motif = allele_motif(n_variants=3, n_children=2)
    engine = pm.RAMEngine(motif.model)
    V_A, rho_y, rho_g = motif.V_A, motif.rho_y, motif.rho_g
    o1, o2 = motif.children

    parent_offspring = engine.cov(motif.y("m"), motif.y(o1))
    full_sib = engine.cov(motif.y(o1), motif.y(o2))

    assert sp.simplify(full_sib - V_A * (1 + rho_g) / 2) == 0
    excess = sp.simplify(parent_offspring - full_sib)
    assert sp.simplify(excess - V_A * (rho_y - rho_g) / 2) == 0
    # ...and it comes ENTIRELY from one parent's environment correlating with the other
    # parent's transmitted alleles
    assert sp.simplify(excess - engine.cov(motif.e("m"), motif.g(o1))) == 0

    values = numbers(motif)
    assert float(sp.N(excess.subs(values))) > 0
    # under random mating the two are equal
    assert sp.simplify(excess.subs({rho_y: 0})) == 0


def test_rho_y_zero_collapses_to_random_mating():
    motif = allele_motif(n_variants=2, n_children=2)
    engine = pm.RAMEngine(motif.model)
    V_A, rho_y = motif.V_A, motif.rho_y
    o1, o2 = motif.children
    zero = {rho_y: 0}

    assert sp.simplify(engine.var(motif.x(o1, 0)).subs(zero) - 1) == 0          # no inflation
    assert sp.simplify(engine.cov(motif.x(o1, 0), motif.x(o1, 1)).subs(zero)) == 0  # no LD
    assert sp.simplify(
        engine.cov(motif.z(o1, "mat", 0), motif.z(o1, "pat", 0)).subs(zero)
    ) == 0                                                                       # Hardy-Weinberg
    assert sp.simplify(engine.var(motif.g(o1)).subs(zero) - V_A) == 0
    assert sp.simplify(engine.cov(motif.y("m"), motif.y(o1)).subs(zero) - V_A / 2) == 0
    assert sp.simplify(engine.cov(motif.y(o1), motif.y(o2)).subs(zero) - V_A / 2) == 0
    assert sp.simplify(engine.cov(motif.z("m", "mat", 0), motif.z("p", "mat", 0)).subs(zero)) == 0


def test_the_founders_are_unaffected_by_their_own_assortment():
    """The cleanest statement of what assortative mating does and does not do.

    All the linkage disequilibrium appears in the offspring; none of it in the parents. Their
    genotypes still have variance 1 and are still uncorrelated across variants.
    """
    motif = allele_motif(n_variants=3)
    engine = pm.RAMEngine(motif.model)
    for founder in motif.founders:
        for k in range(motif.n_variants):
            assert sp.simplify(engine.var(motif.x(founder, k)) - 1) == 0
            assert sp.simplify(
                engine.cov(motif.z(founder, "mat", k), motif.z(founder, "pat", k))
            ) == 0
            for j in range(motif.n_variants):
                if j != k:
                    assert sp.simplify(engine.cov(motif.x(founder, k), motif.x(founder, j))) == 0
    # but their genetic values ARE correlated with each other -- that is the assortment
    assert sp.simplify(engine.cov(motif.g("m"), motif.g("p")) - motif.rho_g * motif.V_A) == 0


# ======================================================================================
# segregation
# ======================================================================================
def test_segregation_variance_makes_the_genetic_variance_persist():
    """Omit it and you have blending inheritance, which halves V_A every generation."""
    motif = allele_motif(n_variants=2)
    engine = pm.RAMEngine(motif.model)
    V_A, rho_y = motif.V_A, motif.rho_y

    assert sp.simplify(engine.var(motif.g("o")).subs({rho_y: 0}) - V_A) == 0

    blending = motif.model.copy("no segregation")
    for k in range(motif.n_variants):
        for origin in ("mat", "pat"):
            blending.remove_cov(motif.s("o", origin, k), motif.s("o", origin, k))
    halved = pm.RAMEngine(blending).var(motif.g("o")).subs({rho_y: 0})
    assert sp.simplify(halved - V_A / 2) == 0


def test_the_effect_weighted_segregation_variance_is_V_K():
    """Two residuals per variant (one per allele), so V_K = sum beta_k^2 / 2 = V_A/2."""
    motif = allele_motif(n_variants=3)
    engine = pm.RAMEngine(motif.model)
    total = sp.Add(
        *[
            motif.betas[k] ** 2
            * (engine.var(motif.s("o", "mat", k)) + engine.var(motif.s("o", "pat", k)))
            for k in range(motif.n_variants)
        ]
    )
    assert sp.simplify(total - motif.V_A / 2) == 0


def test_the_documented_caveat_holds_exactly_here_and_would_bite_in_generation_three():
    """Var(s) = 1/4 requires the transmitting parent's alleles to be uncorrelated.

    Founders satisfy that exactly, so this motif is exact. The offspring do NOT: they acquire
    c = beta_k^2 rho_y / (4 V_P), so a third generation would need 1/4 - c/2. Pinned here so
    task-20260804-151350 cannot walk past it.
    """
    motif = allele_motif(n_variants=2)
    engine = pm.RAMEngine(motif.model)
    b, V_P, rho_y = motif.betas, motif.V_P, motif.rho_y

    for founder in motif.founders:
        assert engine.cov(motif.z(founder, "mat", 0), motif.z(founder, "pat", 0)) == 0

    c = engine.cov(motif.z("o", "mat", 0), motif.z("o", "pat", 0))
    assert sp.simplify(c - b[0] ** 2 * rho_y / (4 * V_P)) == 0
    assert sp.simplify(c.subs({rho_y: 0})) == 0  # vanishes under random mating


# ======================================================================================
# the co-path is doing the work
# ======================================================================================
def test_a_bidirected_edge_instead_of_the_copath_gives_zero_at_the_alleles():
    """The sharpest available test of task-20260804-173343."""
    motif = allele_motif(n_variants=2)
    faked = motif.model.copy("bidirected instead")
    faked.remove_copath("y_m", "y_p")
    faked.add_cov("y_m", "y_p", motif.rho_y * motif.V_P)

    engine = pm.RAMEngine(faked)
    assert sp.simplify(engine.cov("y_m", "y_p") - motif.rho_y * motif.V_P) == 0
    assert engine.cov(motif.z("m", "mat", 0), motif.z("p", "mat", 0)) == 0
    assert engine.cov(motif.g("m"), motif.g("p")) == 0
    assert engine.cov(motif.x("o", 0), motif.x("o", 1)) == 0  # and no LD reaches the offspring


def test_both_engines_agree_on_the_allele_motif():
    motif = allele_motif(n_variants=2)
    engine = pm.RAMEngine(motif.model)
    tracer = pm.WrightTracer(motif.model, max_chains=500_000)
    for x, y in [
        (motif.z("m", "mat", 0), motif.z("p", "mat", 0)),
        (motif.x("o", 0), motif.x("o", 1)),
        (motif.g("m"), motif.g("o")),
        (motif.y("m"), motif.y("p")),
        (motif.z("o", "mat", 0), motif.z("o", "pat", 0)),
    ]:
        assert sp.simplify(tracer.cov(x, y) - engine.cov(x, y)) == 0, f"Cov[{x}, {y}]"


# ======================================================================================
# builder API
# ======================================================================================
def test_effects_may_be_supplied_numerically():
    motif = allele_motif(n_variants=2, effects=[sp.Rational(3, 10), sp.Rational(4, 10)])
    assert sp.simplify(motif.V_A - sp.Rational(1, 4)) == 0
    engine = pm.RAMEngine(motif.model)
    assert sp.simplify(engine.var(motif.x("m", 0)) - 1) == 0


def test_builder_rejects_bad_arguments():
    with pytest.raises(ValueError, match="at least one variant"):
        allele_motif(n_variants=0)
    with pytest.raises(ValueError, match="at least one child"):
        allele_motif(n_children=0)
    with pytest.raises(ValueError, match="must match"):
        allele_motif(n_variants=3, effects=[1, 2])
    with pytest.raises(ValueError, match="'mat' or 'pat'"):
        allele_motif(n_variants=1).z("m", "transmitted", 0)


def test_node_count_grows_as_documented():
    """2M allele nodes + M genotype nodes + 3 per individual, plus 2M residuals per child."""
    for M in (1, 2, 3):
        motif = allele_motif(n_variants=M, n_children=1)
        per_individual = 3 * M + 3  # z_mat, z_pat, x per variant; g, e, y
        expected = 3 * per_individual + 2 * M  # three individuals + the child's residuals
        assert len(motif.model.names) == expected, M
