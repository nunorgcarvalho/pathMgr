"""Co-paths: the edge type for covariance attributable to matching.

Every expected value here is hand-derived (task-20260804-173343) and independently confirmed
against Sunde et al. 2025 Nat Commun, Supplementary Notes 1 and 3. Agreement between the two
engines on co-path models is enforced by the battery in ``test_agreement.py``; what is checked
here is that the co-path *semantics* are right, which agreement alone could not establish.

The claim that motivates the whole edge type: assortment induces correlation among **all the
causes** of the matched variables, so the association propagates backward up the graph. A
bidirected edge cannot do that. ``test_copath_reaches_the_causes_but_a_bidirected_edge_does_not``
is the decisive case.
"""

from pathlib import Path

import pytest
import sympy as sp

import pathmgr as pm

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


# ======================================================================================
# fixtures
# ======================================================================================
def mated_pair() -> pm.Model:
    """One mated pair, y = g + e, assorting on y with a single co-path."""
    m = pm.Model("mated pair", units=pm.Units.unstandardized())
    for v in ("V_A", "V_E"):
        m.declare(v, positive=True)
    V_A, V_E, rho_y = (m.sym(s) for s in ("V_A", "V_E", "rho_y"))
    for i in ("m", "f"):
        m.add_var(f"g_{i}", latent=True)
        m.add_var(f"e_{i}", latent=True)
        m.add_var(f"y_{i}")
        m.add_path(f"g_{i}", f"y_{i}", 1)
        m.add_path(f"e_{i}", f"y_{i}", 1)
        m.add_variance(f"g_{i}", V_A)
        m.add_variance(f"e_{i}", V_E)
    m.add_copath("y_m", "y_f", rho_y / (V_A + V_E))  # mu = rho_y / V_P
    return m


def allele_level_pair() -> pm.Model:
    """As above, but each parent's g is built from two allele nodes: g = beta*(z_mat + z_pat)."""
    m = pm.Model("allele-level pair", units=pm.Units.unstandardized())
    for v in ("beta", "V_E"):
        m.declare(v, positive=True)
    beta, V_E, rho_y = (m.sym(s) for s in ("beta", "V_E", "rho_y"))
    V_P = beta**2 + V_E  # Var[g] = beta^2 * (1/2 + 1/2) = beta^2
    for i in ("m", "f"):
        for allele in ("mat", "pat"):
            m.add_var(f"z_{allele}_{i}", latent=True)
            m.add_variance(f"z_{allele}_{i}", sp.Rational(1, 2))
        m.add_var(f"g_{i}", latent=True)
        m.add_var(f"e_{i}", latent=True)
        m.add_var(f"y_{i}")
        m.add_path(f"z_mat_{i}", f"g_{i}", beta)
        m.add_path(f"z_pat_{i}", f"g_{i}", beta)
        m.add_path(f"g_{i}", f"y_{i}", 1)
        m.add_path(f"e_{i}", f"y_{i}", 1)
        m.add_variance(f"e_{i}", V_E)
    m.add_copath("y_m", "y_f", rho_y / V_P)
    return m


def shared_partner() -> pm.Model:
    """Individuals 1 and 2 each mated with the same third person P: two mating processes."""
    m = pm.Model("shared partner", units=pm.Units.unstandardized())
    for v in ("V_A", "V_E"):
        m.declare(v, positive=True)
    V_A, V_E, rho_y = (m.sym(s) for s in ("V_A", "V_E", "rho_y"))
    for i in ("1", "P", "2"):
        m.add_var(f"g_{i}", latent=True)
        m.add_var(f"e_{i}", latent=True)
        m.add_var(f"y_{i}")
        m.add_path(f"g_{i}", f"y_{i}", 1)
        m.add_path(f"e_{i}", f"y_{i}", 1)
        m.add_variance(f"g_{i}", V_A)
        m.add_variance(f"e_{i}", V_E)
    mu = rho_y / (V_A + V_E)
    m.add_copath("y_1", "y_P", mu, process="couple_1P")
    m.add_copath("y_2", "y_P", mu, process="couple_2P")
    return m


def _pair_symbols(m: pm.Model):
    V_A, V_E, rho_y = (m.sym(s) for s in ("V_A", "V_E", "rho_y"))
    V_P = V_A + V_E
    return V_A, V_E, rho_y, V_P, rho_y * V_A / V_P  # last is rho_g


# ======================================================================================
# the four single-pair identities
# ======================================================================================
@pytest.mark.parametrize("engine", ["ram", "tracer"])
@pytest.mark.parametrize(
    "x,y,expected",
    [
        ("y_m", "y_f", "rho_y * V_P"),
        ("g_m", "g_f", "rho_g * V_A"),
        ("e_m", "g_f", "rho_g * V_E"),
        ("e_m", "e_f", "rho_y * V_E**2 / V_P"),
    ],
)
def test_single_pair_identities(engine, x, y, expected):
    m = mated_pair()
    V_A, V_E, rho_y, V_P, rho_g = _pair_symbols(m)
    want = eval(expected, {}, {"V_A": V_A, "V_E": V_E, "rho_y": rho_y, "V_P": V_P, "rho_g": rho_g})
    got = (pm.RAMEngine(m) if engine == "ram" else pm.WrightTracer(m)).cov(x, y)
    assert sp.simplify(got - want) == 0


def test_partner_covariance_is_rho_y_times_V_P():
    """The defining property of the mating model. A standing test, per the task."""
    m = mated_pair()
    _, _, rho_y, V_P, _ = _pair_symbols(m)
    for engine in (pm.RAMEngine(m), pm.WrightTracer(m)):
        assert sp.simplify(engine.cov("y_m", "y_f") - rho_y * V_P) == 0
    # and the correlation really is rho_y
    corr = pm.RAMEngine(m).corr("y_m", "y_f")
    assert sp.simplify((corr / rho_y) ** 2 - 1) == 0


def test_partner_correlations_decompose_as_sunde_equation_2():
    """Sunde Eq. (2): the partner correlation is the sum of induced correlations in causes."""
    m = mated_pair()
    V_A, V_E, rho_y, V_P, _ = _pair_symbols(m)
    e = pm.RAMEngine(m)
    total = e.cov("y_m", "y_f")
    parts = (
        e.cov("g_m", "g_f") + e.cov("e_m", "e_f") + e.cov("e_m", "g_f") + e.cov("g_m", "e_f")
    )
    assert sp.simplify(total - parts) == 0
    assert sp.simplify(total - rho_y * V_P) == 0


def test_a_copath_induces_covariance_without_causing_variance():
    """Sunde's defining phrase. Adding a co-path must not change any variance."""
    m = mated_pair()
    without = m.copy()
    without.remove_copath("y_m", "y_f")
    with_engine, without_engine = pm.RAMEngine(m), pm.RAMEngine(without)
    for node in m.names:
        assert sp.simplify(with_engine.var(node) - without_engine.var(node)) == 0, node
    # but it does change the cross-partner covariances
    assert without_engine.cov("y_m", "y_f") == 0
    assert with_engine.cov("y_m", "y_f") != 0


# ======================================================================================
# the decisive test
# ======================================================================================
@pytest.mark.parametrize("engine", ["ram", "tracer"])
def test_copath_reaches_the_causes_but_a_bidirected_edge_does_not(engine):
    """Cov[z_mat_m, z_mat_f] = beta^2 rho_y / (4 V_P), WITHOUT being specified.

    This is what proves a co-path is not reducible to a bidirected edge: the association has to
    propagate backward from the matched phenotypes to their causes. The same model with a
    bidirected edge at (y_m, y_f) gives 0 for every cause-level covariance.
    """
    m = allele_level_pair()
    beta, V_E, rho_y = (m.sym(s) for s in ("beta", "V_E", "rho_y"))
    V_P = beta**2 + V_E
    engine_obj = pm.RAMEngine(m) if engine == "ram" else pm.WrightTracer(m)

    assert sp.simplify(engine_obj.cov("z_mat_m", "z_mat_f") - beta**2 * rho_y / (4 * V_P)) == 0
    # every allele pairing across partners, by symmetry
    for a in ("mat", "pat"):
        for b in ("mat", "pat"):
            got = engine_obj.cov(f"z_{a}_m", f"z_{b}_f")
            assert sp.simplify(got - beta**2 * rho_y / (4 * V_P)) == 0
    # sanity: the allele decomposition really does give Var[g] = beta^2 and the pair identity
    assert sp.simplify(engine_obj.var("g_m") - beta**2) == 0
    assert sp.simplify(engine_obj.cov("y_m", "y_f") - rho_y * V_P) == 0


def test_a_bidirected_edge_gives_zero_at_the_causes():
    """The contrast that makes the point -- same model, bidirected edge instead of a co-path."""
    m = allele_level_pair()
    beta, V_E, rho_y = (m.sym(s) for s in ("beta", "V_E", "rho_y"))
    V_P = beta**2 + V_E

    faked = m.copy("bidirected instead")
    faked.remove_copath("y_m", "y_f")
    faked.add_cov("y_m", "y_f", rho_y * V_P)  # tuned to give the right Cov[y_m, y_f]

    e = pm.RAMEngine(faked)
    assert sp.simplify(e.cov("y_m", "y_f") - rho_y * V_P) == 0  # the matched variables: fine
    assert e.cov("z_mat_m", "z_mat_f") == 0  # the causes: wrong
    assert e.cov("g_m", "g_f") == 0
    assert e.cov("e_m", "e_f") == 0


# ======================================================================================
# multiple co-paths
# ======================================================================================
@pytest.mark.parametrize("engine", ["ram", "tracer"])
def test_two_copaths_in_series_through_a_shared_partner(engine):
    """Cov[g_1, g_2] = rho_g^2 V_P: nonzero despite no shared ancestry at all.

    Two different mating processes, so a chain may cross both. This is the test that
    multi-co-path chains work -- and it is also why half-siblings do not follow
    ((1+rho_g)/2)^d: their two outer parents are correlated through exactly this route,
    with no common ancestor involved.
    """
    m = shared_partner()
    V_A, V_E, rho_y = (m.sym(s) for s in ("V_A", "V_E", "rho_y"))
    V_P = V_A + V_E
    rho_g = rho_y * V_A / V_P
    engine_obj = pm.RAMEngine(m) if engine == "ram" else pm.WrightTracer(m)

    assert sp.simplify(engine_obj.cov("g_1", "g_2") - rho_g**2 * V_P) == 0
    assert sp.simplify(engine_obj.cov("y_1", "y_2") - rho_y**2 * V_P) == 0
    # under random mating (no assortment) both vanish
    assert sp.simplify(engine_obj.cov("g_1", "g_2").subs({rho_y: 0})) == 0


def test_the_chain_crossing_two_copaths_is_itemized():
    """A reader hand-checking against Sunde must see which co-paths a chain crossed."""
    m = shared_partner()
    d = pm.WrightTracer(m).trace("g_1", "g_2")
    assert len(d) == 2  # via g_P and via e_P, i.e. summing to Var[y_P]
    for chain in d:
        assert len(chain.crossings) == 2
        assert set(chain.copath_processes) == {"couple_1P", "couple_2P"}
        assert chain.crosses_copaths
    assert "couple_1P" in str(d) and "--" in str(d)
    # The middle segment goes from y_P back to itself, so summed over chains it bundles to
    # Var[y_P] -- exactly Sunde's shortcut of "multiply by all valid paths from y_P to itself".
    V_A, V_E = m.sym("V_A"), m.sym("V_E")
    middles = [c.segments[1] for c in d]
    assert all(s.start == "y_P" and s.end == "y_P" for s in middles)
    assert sp.simplify(sum(s.bidirected_value for s in middles) - (V_A + V_E)) == 0


def test_the_same_mating_process_is_used_at_most_once_per_chain():
    """Sunde, Supplementary Note 3. Without it, Var[y] would gain spurious co-path terms."""
    m = pm.Model("one couple")
    for i in ("m", "f"):
        m.add_var(f"y_{i}")
        m.add_variance(f"y_{i}", "V_P")
    m.add_copath("y_m", "y_f", "mu")

    d = pm.WrightTracer(m).trace("y_m", "y_m")
    assert max((len(c.crossings) for c in d), default=0) == 0  # no chain reuses the co-path
    assert sp.simplify(pm.WrightTracer(m).var("y_m") - m.sym("V_P")) == 0
    assert sp.simplify(pm.RAMEngine(m).var("y_m") - m.sym("V_P")) == 0


def test_two_copaths_on_one_mating_process_are_still_limited_to_one_per_chain():
    """Cross-trait assortment: one couple, several co-paths, sharing a process identifier."""
    m = pm.Model("cross-trait")
    for i in ("m", "f"):
        for v in ("S", "Sx"):
            m.add_var(f"{v}_{i}")
            m.add_variance(f"{v}_{i}", f"V_{v}")
    m.add_copath("S_m", "S_f", "mu", process="couple")
    m.add_copath("S_m", "Sx_f", "mu_prime", process="couple")
    assert m.mating_processes == ("couple",)

    for chain in pm.WrightTracer(m).trace("S_m", "Sx_f"):
        assert len(chain.crossings) <= 1
    # naming them as distinct processes WOULD allow a two-crossing chain -- the rule is real
    split = pm.Model("cross-trait, split processes")
    for i in ("m", "f"):
        for v in ("S", "Sx"):
            split.add_var(f"{v}_{i}")
            split.add_variance(f"{v}_{i}", f"V_{v}")
    split.add_copath("S_m", "S_f", "mu", process="a")
    split.add_copath("S_m", "Sx_f", "mu_prime", process="b")
    assert any(len(c.crossings) == 2 for c in pm.WrightTracer(split).trace("S_f", "Sx_f"))


# ======================================================================================
# RAM/tracer agreement, and why the geometric series is not the closed form
# ======================================================================================
@pytest.mark.parametrize(
    "factory", [mated_pair, allele_level_pair, shared_partner], ids=lambda f: f.__name__
)
def test_engines_agree_on_copath_models(factory):
    m = factory()
    e, t = pm.RAMEngine(m), pm.WrightTracer(m)
    for i, x in enumerate(m.names):
        for y in m.names[i:]:
            assert sp.simplify(t.cov(x, y) - e.cov(x, y)) == 0, f"Cov[{x}, {y}]"


@pytest.mark.parametrize(
    "factory", [mated_pair, allele_level_pair, shared_partner], ids=lambda f: f.__name__
)
def test_copath_sigma_stays_symmetric(factory):
    sigma = pm.RAMEngine(factory()).sigma()
    assert sp.simplify(sigma - sigma.T) == sp.zeros(sigma.rows, sigma.rows)


def test_the_geometric_series_form_overcounts():
    """`Sigma_0 (I - C Sigma_0)^-1` is NOT the closed form: it reuses the same co-path.

    Expanding it term by term on a single mated pair gives `rho_y V_P` (legal) plus
    `rho_y^3 V_P`, `rho_y^5 V_P`, ... -- chains traversing the one co-path three, five, ...
    times, which Sunde's rules forbid. Recorded as a test so nobody 'simplifies' the enumeration
    into a matrix inverse later.
    """
    m = mated_pair()
    V_A, V_E, rho_y, V_P, _ = _pair_symbols(m)
    e = pm.RAMEngine(m)
    index = {n: i for i, n in enumerate(e.order)}

    without = m.copy()
    without.remove_copath("y_m", "y_f")
    sigma0 = pm.RAMEngine(without).sigma()

    n = sigma0.rows
    C = sp.zeros(n, n)
    mu = rho_y / V_P
    C[index["y_m"], index["y_f"]] = mu
    C[index["y_f"], index["y_m"]] = mu
    geometric = sigma0 * (sp.eye(n) - C * sigma0).inv()

    entry = (index["y_m"], index["y_f"])
    assert sp.simplify(e.sigma()[entry] - rho_y * V_P) == 0        # pathMgr: correct
    assert sp.simplify(geometric[entry] - rho_y * V_P) != 0        # series: not
    # and the excess is exactly the odd-power reuse terms
    excess = sp.simplify(geometric[entry] - rho_y * V_P)
    assert sp.simplify(excess - rho_y**3 * V_P / (1 - rho_y**2)) == 0

    numbers = {V_A: sp.Rational(46, 100), V_E: sp.Rational(6, 10), rho_y: sp.Rational(3, 10)}
    assert abs(float(sp.N(e.sigma()[entry].subs(numbers))) - 0.318) < 1e-9
    assert float(sp.N(geometric[entry].subs(numbers))) > 0.349


def test_copath_sequence_limit_raises_rather_than_truncating():
    m = shared_partner()
    with pytest.raises(pm.CoPathLimitError, match="incomplete"):
        pm.RAMEngine(m, max_copath_sequences=1).sigma()


def test_ram_handles_copaths_on_a_cyclic_model_where_the_tracer_cannot():
    """The co-path machinery composes with the inverse fallback, so this still works."""
    m = pm.Model("cyclic with a co-path")
    m.add_vars("x", "y", "z", "w")
    m.add_path("x", "y", "a")
    m.add_path("y", "z", "b")
    m.add_path("z", "y", "d")  # feedback loop
    for v in ("x", "y", "z", "w"):
        m.add_variance(v, f"S_{v}")
    m.add_copath("y", "w", "mu")

    e = pm.RAMEngine(m)
    assert e.used_inverse
    assert sp.simplify(e.cov("y", "w") - m.sym("mu") * e.var("y") * e.var("w")) == 0
    with pytest.raises(pm.UntraceableModelError):
        pm.WrightTracer(m).trace("y", "w")


# ======================================================================================
# the corrected example, against the superseded one
# ======================================================================================
def test_copath_example_reproduces_the_writeup_results():
    """The 151346/151347 validation must still hold with the co-path encoding."""
    m = pm.from_text((EXAMPLES / "am_equilibrium.pmg").read_text(), name="AM co-path")
    e = pm.RAMEngine(m)
    V_A, V_E, V_K, V_A0, rho_y = (
        m.sym(s) for s in ("V_A_eq", "V_E", "V_K", "V_A0", "rho_y")
    )
    V_P = V_A + V_E
    rho_g = rho_y * V_A / V_P

    assert [i for i in m.validate() if i.severity == "error"] == []
    assert sp.simplify(e.cov("y_o1", "y_o2") - V_A * (1 + rho_g) / 2) == 0   # collateral
    assert sp.simplify(e.cov("y_m", "y_o1") - V_A * (1 + rho_y) / 2) == 0    # lineal
    assert sp.simplify(
        e.var("y_o1").subs({V_K: V_A0 / 2}) - (V_A0 / 2 + V_A * (1 + rho_g) / 2 + V_E)
    ) == 0                                                                    # recursion
    assert sp.simplify(e.cov("y_m", "y_f") - rho_y * V_P) == 0                # the fix


def test_the_superseded_encoding_is_wrong_in_exactly_the_documented_way():
    """Pins the bug the co-path fixes, so the account in the fixture cannot rot."""
    old = pm.from_text((EXAMPLES / "am_equilibrium_handwritten.pmg").read_text())
    e = pm.RAMEngine(old)
    V_A, V_E, rho_y, rho_g = (old.sym(s) for s in ("V_A_eq", "V_E", "rho_y", "rho_g"))
    V_P = V_A + V_E

    assert sp.simplify(e.cov("y_m", "y_f") - (V_A * rho_g + 2 * V_E * rho_g)) == 0
    assert e.cov("e_m", "e_f") == 0
    fixed_point = {rho_g: rho_y * V_A / V_P}
    shortfall = sp.simplify((rho_y * V_P - e.cov("y_m", "y_f")).subs(fixed_point))
    assert sp.simplify(shortfall - rho_y * V_E**2 / V_P) == 0

    numbers = {V_A: sp.Rational(46, 100), V_E: sp.Rational(6, 10), rho_y: sp.Rational(3, 10)}
    assert abs(float(sp.N(e.cov("y_m", "y_f").subs(fixed_point).subs(numbers))) - 0.2161) < 1e-3


def test_both_encodings_agree_on_everything_the_old_one_got_right():
    """The co-path machinery reproduces what was verified by hand -- a real cross-check."""
    new = pm.from_text((EXAMPLES / "am_equilibrium.pmg").read_text())
    old = pm.from_text((EXAMPLES / "am_equilibrium_handwritten.pmg").read_text())
    e_new, e_old = pm.RAMEngine(new), pm.RAMEngine(old)
    rho_g = old.sym("rho_g")
    fixed_point = {rho_g: old.sym("rho_y") * old.sym("V_A_eq") / (old.sym("V_A_eq") + old.sym("V_E"))}

    for x, y in [
        ("g_m", "g_f"), ("e_m", "g_f"), ("y_o1", "y_o2"), ("y_m", "y_o1"),
        ("g_o1", "g_o2"), ("y_o1", "y_o1"), ("g_m", "g_o1"),
    ]:
        assert sp.simplify(e_new.cov(x, y) - e_old.cov(x, y).subs(fixed_point)) == 0, f"{x},{y}"
    # and they differ on exactly the omission
    assert sp.simplify(
        e_new.cov("e_m", "e_f") - e_old.cov("e_m", "e_f").subs(fixed_point)
    ) != 0


# ======================================================================================
# model API and text grammar
# ======================================================================================
def test_copath_api_basics():
    m = pm.Model()
    m.add_vars("a", "b")
    m.add_copath("b", "a", "mu")
    assert len(m.copaths) == 1
    assert m.has_copaths
    assert m.copath_value("a", "b") == m.sym("mu")
    assert m.copath_value("b", "a") == m.sym("mu")  # order-independent
    assert m.mating_processes == ("a--b",)
    assert "co-paths (1)" in m.describe()
    m.remove_copath("a", "b")
    assert not m.has_copaths


def test_a_copath_has_no_self_form():
    m = pm.Model().add_vars("a")
    with pytest.raises(ValueError, match="no self form"):
        m.add_copath("a", "a", "mu")


def test_duplicate_copath_rejected_but_a_second_process_is_allowed():
    m = pm.Model().add_vars("a", "b")
    m.add_copath("a", "b", "mu")
    with pytest.raises(ValueError, match="already specified"):
        m.add_copath("a", "b", "mu2")
    m.add_copath("a", "b", "mu2", process="other")
    assert len(m.copaths) == 2
    assert set(m.mating_processes) == {"a--b", "other"}


def test_copath_survives_copy_and_bumps_revision():
    m = pm.Model().add_vars("a", "b")
    before = m.revision
    m.add_copath("a", "b", "mu")
    assert m.revision > before
    assert m.copy().copaths == m.copaths


def test_text_grammar_round_trips_copaths():
    m = pm.from_text(
        """
        y_m ~~ V_P*y_m
        y_f ~~ V_P*y_f
        y_m -- mu*y_f
        S_m -- mu2*S_p [couple0]
        """
    )
    assert len(m.copaths) == 2
    assert set(m.mating_processes) == {"y_f--y_m", "couple0"}
    again = pm.from_text(m.to_text())
    assert [str(c) for c in again.copaths] == [str(c) for c in m.copaths]


def test_a_coefficient_containing_a_double_minus_is_not_a_copath():
    """Operator choice must not misparse `(a--b)` as the co-path operator."""
    m = pm.from_text("y ~ (a--b)*x")
    assert not m.has_copaths
    assert sp.simplify(m.path_coeff("x", "y") - (m.sym("a") + m.sym("b"))) == 0


def test_empty_process_name_rejected():
    with pytest.raises(pm.TextSyntaxError, match="empty mating-process name"):
        pm.from_text("y_m -- mu*y_f []")


def test_copath_edges_are_exposed_for_diagram_highlighting():
    """task-20260804-151349 will need to draw the co-path Sunde renders as a plain line."""
    m = shared_partner()
    chain = pm.WrightTracer(m).trace("g_1", "g_2").chains[0]
    assert chain.copath_edges()
    for a, b in chain.copath_edges():
        assert m.copath_value(a, b) is not None or any(
            c.a == a and c.b == b for c in m.copaths
        )


def test_validate_flags_a_copath_and_bidirected_edge_on_the_same_pair():
    """Almost always double-counting: the co-path already induces that covariance."""
    m = pm.Model().add_vars("a", "b")
    m.add_variance("a", "V")
    m.add_variance("b", "V")
    m.add_copath("a", "b", "mu")
    m.add_cov("a", "b", "c")
    messages = [str(i) for i in m.validate()]
    assert any("BOTH a co-path and a bidirected edge" in s for s in messages)


def test_validate_flags_an_inert_copath():
    """A co-path contributes mu*Var[a]*Var[b]; with no variance at an end it does nothing."""
    m = pm.Model().add_vars("a", "b")
    m.add_variance("a", "V")
    m.add_copath("a", "b", "mu")  # b has no variance and no causes
    messages = [str(i) for i in m.validate()]
    assert any("contributes\nnothing" in s.replace(" nothing", "\nnothing") for s in messages) or any(
        "this one contributes" in s for s in messages
    )
    assert pm.RAMEngine(m).cov("a", "b") == 0


# ======================================================================================
# the coordinator's validation pedigree, and where a sequential rank-one update breaks
# ======================================================================================
def test_half_sibling_pedigree_matches_every_hand_derived_value():
    """The validation set from task-20260804-173343, on both engines.

    Two mating processes sharing an individual A. Note two results random mating cannot produce
    at all: the in-laws B and B2 covary at rho_g^2 V_P with no common ancestor, and the half-sib
    value exceeds the collateral formula V_A(1+rho_g)^2/4 by rho_g^2 V_E/4 -- which is why
    half-sibs do not follow ((1+rho_g)/2)^d.
    """
    from battery import half_sibling_pedigree

    m = half_sibling_pedigree()
    V_A, V_E, V_K, rho_y = (m.sym(s) for s in ("V_A", "V_E", "V_K", "rho_y"))
    V_P = V_A + V_E
    rho_g = rho_y * V_A / V_P
    expected = {
        ("y_A", "y_B"): rho_y * V_P,
        ("g_A", "g_B"): rho_g * V_A,
        ("e_A", "e_B"): rho_y * V_E**2 / V_P,
        ("e_A", "g_B"): rho_g * V_E,
        ("y_A", "y_A"): V_P,                                             # variance unchanged
        ("g_A", "g_A"): V_A,                                             # variance unchanged
        ("g_A", "g_E1"): V_A * (1 + rho_g) / 2,
        ("y_A", "y_E1"): V_A * (1 + rho_y) / 2,
        ("g_E1", "g_E2"): V_A * (1 + rho_g) / 2,
        ("g_B", "g_B2"): rho_g**2 * V_P,                                 # in-laws
        ("g_E1", "g_H"): (V_A * (1 + 2 * rho_g) + rho_g**2 * V_P) / 4,    # half-sibs
    }
    for engine in (pm.RAMEngine(m), pm.WrightTracer(m)):
        for (x, y), want in expected.items():
            assert sp.simplify(engine.cov(x, y) - want) == 0, f"Cov[{x}, {y}]"

    # Var[g_E1] = V_A(1+rho_g)/2 + V_K = V_A + rho_g V_A / 2 with V_K = V_A/2.
    # The task's expected value was written as V_A + rho_y V_A^2 / 2, which is this ONLY when
    # V_P = 1 -- exactly the generation-indexing trap the task itself warns about, hidden by the
    # numeric oracle's choice of V_A = 0.4, V_E = 0.6. Assert the general form.
    got = pm.RAMEngine(m).var("g_E1").subs({V_K: V_A / 2})
    assert sp.simplify(got - (V_A + rho_g * V_A / 2)) == 0
    assert sp.simplify(got - (V_A + rho_y * V_A**2 / (2 * V_P))) == 0
    assert sp.simplify((got - (V_A + rho_y * V_A**2 / 2)).subs({V_E: 1 - V_A})) == 0

    excess = sp.simplify(expected[("g_E1", "g_H")] - V_A * (1 + rho_g) ** 2 / 4)
    assert sp.simplify(excess - rho_g**2 * V_E / 4) == 0


def test_rho_y_zero_collapses_to_random_mating():
    from battery import half_sibling_pedigree

    m = half_sibling_pedigree()
    V_A, V_K, rho_y = (m.sym(s) for s in ("V_A", "V_K", "rho_y"))
    e = pm.RAMEngine(m)
    zero = {rho_y: 0, V_K: V_A / 2}
    assert sp.simplify(e.var("g_E1").subs(zero) - V_A) == 0
    assert sp.simplify(e.cov("g_A", "g_E1").subs(zero) - V_A / 2) == 0
    assert sp.simplify(e.cov("g_E1", "g_E2").subs(zero) - V_A / 2) == 0
    assert sp.simplify(e.cov("g_E1", "g_H").subs(zero) - V_A / 4) == 0
    assert sp.simplify(e.cov("g_A", "g_B").subs(zero)) == 0
    assert sp.simplify(e.cov("g_B", "g_B2").subs(zero)) == 0


def _sequential_rank_one_sigma(model, copaths):
    """Sigma by one symmetric rank-one update per co-path against the RUNNING Sigma.

    The construction proposed in task-20260804-173343 §3, reproduced only so the test below can
    show where it departs from the tracer. This is NOT what the engine does.
    """
    bare = model.copy()
    for c in list(bare.copaths):
        bare.remove_copath(c.a, c.b, c.process)
    sigma = pm.RAMEngine(bare).sigma()
    index = {n: i for i, n in enumerate(bare.names)}
    for c in copaths:
        v1 = sigma[:, index[c.a]]
        v2 = sigma[:, index[c.b]]
        sigma = sp.expand(sigma + c.coefficient * (v1 * v2.T + v2 * v1.T))
    return sigma, index


def test_sequential_rank_one_updates_agree_where_partners_are_unrelated():
    """On the half-sibling pedigree the two constructions coincide -- hence it validated."""
    from battery import half_sibling_pedigree

    m = half_sibling_pedigree()
    engine = pm.RAMEngine(m)
    sigma, index = _sequential_rank_one_sigma(m, list(m.copaths))
    for x in ("g_E1", "g_H", "g_B", "g_B2", "y_A"):
        for y in ("g_E1", "g_H", "g_B", "g_B2", "y_A"):
            assert sp.simplify(sigma[index[x], index[y]] - engine.cov(x, y)) == 0


def test_sequential_rank_one_updates_reuse_a_copath():
    """...but they break once the COUPLE-relatedness graph has a cycle.

    `A x B` mated; `A` also has child `C`, `B` also has child `D`; then `C x D` mate, so couple 2
    has one member related to each member of couple 1. Updating against the running Sigma lets
    couple 1's co-path be crossed on *both* legs of couple 2's update -- one mating process used
    twice in a chain, which Sunde's rule forbids. The symptom is that the answer depends on the
    order the updates are applied, so it cannot be right in both orders.

    pathMgr enumerates sequences of *distinct* mating processes instead, which is order-free by
    construction and agrees with the tracer here.
    """
    from battery import couple_relatedness_cycle

    m = couple_relatedness_cycle()
    engine, tracer = pm.RAMEngine(m), pm.WrightTracer(m)
    copaths = list(m.copaths)
    forward, index = _sequential_rank_one_sigma(m, copaths)
    backward, _ = _sequential_rank_one_sigma(m, list(reversed(copaths)))
    entry = (index["g_C"], index["g_D"])

    assert sp.simplify(tracer.cov("g_C", "g_D") - engine.cov("g_C", "g_D")) == 0
    assert sp.simplify(forward[entry] - engine.cov("g_C", "g_D")) != 0
    assert sp.simplify(forward[entry] - backward[entry]) != 0

    numbers = {
        m.sym("V_A"): sp.Rational(40, 100),
        m.sym("V_E"): sp.Rational(60, 100),
        m.sym("rho_y"): sp.Rational(30, 100),
        m.sym("V_K"): sp.Rational(20, 100),
    }
    exact = float(sp.N(engine.cov("g_C", "g_D").subs(numbers)))
    assert abs(exact - 0.06) < 1e-12
    assert float(sp.N(forward[entry].subs(numbers))) > exact
    assert float(sp.N(backward[entry].subs(numbers))) > exact


def test_pathmgr_copath_result_is_order_free():
    """Nothing in the engine depends on the order co-paths were added."""
    from battery import couple_relatedness_cycle

    m = couple_relatedness_cycle()
    shuffled = m.copy()  # copies the symbol registry, so assumptions match
    original = list(m.copaths)
    for c in original:
        shuffled.remove_copath(c.a, c.b, c.process)
    for c in reversed(original):  # re-added in the opposite order
        shuffled.add_copath(c.a, c.b, c.coefficient, process=c.process)
    assert [c.process for c in shuffled.copaths] == [c.process for c in reversed(original)]

    a, b = pm.RAMEngine(m), pm.RAMEngine(shuffled)
    for x in ("g_C", "g_D", "y_C", "y_D", "g_A", "g_B"):
        for y in ("g_C", "g_D", "y_C", "y_D", "g_A", "g_B"):
            assert sp.simplify(a.cov(x, y) - b.cov(x, y)) == 0


def test_a_copath_is_findable_by_its_endpoints_without_naming_the_process():
    """Naming a process at construction must not make the edge unfindable afterwards."""
    m = pm.Model().add_vars("a", "b")
    m.add_copath("a", "b", "mu", process="founding couple")
    assert m.copath_value("a", "b") == m.sym("mu")
    assert m.copath_value("b", "a") == m.sym("mu")           # order-independent
    assert m.copath_value("a", "b", process="founding couple") == m.sym("mu")
    assert m.copath_value("a", "b", process="nope") is None
    assert len(m.copaths_between("a", "b")) == 1
    m.remove_copath("a", "b")
    assert not m.has_copaths


def test_an_ambiguous_copath_lookup_demands_the_process():
    m = pm.Model().add_vars("a", "b")
    m.add_copath("a", "b", "mu", process="first")
    m.add_copath("a", "b", "mu2", process="second")
    with pytest.raises(ValueError, match="Name the process"):
        m.copath_value("a", "b")
    assert m.copath_value("a", "b", process="second") == m.sym("mu2")
    m.remove_copath("a", "b", process="first")
    assert m.copath_value("a", "b") == m.sym("mu2")  # unambiguous again


def test_removing_a_missing_copath_is_a_clear_error():
    m = pm.Model().add_vars("a", "b")
    with pytest.raises(KeyError, match="no co-path between"):
        m.remove_copath("a", "b")
