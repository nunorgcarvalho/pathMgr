"""Can the specification API express the *hard* case correctly? A spike, not the genetics layer.

Task-20260804-151346 only requires the two smoke models of §3. This file goes one step
further on purpose: it encodes a single assortative-mating transmission unit (a mated pair
and two children) and checks, via the same throwaway RAM helper, that the specification
reproduces three hand-derived results from popstatgenwriteups'
``relative_covariance.tex``:

  - lineal    (eq:am-level3-lin, d=1):  Cov[y_m, y_o]   = V_A_eq (1 + rho_y) / 2
  - collateral(eq:am-level3-col, d=1):  Cov[y_o1, y_o2] = V_A_eq (1 + rho_g) / 2
  - the variance recursion (eq:am-recursion): V_A(t+1) = V_A0/2 + V_A(t)(1 + rho_g)/2

The point is to de-risk every downstream task before the engine is written. Two things it
establishes about the API, both of which drove design choices:

1. The lineal result only appears **after** substituting the fixed-point relation
   ``rho_g = rho_y h2_eq``. Before substitution the model gives
   ``V_A_eq(1 + rho_g)/2 + rho_g V_E/2`` -- correct but unrecognisable. That relation is not
   an edge, which is why :meth:`Model.assume` exists.
2. The lineal/collateral asymmetry is carried entirely by ``Cov[e_m, g_f] = rho_g V_E`` --
   a bidirected edge between one individual's *environment* and another's *genes*. So
   bidirected edges must be allowed between arbitrary latents, with expression values.

The real versions of these checks belong to task-20260804-151351, against the real engine.
"""

from pathlib import Path

import sympy as sp

import pathmgr as pm

from conftest import canonical, ram_sigma

# The builder model below is the hand-written-covariance encoding, superseded by co-paths in
# task-20260804-173343. Its text twin is therefore the handwritten fixture, not the live example.
AM_TEXT_FILE = (
    Path(__file__).resolve().parent.parent / "examples" / "am_equilibrium_handwritten.pmg"
)
AM_COPATH_FILE = Path(__file__).resolve().parent.parent / "examples" / "am_equilibrium.pmg"


def am_pair_with_two_children() -> pm.Model:
    """Mated pair (m, f) at AM equilibrium, plus two full-sib children (o1, o2)."""
    m = pm.Model("AM equilibrium: pair + two full sibs", units=pm.Units.unstandardized())
    for v in ("V_A_eq", "V_E", "V_K", "V_A0"):
        m.declare(v, positive=True)

    for i in ("m", "f", "o1", "o2"):
        m.add_var(f"g_{i}", latent=True, label=rf"$g_{{{i}}}$")
        m.add_var(f"e_{i}", latent=True, label=rf"$e_{{{i}}}$")
        m.add_var(f"y_{i}", label=rf"$y_{{{i}}}$")
        m.add_path(f"g_{i}", f"y_{i}", 1)
        m.add_path(f"e_{i}", f"y_{i}", 1)
        m.add_variance(f"e_{i}", "V_E")

    # parents are exogenous at the equilibrium scale; assortment correlates their g
    for i in ("m", "f"):
        m.add_variance(f"g_{i}", "V_A_eq")
    m.add_cov("g_m", "g_f", "rho_g * V_A_eq")

    # each parent's environment is correlated with the OTHER parent's genes
    m.add_cov("e_m", "g_f", "rho_g * V_E")
    m.add_cov("e_f", "g_m", "rho_g * V_E")

    # transmission: g_o = (g_m + g_f)/2 + s_o,  Var[s_o] = V_K
    for o in ("o1", "o2"):
        m.add_var(f"s_{o}", latent=True, label=rf"$s_{{{o}}}$")
        m.add_path("g_m", f"g_{o}", sp.Rational(1, 2))
        m.add_path("g_f", f"g_{o}", sp.Rational(1, 2))
        m.add_path(f"s_{o}", f"g_{o}", 1)
        m.add_variance(f"s_{o}", "V_K")

    # the coupled equilibrium fixed point -- assumptions, to be solved, never traced
    m.assume("rho_g", "rho_y * h2_eq")
    m.assume("h2_eq", "V_A_eq / (V_A_eq + V_E)")
    m.assume("V_A_eq", "V_A0 / (1 - rho_g)")
    m.assume("V_K", "V_A0 / 2")
    return m


def test_text_front_end_produces_the_identical_model():
    """The hardest equivalence check available: the full AM model, both ways.

    ``examples/am_equilibrium_handwritten.pmg`` is this same model in the text grammar. If the
    two front-ends ever drift, this is where it shows up.
    """
    from_text = pm.from_text(AM_TEXT_FILE.read_text(), name="AM equilibrium: pair + two full sibs")
    assert canonical(from_text) == canonical(am_pair_with_two_children())
    # and it survives a round trip through to_text
    assert canonical(pm.from_text(from_text.to_text())) == canonical(from_text)


def test_the_copath_encoding_round_trips_too():
    """The live example uses a co-path; the text layer must carry that faithfully."""
    model = pm.from_text(AM_COPATH_FILE.read_text())
    assert model.has_copaths
    assert canonical(pm.from_text(model.to_text())) == canonical(model)


def test_am_model_is_structurally_sane():
    m = am_pair_with_two_children()
    assert m.observed == ("y_m", "y_f", "y_o1", "y_o2")
    assert m.is_recursive
    assert m.validate() == []
    # g_o is endogenous: its variance is implied, never stated
    for o in ("o1", "o2"):
        assert m.cov_value(f"g_{o}", f"g_{o}") is None
        assert set(m.parents(f"g_{o}")) == {"g_m", "g_f", f"s_{o}"}


def test_collateral_result_full_siblings():
    """eq:am-level3-col at d=1: Cov[y_o1, y_o2] = V_A_eq (1 + rho_g)/2."""
    m = am_pair_with_two_children()
    V_A_eq, rho_g = m.sym("V_A_eq"), m.sym("rho_g")
    Sigma, idx = ram_sigma(m)
    got = Sigma[idx["y_o1"], idx["y_o2"]]
    assert sp.simplify(got - V_A_eq * (1 + rho_g) / 2) == 0


def test_lineal_result_parent_offspring_needs_the_fixed_point_relation():
    """eq:am-level3-lin at d=1: Cov[y_m, y_o] = V_A_eq (1 + rho_y)/2.

    Note rho_y, not rho_g -- the asymmetry the writeup calls out. It only appears once
    rho_g = rho_y h2_eq is substituted, which is why side relations are first-class.
    """
    m = am_pair_with_two_children()
    V_A_eq, V_E, rho_g, rho_y = (m.sym(s) for s in ("V_A_eq", "V_E", "rho_g", "rho_y"))
    Sigma, idx = ram_sigma(m)
    got = Sigma[idx["y_m"], idx["y_o1"]]

    # before substitution: correct, but not in the writeup's form
    assert sp.simplify(got - (V_A_eq * (1 + rho_g) / 2 + rho_g * V_E / 2)) == 0

    # after substituting the fixed point, it is exactly the boxed result
    fixed_point = {rho_g: rho_y * V_A_eq / (V_A_eq + V_E)}
    assert sp.simplify(got.subs(fixed_point) - V_A_eq * (1 + rho_y) / 2) == 0


def test_lineal_exceeds_collateral_by_the_stated_amount():
    """The writeup's V_A_eq (rho_y - rho_g)/2 excess, purely environmental in origin."""
    m = am_pair_with_two_children()
    V_A_eq, rho_g, rho_y = (m.sym(s) for s in ("V_A_eq", "rho_g", "rho_y"))
    Sigma, idx = ram_sigma(m)
    fixed_point = {rho_g: rho_y * V_A_eq / (V_A_eq + m.sym("V_E"))}
    lineal = Sigma[idx["y_m"], idx["y_o1"]].subs(fixed_point)
    collateral = Sigma[idx["y_o1"], idx["y_o2"]].subs(fixed_point)
    excess = sp.simplify(lineal - collateral)
    assert sp.simplify(excess - V_A_eq * (rho_y - rho_g.subs(fixed_point)) / 2) == 0


def test_variance_recursion():
    """eq:am-recursion: V_A(t+1) = V_A0/2 + V_A(t)(1 + rho_g)/2, plus V_E."""
    m = am_pair_with_two_children()
    V_A_eq, V_E, V_K, V_A0, rho_g = (
        m.sym(s) for s in ("V_A_eq", "V_E", "V_K", "V_A0", "rho_g")
    )
    Sigma, idx = ram_sigma(m)
    got = Sigma[idx["y_o1"], idx["y_o1"]].subs({V_K: V_A0 / 2})  # eq:am-VK
    expected = V_A0 / 2 + V_A_eq * (1 + rho_g) / 2 + V_E
    assert sp.simplify(got - expected) == 0

    # and the child's phenotypic variance equals the parents' iff V_A_eq = V_A0/(1 - rho_g)
    at_equilibrium = sp.solve(sp.Eq(got - V_E, V_A_eq), V_A_eq)
    assert sp.simplify(at_equilibrium[0] - V_A0 / (1 - rho_g)) == 0
