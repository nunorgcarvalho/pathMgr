"""Unit tests for the specification API: variables, edges, symbols, units, RAM matrices."""

import pytest
import sympy as sp

import pathmgr as pm


# -- variables ------------------------------------------------------------------------
def test_latent_observed_split():
    m = pm.Model()
    m.add_vars("x", "y")
    m.add_var("f", latent=True)
    assert m.observed == ("x", "y")
    assert m.latent == ("f",)
    assert m.var("f").latent and not m.var("f").observed
    assert m.var("x").observed


def test_duplicate_variable_rejected():
    m = pm.Model().add_var("x")
    with pytest.raises(ValueError, match="already in model"):
        m.add_var("x")


def test_edge_on_unknown_variable_rejected():
    m = pm.Model().add_var("x")
    with pytest.raises(KeyError, match="unknown variable 'y'"):
        m.add_path("x", "y")
    with pytest.raises(KeyError):
        m.add_cov("x", "y", 1)


def test_variable_label_defaults_to_name():
    m = pm.Model().add_var("g_i", latent=True, label=r"$g_i$")
    m.add_var("plain")
    assert m.var("g_i").display() == r"$g_i$"
    assert m.var("plain").display() == "plain"


# -- edges ----------------------------------------------------------------------------
def test_directed_self_loop_rejected_pointing_at_add_cov():
    m = pm.Model().add_var("x")
    with pytest.raises(ValueError, match="bidirected edge"):
        m.add_path("x", "x")


def test_duplicate_edges_rejected():
    m = pm.Model().add_vars("x", "y")
    m.add_path("x", "y", "b")
    with pytest.raises(ValueError, match="already specified"):
        m.add_path("x", "y", "c")
    m.add_cov("x", "y", "c")
    with pytest.raises(ValueError, match="already specified"):
        m.add_cov("y", "x", "d")


def test_bidirected_edges_are_order_independent():
    m = pm.Model().add_vars("x", "y")
    m.add_cov("y", "x", "c")
    assert m.cov_value("x", "y") == m.cov_value("y", "x") == m.sym("c")


def test_path_coefficient_defaults_to_one():
    m = pm.Model().add_vars("g", "y")
    m.add_path("g", "y")
    assert m.path_coeff("g", "y") == 1


def test_coefficients_accept_numbers_symbols_and_expressions():
    m = pm.Model().add_vars("g_m", "g_p", "g_o")
    m.add_path("g_m", "g_o", sp.Rational(1, 2))
    m.add_path("g_p", "g_o", "1/2")
    m.add_cov("g_m", "g_p", "rho_g * V_A")
    assert m.path_coeff("g_m", "g_o") == m.path_coeff("g_p", "g_o") == sp.Rational(1, 2)
    assert m.cov_value("g_m", "g_p") == m.sym("rho_g") * m.sym("V_A")


# -- symbols --------------------------------------------------------------------------
def test_symbol_names_are_never_captured_by_sympy_builtins():
    """pi (relatedness), E (environment), beta (effects), I, S, N all collide with sympy."""
    m = pm.Model().add_vars("a", "b")
    m.add_cov("a", "b", "pi * E * beta * I * S * N")
    free = {s.name for s in m.cov_value("a", "b").free_symbols}
    assert free == {"pi", "E", "beta", "I", "S", "N"}
    assert m.cov_value("a", "b").has(sp.pi) is False


def test_one_name_is_always_one_symbol():
    """One name -> one set of assumptions, so expressions built from both cancel.

    Deliberately NOT asserting object identity (``is``). sympy's Symbol constructor is
    LRU-cached at size 1000, so once enough symbols exist in a process, an equal symbol comes
    back as a distinct object -- identity is an artifact of that cache, not an invariant
    anything should rely on. What matters, and what the registry actually guarantees, is that
    the assumptions agree, because two same-named symbols with *differing* assumptions are
    unequal in sympy and will not cancel.
    """
    m = pm.Model().add_vars("x", "y", "z")
    m.add_path("x", "y", "b")
    m.add_path("x", "z", "b")
    first, second = m.path_coeff("x", "y"), m.path_coeff("x", "z")
    assert first == second
    assert first.assumptions0 == second.assumptions0
    assert sp.simplify(first - second) == 0
    assert sp.simplify(first / second - 1) == 0


def test_declared_assumptions_are_honoured():
    m = pm.Model()
    m.declare("V_A", positive=True)
    m.add_vars("g")
    m.add_variance("g", "V_A")
    assert m.sym("V_A").is_positive
    assert sp.sqrt(m.sym("V_A") ** 2) == m.sym("V_A")


def test_redeclaring_a_symbol_differently_is_an_error():
    m = pm.Model()
    m.declare("V_A", positive=True)
    with pytest.raises(ValueError, match="already exists with assumptions"):
        m.declare("V_A", negative=True)


def test_prebuilt_sympy_symbols_are_canonicalised_to_the_registry():
    """A bare Symbol handed in must pick up the registry's assumptions.

    This is the case that genuinely matters: ``Symbol('b')`` and ``Symbol('b', positive=True)``
    are *unequal* in sympy, so without canonicalisation the two edges would carry
    non-cancelling coefficients.
    """
    m = pm.Model().add_vars("x", "y", "z")
    m.declare("b", positive=True)
    m.add_path("x", "y", "b")
    m.add_path("x", "z", sp.Symbol("b"))  # bare, no assumptions -- must be unified
    assert m.path_coeff("x", "z") == m.sym("b")
    assert m.path_coeff("x", "z").is_positive
    assert m.path_coeff("x", "z") == m.path_coeff("x", "y")
    assert sp.simplify(m.path_coeff("x", "z") - m.path_coeff("x", "y")) == 0


# -- units ----------------------------------------------------------------------------
def test_default_units_are_unstandardized():
    assert pm.Model().units == pm.Units.unstandardized()


def test_standardized_units_require_a_reference_population():
    with pytest.raises(ValueError, match="reference population"):
        pm.Units.standardized("")
    u = pm.Units.standardized("base generation (gen 0)")
    assert u.is_standardized
    assert "gen 0" in str(u)


def test_unstandardized_units_reject_a_reference():
    with pytest.raises(ValueError, match="no reference population"):
        pm.Units("unstandardized", reference="gen 0")


def test_units_are_carried_on_the_model():
    m = pm.Model("std", units=pm.Units.standardized("gen 0"))
    assert m.units.reference == "gen 0"
    assert "standardized to gen 0" in m.describe()


# -- side relations -------------------------------------------------------------------
def test_assumptions_and_substitutions():
    m = pm.Model()
    m.assume("V_A + V_E", 1)
    m.assume("rho_g", "rho_y * h2_eq")
    assert len(m.assumptions) == 2
    subs = m.substitutions()
    # only the Symbol = expr form becomes a substitution
    assert subs == {m.sym("rho_g"): m.sym("rho_y") * m.sym("h2_eq")}


# -- structure ------------------------------------------------------------------------
def test_parents_children_exogenous_endogenous():
    m = pm.Model().add_vars("g", "e", "y")
    m.add_path("g", "y")
    m.add_path("e", "y")
    assert set(m.parents("y")) == {"g", "e"}
    assert m.children("g") == ("y",)
    assert set(m.exogenous) == {"g", "e"}
    assert m.endogenous == ("y",)


def test_recursive_model_has_no_cycles():
    m = pm.Model().add_vars("x", "y", "z")
    m.add_path("x", "y", "a")
    m.add_path("y", "z", "b")
    assert m.is_recursive
    assert m.cycles() == []


def test_cycle_is_detected_and_reported():
    m = pm.Model().add_vars("x", "y")
    m.add_path("x", "y", "a")
    m.add_path("y", "x", "b")
    assert not m.is_recursive
    assert any("cycle" in str(i) for i in m.validate())


def test_validate_warns_about_exogenous_variable_with_no_variance():
    m = pm.Model().add_vars("x", "y")
    m.add_path("x", "y", "b")
    msgs = [str(i) for i in m.validate()]
    assert any("'x' has no variance" in s for s in msgs)
    m.add_variance("x", "V_x")
    assert not any("'x' has no variance" in str(i) for i in m.validate())


# -- RAM matrices ---------------------------------------------------------------------
def test_A_matrix_is_dst_by_src():
    m = pm.Model().add_vars("x", "y")
    m.add_path("x", "y", "b")
    A = m.A_matrix()
    assert A[1, 0] == m.sym("b")  # A[dst, src]
    assert A[0, 1] == 0


def test_S_matrix_is_symmetric():
    m = pm.Model().add_vars("x", "y")
    m.add_cov("x", "x", "V_x")
    m.add_cov("x", "y", "c")
    S = m.S_matrix()
    assert S == S.T
    assert S[0, 0] == m.sym("V_x")
    assert S[0, 1] == S[1, 0] == m.sym("c")


def test_F_matrix_selects_observed_rows_only():
    m = pm.Model()
    m.add_var("g", latent=True)
    m.add_var("y")
    F = m.F_matrix()
    assert F.shape == (1, 2)
    assert list(F) == [0, 1]


def test_ram_returns_matrices_in_insertion_order():
    m = pm.Model().add_vars("c", "a", "b")
    A, S, F, order = m.ram()
    assert order == ("c", "a", "b")
    assert A.shape == S.shape == (3, 3)


# -- copying / revision ---------------------------------------------------------------
def test_revision_increments_on_structural_change():
    m = pm.Model()
    r0 = m.revision
    m.add_var("x")
    assert m.revision > r0
    r1 = m.revision
    m.add_variance("x", "V_x")
    assert m.revision > r1


def test_copy_is_independent():
    m = pm.Model("orig").add_vars("x", "y")
    m.add_path("x", "y", "b")
    n = m.copy("branch")
    n.add_var("z")
    n.add_path("y", "z", "c")
    assert n.name == "branch" and m.name == "orig"
    assert not m.has_var("z")
    assert m.path_coeff("y", "z") is None
    assert n.path_coeff("x", "y") == m.path_coeff("x", "y")
