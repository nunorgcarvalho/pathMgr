"""Tests for the text front-end: parsing, error reporting, and builder/text equivalence.

The load-bearing tests are the two equivalence ones -- ``from_text`` must produce exactly
what the builder produces, and ``to_text`` must round-trip -- because that is the whole
claim of the text layer: a thin front-end over the builder that cannot diverge from it.
"""

import pytest
import sympy as sp

import pathmgr as pm

from conftest import canonical
from test_validation_models import bivariate_regression, relative_covariance_section1


# ======================================================================================
# equivalence with the builder
# ======================================================================================
BIVARIATE_TEXT = """
# bivariate regression
positive: V_1, V_2, V_r

y ~ b1*x1 + b2*x2

x1 ~~ V_1*x1
x2 ~~ V_2*x2
y  ~~ V_r*y
x1 ~~ c12*x2
"""

RELCOV_TEXT = """
# relative covariance, Section 1 (random mating, independent environment)
units: unstandardized
latent: g_i, e_i, g_j, e_j
positive: V_A, V_E
label: g_i = $g_i$
label: e_i = $e_i$
label: y_i = $y_i$
label: g_j = $g_j$
label: e_j = $e_j$
label: y_j = $y_j$

y_i ~ g_i + e_i
y_j ~ g_j + e_j

g_i ~~ V_A*g_i
e_i ~~ V_E*e_i
g_j ~~ V_A*g_j
e_j ~~ V_E*e_j
g_i ~~ (V_A*pi_ij)*g_j

assume: V_A + V_E = 1
"""


def test_text_and_builder_agree_on_bivariate_regression():
    from_builder = bivariate_regression()
    from_text = pm.from_text(BIVARIATE_TEXT, name=from_builder.name)
    assert canonical(from_text) == canonical(from_builder)


def test_text_and_builder_agree_on_relative_covariance_section1():
    from_builder = relative_covariance_section1()
    from_text = pm.from_text(RELCOV_TEXT, name=from_builder.name)
    # the builder version declares variables in per-individual order; text follows the
    # equations, so compare order-insensitively as well as on content
    assert canonical(from_text) == canonical(from_builder)


@pytest.mark.parametrize("text", [BIVARIATE_TEXT, RELCOV_TEXT])
def test_to_text_round_trips(text):
    once = pm.from_text(text, name="m")
    twice = pm.from_text(once.to_text(), name="m")
    assert canonical(twice) == canonical(once)


def test_round_trip_preserves_rationals_and_expressions():
    m = pm.from_text(
        """
        latent: g_m, g_p, g_o, s_o
        g_o ~ 1/2*g_m + 1/2*g_p + s_o
        g_c ~ ((1 + rho_g)/2)*g_o
        g_m ~~ (rho_g*V_A_eq)*g_p
        """
    )
    again = pm.from_text(m.to_text())
    assert canonical(again) == canonical(m)
    assert m.path_coeff("g_m", "g_o") == sp.Rational(1, 2)
    assert sp.simplify(
        m.path_coeff("g_o", "g_c") - (1 + m.sym("rho_g")) / 2
    ) == 0


# ======================================================================================
# grammar
# ======================================================================================
def test_directives_may_appear_after_the_equations():
    m = pm.from_text(
        """
        y_i ~ g_i + e_i
        latent: g_i, e_i
        units: standardized to gen 0
        """
    )
    assert set(m.latent) == {"g_i", "e_i"}
    assert m.observed == ("y_i",)
    assert m.units == pm.Units.standardized("gen 0")


def test_implied_coefficient_is_one_and_variables_are_created_on_first_use():
    m = pm.from_text("y ~ g + e")
    assert m.names == ("y", "g", "e")
    assert m.path_coeff("g", "y") == 1
    assert m.latent == ()


def test_directed_direction_is_dst_on_the_left():
    m = pm.from_text("y ~ b*x")
    assert m.path_coeff("x", "y") == m.sym("b")
    assert m.path_coeff("y", "x") is None
    assert m.endogenous == ("y",)


def test_variance_and_covariance_forms():
    m = pm.from_text(
        """
        x1 ~~ V_1*x1
        x1 ~~ c12*x2
        g_i ~~ (V_A*pi_ij)*g_j
        """
    )
    assert m.cov_value("x1", "x1") == m.sym("V_1")
    assert m.cov_value("x1", "x2") == m.sym("c12")
    assert m.cov_value("g_i", "g_j") == m.sym("V_A") * m.sym("pi_ij")


def test_negative_and_multi_term_coefficients():
    m = pm.from_text("y ~ -b1*x1 + b2*x2 - x3")
    assert m.path_coeff("x1", "y") == -m.sym("b1")
    assert m.path_coeff("x2", "y") == m.sym("b2")
    assert m.path_coeff("x3", "y") == -1


def test_parenthesised_coefficient_with_internal_plus_is_one_term():
    m = pm.from_text("g_c ~ ((1 + rho_g)/2)*g_p + s_c")
    assert sp.simplify(m.path_coeff("g_p", "g_c") - (1 + m.sym("rho_g")) / 2) == 0
    assert m.path_coeff("s_c", "g_c") == 1


def test_symbol_assumptions_from_directives():
    m = pm.from_text(
        """
        positive: V_A
        g ~~ V_A*g
        """
    )
    assert m.sym("V_A").is_positive


def test_comments_and_blank_lines_ignored():
    m = pm.from_text(
        """
        # a comment

        y ~ b*x    # trailing comment
        """
    )
    assert m.names == ("y", "x")


def test_labels_are_attached_to_variables():
    m = pm.from_text(
        """
        latent: g_i
        label: g_i = $g_i$
        y_i ~ g_i
        """
    )
    assert m.var("g_i").display() == "$g_i$"


def test_observed_directive_declares_isolated_nodes():
    m = pm.from_text(
        """
        observed: z
        y ~ b*x
        """
    )
    assert "z" in m.names and m.var("z").observed
    assert m.parents("z") == () and m.children("z") == ()


def test_units_accepts_british_spelling_and_quoted_reference():
    assert pm.from_text("units: unstandardised\ny ~ x").units.is_standardized is False
    m = pm.from_text('units: standardised to "base generation (gen 0)"\ny ~ x')
    assert m.units.reference == "base generation (gen 0)"


def test_assume_lines_become_side_relations():
    m = pm.from_text(
        """
        g ~~ V_A*g
        e ~~ V_E*e
        assume: V_A + V_E = 1
        assume: rho_g = rho_y*h2_eq
        """
    )
    assert len(m.assumptions) == 2
    assert m.substitutions() == {m.sym("rho_g"): m.sym("rho_y") * m.sym("h2_eq")}


# ======================================================================================
# errors -- every one reports the line number and the offending line
# ======================================================================================
def test_bare_variance_shorthand_is_rejected_not_silently_conflated():
    """'g ~~ V_A' would make V_A a variable. The clash backstop must catch it."""
    with pytest.raises(ValueError, match="both as a variable and as a coefficient symbol"):
        pm.from_text(
            """
            g ~~ V_A
            y ~ V_A*x
            """
        )


def test_term_not_ending_in_a_variable_is_rejected():
    with pytest.raises(pm.TextSyntaxError, match="must end in a variable name"):
        pm.from_text("g ~~ (V_A*pi_ij)")


def test_standardized_without_a_reference_is_rejected():
    with pytest.raises(pm.TextSyntaxError, match="name its reference population"):
        pm.from_text("units: standardized\ny ~ x")


def test_unknown_units_is_rejected():
    with pytest.raises(pm.TextSyntaxError, match="unknown units"):
        pm.from_text("units: centered\ny ~ x")


def test_line_without_an_operator_is_rejected():
    with pytest.raises(pm.TextSyntaxError, match="not a directive and not an equation"):
        pm.from_text("y ~ b*x\nthis is not a model")


def test_error_carries_the_line_number_and_the_line():
    with pytest.raises(pm.TextSyntaxError) as exc:
        pm.from_text("y ~ b*x\n\ng ~~ (V_A)")
    assert exc.value.lineno == 3
    assert "g ~~ (V_A)" in str(exc.value)


def test_unbalanced_parentheses_rejected():
    with pytest.raises(pm.TextSyntaxError, match="unbalanced parentheses"):
        pm.from_text("y ~ ((1 + r)/2*x")


def test_multi_variable_left_hand_side_rejected():
    with pytest.raises(pm.TextSyntaxError, match="must be a single variable name"):
        pm.from_text("y + z ~ b*x")


def test_duplicate_edge_reports_its_line():
    with pytest.raises(pm.TextSyntaxError, match="already specified"):
        pm.from_text("y ~ b1*x\ny ~ b2*x")


def test_empty_side_rejected():
    with pytest.raises(pm.TextSyntaxError, match="needs both a left and right side"):
        pm.from_text("y ~")


def test_malformed_assume_and_label_rejected():
    with pytest.raises(pm.TextSyntaxError, match="assume needs"):
        pm.from_text("assume: V_A + V_E\ny ~ x")
    with pytest.raises(pm.TextSyntaxError, match="label needs"):
        pm.from_text("label: g_i\ny ~ x")
