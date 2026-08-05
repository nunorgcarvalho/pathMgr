"""Tests for the Wright tracer: enumeration, the decomposition, limits, and output.

Agreement with the RAM engine lives in ``test_agreement.py`` and is driven by the battery.
What is checked here is that the *enumeration itself* is right on cases small enough to verify
by hand -- because agreement alone would not catch two engines that are wrong in the same way,
and because the itemized chains, not the total, are this module's product.
"""

import pytest
import sympy as sp

import pathmgr as pm

from battery import (
    diamond,
    mediation_chain,
    confounded_pair,
    siblings_sharing_a_genetic_factor,
    turning_point_with_ancestors,
)


def paths(decomposition) -> set[str]:
    return {c.path_string() for c in decomposition}


# ======================================================================================
# hand-checkable enumeration
# ======================================================================================
def test_mediation_chain_separates_direct_and_indirect_routes():
    m = mediation_chain()
    d = pm.WrightTracer(m).trace("x", "y")
    a, b, c, V_x = (m.sym(s) for s in ("a", "b", "c", "V_x"))

    assert paths(d) == {"x <-> x -> y", "x <-> x -> m -> y"}
    by_path = {c_.path_string(): c_.contribution for c_ in d}
    assert sp.simplify(by_path["x <-> x -> y"] - V_x * c) == 0            # direct
    assert sp.simplify(by_path["x <-> x -> m -> y"] - V_x * a * b) == 0   # indirect
    assert sp.simplify(d.total - V_x * (a * b + c)) == 0


def test_diamond_finds_both_routes():
    m = diamond()
    d = pm.WrightTracer(m).trace("x", "y")
    assert paths(d) == {"x <-> x -> a -> y", "x <-> x -> b -> y"}
    p1, p2, q1, q2, V_x = (m.sym(s) for s in ("p1", "p2", "q1", "q2", "V_x"))
    assert sp.simplify(d.total - V_x * (p1 * q1 + p2 * q2)) == 0


def test_confounded_pair_traces_through_the_bidirected_edge_both_ways():
    m = confounded_pair()
    tracer = pm.WrightTracer(m)

    d = tracer.trace("x1", "y")
    assert paths(d) == {"x1 <-> x1 -> y", "x1 <-> x2 -> y"}
    b1, b2, V_1, c12 = (m.sym(s) for s in ("b1", "b2", "V_1", "c12"))
    assert sp.simplify(d.total - (b1 * V_1 + b2 * c12)) == 0

    # the bidirected edge is symmetric, so it is traversed in both orientations
    dy = tracer.trace("y", "y")
    assert "y <- x1 <-> x2 -> y" in paths(dy)
    assert "y <- x2 <-> x1 -> y" in paths(dy)


def test_siblings_sharing_a_genetic_factor():
    """Every route between sibs must go up through a shared parent and back down."""
    m = siblings_sharing_a_genetic_factor()
    d = pm.WrightTracer(m).trace("y_1", "y_2")
    V_A = m.sym("V_A")

    assert paths(d) == {
        "y_1 <- g_1 <- g_m <-> g_m -> g_2 -> y_2",  # up through the maternal and back down
        "y_1 <- g_1 <- g_p <-> g_p -> g_2 -> y_2",  # and through the paternal
    }
    # each shared parent contributes (1/2)(V_A)(1/2)
    for chain in d:
        assert sp.simplify(chain.contribution - V_A / 4) == 0
        assert chain.is_variance_pivot
    assert sp.simplify(d.total - V_A / 2) == 0  # the familiar sib covariance


def test_no_chain_between_disconnected_variables():
    m = pm.from_text("y ~ b*x\nx ~~ V_x*x\nz ~~ V_z*z")
    d = pm.WrightTracer(m).trace("x", "z")
    assert len(d) == 0
    assert d.total == 0
    assert "no chains" in str(d)


def test_a_variance_is_the_same_enumeration_with_both_endpoints_equal():
    m = mediation_chain()
    tracer = pm.WrightTracer(m)
    a, b, c, V_x, V_m, V_y = (m.sym(s) for s in ("a", "b", "c", "V_x", "V_m", "V_y"))
    expected = b**2 * (a**2 * V_x + V_m) + c**2 * V_x + 2 * a * b * c * V_x + V_y
    assert sp.simplify(tracer.var("y") - expected) == 0
    assert tracer.trace("y", "y").x == tracer.trace("y", "y").y == "y"


# ======================================================================================
# the classical "no variable twice" rule, and why it is not enforced
# ======================================================================================
def test_a_node_may_appear_in_both_legs_and_must_for_correctness():
    """Cov[x, y] = q r Var[w]; Var[w]'s ancestral part rides on w-revisiting chains."""
    m = turning_point_with_ancestors()
    tracer = pm.WrightTracer(m)
    d = tracer.trace("x", "y")
    q, r, p_b, p_c = (m.sym(s) for s in ("q", "r", "p_b", "p_c"))
    V_b, V_c, V_w, C_bc = (m.sym(s) for s in ("V_b", "V_c", "V_w", "C_bc"))

    revisiting = [c for c in d if c.revisits]
    assert revisiting, "the w-revisiting chains must be enumerated"
    assert all(c.revisits == ("w",) for c in revisiting)

    # the total is exactly q*r*Var[w], including the ancestral part
    var_w = V_w + p_b**2 * V_b + p_c**2 * V_c + 2 * p_b * p_c * C_bc
    assert sp.simplify(d.total - q * r * var_w) == 0

    # dropping the revisiting chains would lose precisely the ancestral part
    without = sp.expand(sum((c.contribution for c in d if not c.revisits), sp.Integer(0)))
    assert sp.simplify(without - q * r * V_w) == 0
    assert sp.simplify(d.total - without - q * r * (var_w - V_w)) == 0


def test_a_node_never_repeats_within_a_single_leg():
    """Within a leg, repeats are impossible in a DAG -- assert it rather than assume it."""
    m = turning_point_with_ancestors()
    for chain in pm.WrightTracer(m).trace("x", "y"):
        assert len(set(chain.backward)) == len(chain.backward)
        assert len(set(chain.forward)) == len(chain.forward)


def test_exactly_one_bidirected_edge_per_chain():
    m = siblings_sharing_a_genetic_factor()
    for chain in pm.WrightTracer(m).trace("y_1", "y_2"):
        assert chain.pivot[0] in m.names and chain.pivot[1] in m.names
        assert m.cov_value(*chain.pivot) is not None
        assert chain.bidirected_value == m.cov_value(*chain.pivot)


# ======================================================================================
# Chain accessors (a diagram renderer will use these)
# ======================================================================================
def test_chain_structure_accessors():
    m = mediation_chain()
    chain = next(c for c in pm.WrightTracer(m).trace("x", "y") if c.length == 2)

    assert chain.x == "x" and chain.y == "y"
    assert chain.backward == ("x",)
    assert chain.forward == ("x", "m", "y")
    assert chain.pivot == ("x", "x")
    assert chain.is_variance_pivot
    assert chain.nodes == ("x", "m", "y")  # the pivot node appears once
    assert chain.length == 2


def test_directed_edges_are_reported_for_highlighting():
    """task-20260804-151349 stretch goal: highlight one traced chain on the diagram."""
    m = mediation_chain()
    d = pm.WrightTracer(m).trace("m", "y")
    chain = next(c for c in d if c.path_string() == "m <- x <-> x -> m -> y")

    assert chain.directed_edges() == (("x", "m"), ("x", "m"), ("m", "y"))
    # every reported edge really is in the model
    for src, dst in chain.directed_edges():
        assert m.path_coeff(src, dst) is not None


# ======================================================================================
# output formatting
# ======================================================================================
def test_readable_output_lists_chains_and_the_total():
    m = mediation_chain()
    text = str(pm.WrightTracer(m).trace("x", "y"))
    assert "Cov[x, y]" in text
    assert "2 chains" in text
    assert "unstandardized" in text
    assert "x <-> x -> m -> y" in text
    assert "total" in text
    # factors are shown before multiplying out, so the product is transcribable
    assert "V_x * a * b" in text or "V_x * b * a" in text


def test_revisits_are_flagged_in_the_readable_output():
    text = str(pm.WrightTracer(turning_point_with_ancestors()).trace("x", "y"))
    assert "revisits w" in text


def test_chains_are_listed_shortest_first():
    d = pm.WrightTracer(mediation_chain()).trace("x", "y")
    lengths = [c.length for c in d.sorted_chains()]
    assert lengths == sorted(lengths)


def test_latex_align_output():
    m = mediation_chain()
    tex = pm.WrightTracer(m).trace("x", "y").to_latex()
    assert tex.startswith("\\begin{align*}")
    assert tex.rstrip().endswith("\\end{align*}")
    assert "\\operatorname{Cov}" in tex
    assert "\\leftrightarrow" in tex and "\\rightarrow" in tex
    assert tex.count("\\\\") >= 2  # one row per chain


def test_latex_tabular_output():
    tex = pm.WrightTracer(mediation_chain()).trace("x", "y").to_latex(style="tabular")
    assert tex.startswith("\\begin{tabular}")
    assert "\\hline" in tex
    assert tex.rstrip().endswith("\\end{tabular}")


def test_latex_uses_variable_labels_when_present():
    """The labels set on a model should be what appears in the document."""
    m = pm.Model("labelled")
    m.add_var("g_i", latent=True, label=r"$g_i$")
    m.add_var("y_i", label=r"$y_i$")
    m.add_path("g_i", "y_i", 1)
    m.add_variance("g_i", "V_A")
    tex = pm.WrightTracer(m).trace("g_i", "y_i").to_latex()
    assert "g_i \\leftrightarrow g_i \\rightarrow y_i" in tex
    assert "$" not in tex.split("\\text{[")[1].split("]")[0]  # label's $ stripped


def test_latex_total_form_and_style_are_validated():
    d = pm.WrightTracer(mediation_chain()).trace("x", "y")
    assert "\\left" in d.to_latex(total_form="expanded")
    with pytest.raises(ValueError, match="style must be"):
        d.to_latex(style="markdown")
    with pytest.raises(ValueError, match="total_form must be"):
        d.to_latex(total_form="pretty")


def test_empty_decomposition_renders_as_zero():
    m = pm.from_text("y ~ b*x\nx ~~ V_x*x\nz ~~ V_z*z")
    tex = pm.WrightTracer(m).trace("x", "z").to_latex()
    assert "= 0" in tex


# ======================================================================================
# limits and errors
# ======================================================================================
def test_cyclic_model_gives_a_clear_error_pointing_at_the_ram_engine():
    m = pm.Model("feedback")
    m.add_vars("x", "y", "z")
    m.add_path("x", "y", "a")
    m.add_path("y", "z", "b")
    m.add_path("z", "y", "d")
    m.add_variance("x", "S_x")

    with pytest.raises(pm.UntraceableModelError) as exc:
        pm.WrightTracer(m).trace("x", "y")
    message = str(exc.value)
    assert "RAMEngine" in message
    assert "infinitely many" in message
    assert "y -> z" in message or "z -> y" in message

    # and the RAM engine really does handle it
    assert sp.simplify(
        pm.RAMEngine(m).cov("x", "y") - m.sym("a") * m.sym("S_x") / (1 - m.sym("b") * m.sym("d"))
    ) == 0


def test_chain_limit_raises_instead_of_hanging():
    """A wide diamond lattice has exponentially many paths; the cap must be loud."""
    m = pm.Model("lattice")
    m.add_var("v0")
    m.add_variance("v0", "S")
    previous = ["v0"]
    for layer in range(1, 7):
        current = []
        for k in range(3):
            name = f"v{layer}_{k}"
            m.add_var(name)
            for p in previous:
                m.add_path(p, name, f"a_{p}_{name}")
            current.append(name)
        previous = current
    tracer = pm.WrightTracer(m, max_chains=50)
    with pytest.raises(pm.ChainLimitError, match="RAMEngine"):
        tracer.trace("v0", previous[0])


def test_unknown_variable_is_a_clear_error():
    tracer = pm.WrightTracer(mediation_chain())
    with pytest.raises(KeyError, match="unknown variable 'nope'"):
        tracer.trace("nope", "y")


def test_zero_valued_bidirected_edge_contributes_no_chain():
    m = pm.Model()
    m.add_vars("x1", "x2", "y")
    m.add_path("x1", "y", "b1")
    m.add_path("x2", "y", "b2")
    m.add_variance("x1", "V_1")
    m.add_variance("x2", "V_2")
    m.add_cov("x1", "x2", 0)
    d = pm.WrightTracer(m).trace("x1", "x2")
    assert len(d) == 0


@pytest.mark.parametrize("form", ["raw", "expanded", "simplified", "factored"])
def test_cov_forms_agree(form):
    m = mediation_chain()
    tracer = pm.WrightTracer(m)
    assert sp.simplify(tracer.cov("x", "y", form=form) - tracer.cov("x", "y")) == 0


def test_bad_form_is_rejected():
    with pytest.raises(ValueError, match="unknown form"):
        pm.WrightTracer(mediation_chain()).cov("x", "y", form="pretty")
