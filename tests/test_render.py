"""Tests for diagram rendering: TikZ, raster, layout, and chain highlighting.

Two things here are more than cosmetic and are tested as such.

**The three edge types must be unmistakable.** A reader who takes a co-path for a covariance
arrow will apply the wrong tracing rules and get a wrong answer by hand, so the co-path is
asserted to differ from a bidirected edge on arrowheads *and* weight *and* colour, and to carry
no arrow tips at all.

**The TikZ must actually compile**, not merely look plausible -- verified by running pdflatex
when it is available, and skipped rather than faked when it is not.
"""

import shutil

import pytest
import sympy as sp

import pathmgr as pm
from pathmgr.render import (
    DiagramStyle,
    Layout,
    coefficient_label,
    layered_layout,
    pedigree_layout,
    to_image,
    to_standalone,
    to_tikz,
    write_pdf,
)
from pathmgr.render.tikz import highlight_sets

from battery import all_models

HAS_PDFLATEX = shutil.which("pdflatex") is not None
try:
    import matplotlib  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False


def copath_chain(model: pm.Model):
    """The chain between the two g's that crosses the co-path.

    The fixture deliberately carries BOTH a bidirected edge and a co-path between the partners,
    so `chains[0]` is whichever came first -- pick by what the chain actually crosses.
    """
    return next(c for c in pm.WrightTracer(model).trace("g_m", "g_p") if c.crosses_copaths)


def three_edge_model() -> pm.Model:
    """A model carrying all three edge types, a latent/observed mix, and a variance."""
    return pm.from_text(
        """
        latent: g_m, e_m, g_p, e_p
        positive: V_A, V_E
        label: g_m = $g_m$
        y_m ~ g_m + e_m
        y_p ~ g_p + e_p
        g_m ~~ V_A*g_m
        e_m ~~ V_E*e_m
        g_p ~~ V_A*g_p
        e_p ~~ V_E*e_p
        g_m ~~ c_gg*g_p
        y_m -- (rho_y/(V_A + V_E))*y_p
        """,
        name="three edge types",
    )


# ======================================================================================
# the conventions
# ======================================================================================
def test_latent_and_observed_get_different_shapes():
    tex = to_tikz(three_edge_model())
    assert "pmLatent/.style" in tex and "ellipse" in tex
    assert "pmObserved/.style" in tex and "rectangle" in tex
    assert "\\node[pmLatent] (g_m)" in tex
    assert "\\node[pmObserved] (y_m)" in tex


def test_the_three_edge_types_are_visually_distinct():
    """The load-bearing test: a co-path must not be confusable with a covariance arrow."""
    style = DiagramStyle()
    tex = to_tikz(three_edge_model(), style=style)

    # arrowheads: one, two, none
    assert f"pmDirected/.style={{{style.arrow_tip_directed}" in tex
    assert f"pmBidirected/.style={{{style.arrow_tip_bidirected}" in tex
    assert style.arrow_tip_directed.count("Stealth") == 1  # one head
    assert style.arrow_tip_bidirected.count("Stealth") == 2  # two heads
    copath_style = [line for line in tex.splitlines() if "pmCopath/.style" in line][0]
    assert "->" not in copath_style and "<-" not in copath_style, copath_style

    # weight: the co-path is thicker
    assert style.copath_width > style.bidirected_width
    assert style.copath_width > style.directed_width

    # colour: the co-path is drawn in its own, declared as a real colour not a raw hex
    assert "\\definecolor" in tex
    assert style.copath_colour.lstrip("#").upper() in tex

    # curvature: bidirected bends, the co-path is straight
    assert "to[bend left=30]" in tex
    assert "(y_p) -- " in tex or "(y_m) -- " in tex


def test_all_three_edge_kinds_are_actually_emitted():
    tex = to_tikz(three_edge_model())
    assert tex.count("pmDirected") >= 4 + 1  # 4 paths plus the style definition
    assert "pmBidirected" in tex
    assert "pmCopath" in tex


def test_variance_is_a_self_loop_and_can_be_suppressed():
    model = three_edge_model()
    assert "loop above" in to_tikz(model)
    assert "loop above" not in to_tikz(model, style=DiagramStyle(show_variances=False))


def test_several_copaths_on_one_couple_stay_separable():
    """Cross-trait assortment: two co-paths between the same pair must not overlap."""
    model = pm.Model()
    model.add_vars("S_m", "S_p")
    model.add_variance("S_m", "V")
    model.add_variance("S_p", "V")
    model.add_copath("S_m", "S_p", "mu", process="couple")
    model.add_copath("S_m", "S_p", "mu_prime", process="other")
    tex = to_tikz(model)
    copath_lines = [line for line in tex.splitlines() if "pmCopath," in line]
    assert len(copath_lines) == 2
    assert sum("bend right" in line for line in copath_lines) == 1  # the second is bowed


# ======================================================================================
# labels
# ======================================================================================
def test_coefficients_render_as_latex_math_not_raw_names():
    assert coefficient_label(sp.Symbol("rho_y")) == "\\rho_{y}"
    assert coefficient_label(sp.Rational(1, 2)) == "\\frac{1}{2}"
    assert coefficient_label(sp.Integer(1)) == ""  # unit coefficients are omitted
    assert coefficient_label(sp.Integer(1), omit_unit=False) == "1"


def test_rho_y_appears_as_a_greek_letter_in_the_output():
    tex = to_tikz(three_edge_model())
    assert "\\rho_{y}" in tex
    assert "rho_y" not in tex.replace("\\rho_{y}", "")


def test_edge_and_node_labels_can_be_overridden():
    model = three_edge_model()
    style = DiagramStyle(
        label_overrides={("g_m", "y_m"): r"\alpha"},
        node_label_overrides={"y_m": r"P_{\text{m}}"},
    )
    tex = to_tikz(model, style=style)
    assert "\\alpha" in tex
    assert "P_{\\text{m}}" in tex


def test_variable_labels_are_used_and_their_dollars_stripped():
    tex = to_tikz(three_edge_model())
    assert "{$g_m$}" in tex  # from `label: g_m = $g_m$`, not double-wrapped
    assert "$$" not in tex


def test_unit_coefficients_can_be_shown():
    tex = to_tikz(three_edge_model(), style=DiagramStyle(show_unit_coefficients=True))
    assert "{$1$}" in tex


# ======================================================================================
# layout
# ======================================================================================
def test_explicit_coordinates_are_honoured_exactly():
    model = three_edge_model()
    layout = Layout({"g_m": (3.0, 4.0)})
    tex = to_tikz(model, layout=layout)
    assert "(g_m) at (3.000,4.000)" in tex


def test_partial_layout_is_completed_automatically():
    model = three_edge_model()
    completed = Layout({"g_m": (9.0, 9.0)}).completed(model)
    assert completed["g_m"] == (9.0, 9.0)
    assert set(completed.positions) == set(model.names)


def test_layered_layout_puts_parents_above_children():
    model = three_edge_model()
    layout = layered_layout(model)
    assert layout["g_m"][1] > layout["y_m"][1]
    assert layout["g_p"][1] > layout["y_p"][1]


def test_layered_layout_handles_a_cyclic_model():
    """A feedback model has no well-defined depth, but is still worth drawing."""
    model = pm.Model()
    model.add_vars("x", "y", "z")
    model.add_path("x", "y", "a")
    model.add_path("y", "z", "b")
    model.add_path("z", "y", "d")
    model.add_variance("x", "S")
    layout = layered_layout(model)
    assert set(layout.positions) == {"x", "y", "z"}
    assert "\\node" in to_tikz(model)


def test_pedigree_layout_places_generations_as_rows():
    layout = pedigree_layout({"y_m": 0, "y_p": 0, "y_o": 1})
    assert layout["y_m"][1] == layout["y_p"][1]
    assert layout["y_o"][1] < layout["y_m"][1]
    assert layout["y_m"][0] != layout["y_p"][0]


@pytest.mark.parametrize("name", sorted(all_models()))
def test_every_battery_model_renders_with_auto_layout(name):
    """"Legible and correct" is the bar for auto-layout; at minimum it must never fail."""
    model = all_models()[name]
    tex = to_tikz(model)
    assert tex.startswith("\\definecolor") or tex.startswith("\\begin{tikzpicture}")
    assert tex.rstrip().endswith("\\end{tikzpicture}")
    for variable in model.variables:
        assert f"({variable.name.replace('.', '-')})" in tex
    # no node placed on top of another
    positions = layered_layout(model).positions
    assert len(set(positions.values())) == len(positions), f"{name}: overlapping nodes"


def test_node_ids_are_tikz_safe():
    model = pm.Model()
    model.add_var("a.b:c")
    model.add_var("plain")
    model.add_path("a.b:c", "plain", "q")
    tex = to_tikz(model)
    assert "(a-b-c)" in tex
    assert "(a.b:c)" not in tex


# ======================================================================================
# highlighting a traced chain
# ======================================================================================
def test_highlight_sets_come_from_the_chain():
    model = three_edge_model()
    chain = copath_chain(model)
    directed, bidirected, copaths = highlight_sets(chain)
    assert directed
    assert bidirected
    assert copaths == {frozenset({"y_m", "y_p"})}


def test_highlighting_emphasises_the_chain_and_fades_the_rest():
    model = three_edge_model()
    chain = copath_chain(model)
    style = DiagramStyle()
    tex = to_tikz(model, highlight=chain, style=style)

    highlight_hex = style.highlight_colour.lstrip("#").upper()
    faded_hex = style.faded_colour.lstrip("#").upper()
    assert highlight_hex in tex  # the chain's edges
    assert faded_hex in tex  # everything else
    # the co-path the chain crosses is drawn emphasised, at the boosted width
    boosted = f"{style.copath_width * style.highlight_scale:.2f}pt"
    assert boosted in tex


def test_highlight_caption_states_which_term_is_shown():
    model = three_edge_model()
    chain = copath_chain(model)
    tex = to_tikz(model, highlight=chain)
    assert "\\leftrightarrow" in tex
    assert "\\text{---}" in tex  # the co-path crossing, not a minus sign
    assert to_tikz(model, highlight=chain, caption_chain=False).count("\\text{---}") == 0


def test_copath_crossing_is_not_rendered_as_a_minus_sign():
    """`a - b` in math mode reads as subtraction; a co-path must not look like one."""
    model = three_edge_model()
    chain = copath_chain(model)
    assert "\\;\\text{---}\\;" in chain.tex_path()
    assert " - " not in chain.tex_path()


def test_rendering_never_needs_the_engines():
    """A model must be drawable without computing anything."""
    import pathmgr.render.tikz as tikz_module

    source = open(tikz_module.__file__).read()
    assert "RAMEngine" not in source
    assert "WrightTracer" not in source


# ======================================================================================
# standalone document and compilation
# ======================================================================================
def test_standalone_is_a_complete_document():
    source = to_standalone(three_edge_model())
    assert "\\documentclass" in source
    assert "\\usepackage{tikz}" in source
    assert "\\usetikzlibrary{shapes.geometric" in source
    assert "\\begin{document}" in source and "\\end{document}" in source
    assert "\\begin{tikzpicture}" in source


def test_standalone_never_uses_the_standalone_class_by_default():
    """standalone.cls is genuinely absent here and cannot be installed, so it is not the default.

    Note that `arrows.meta` IS available (it ships as `pgflibraryarrows.meta.code.tex`, so a
    `kpsewhich tikzlibraryarrows.meta.code.tex` probe reports a false negative) and the default
    style uses it -- see the library tests below.
    """
    source = to_standalone(three_edge_model())
    assert "\\documentclass{article}" in source
    assert "standalone" not in source


def test_only_the_libraries_the_style_needs_are_emitted():
    default_source = to_standalone(three_edge_model())
    assert "shapes.geometric" in default_source
    assert "arrows.meta" in default_source  # the default Stealth tips need it

    portable_source = to_standalone(three_edge_model(), style=DiagramStyle.portable())
    assert "shapes.geometric" in portable_source
    assert "arrows.meta" not in portable_source
    assert "Stealth" not in portable_source


def test_portable_style_uses_only_built_in_arrow_tips():
    style = DiagramStyle.portable()
    assert not style.needs_arrows_meta
    assert DiagramStyle().needs_arrows_meta
    tex = to_tikz(three_edge_model(), style=style)
    assert "pmDirected/.style={->," in tex
    assert "Stealth" not in tex


def test_standalone_page_is_sized_to_the_drawing():
    source = to_standalone(three_edge_model(), layout=Layout({"g_m": (0, 0), "y_p": (12, -9)}))
    assert "paperwidth=" in source and "paperheight=" in source


@pytest.mark.skipif(not HAS_PDFLATEX, reason="pdflatex not available")
def test_tikz_actually_compiles(tmp_path):
    out = write_pdf(three_edge_model(), tmp_path / "d.pdf")
    assert out.exists() and out.stat().st_size > 1000
    assert out.read_bytes().startswith(b"%PDF")


@pytest.mark.skipif(not HAS_PDFLATEX, reason="pdflatex not available")
def test_highlighted_diagram_compiles(tmp_path):
    model = three_edge_model()
    chain = copath_chain(model)
    out = write_pdf(model, tmp_path / "h.pdf", highlight=chain)
    assert out.exists() and out.stat().st_size > 1000


@pytest.mark.skipif(not HAS_PDFLATEX, reason="pdflatex not available")
@pytest.mark.parametrize(
    "name", ["co-path AM example", "half-sibling pedigree", "relative covariance S1"]
)
def test_representative_battery_models_compile(tmp_path, name):
    model = all_models()[name]
    out = write_pdf(model, tmp_path / f"{name.replace(' ', '_')}.pdf")
    assert out.exists() and out.stat().st_size > 1000


def test_missing_latex_gives_a_clear_error(tmp_path):
    from pathmgr.render.tikz import TikzCompileError

    with pytest.raises(TikzCompileError, match="not on PATH"):
        write_pdf(three_edge_model(), tmp_path / "x.pdf", engine="definitely-not-a-real-engine")


# ======================================================================================
# raster
# ======================================================================================
@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
@pytest.mark.parametrize("suffix", [".png", ".svg", ".pdf"])
def test_raster_export_writes_a_file(tmp_path, suffix):
    out = to_image(three_edge_model(), tmp_path / f"d{suffix}")
    assert out.exists() and out.stat().st_size > 500


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
def test_raster_honours_explicit_layout_and_highlight(tmp_path):
    model = three_edge_model()
    chain = copath_chain(model)
    layout = Layout({"g_m": (0, 0), "e_m": (1.6, 0), "y_m": (0.8, -1.8)})
    out = to_image(model, tmp_path / "h.png", layout=layout, highlight=chain, legend=True)
    assert out.exists()


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
def test_raster_arc_midpoint_lands_on_the_curve():
    """A curved edge's label must sit on its arc, not mirrored to the wrong side."""
    from pathmgr.render.raster import _arc_midpoint

    start, end, rad = (0.0, 0.0), (4.0, 0.0), 0.3
    x, y = _arc_midpoint(start, end, rad)
    assert abs(x - 2.0) < 1e-9  # halfway along
    # matplotlib's arc3 with positive rad bows to NEGATIVE y for a left-to-right chord
    assert y < 0
    assert abs(y - (-rad * 4.0 / 2.0)) < 1e-9


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
def test_every_battery_model_rasterises(tmp_path):
    for name, model in sorted(all_models().items()):
        if len(model.names) > 24:  # keep the suite quick; big pedigrees are covered by TikZ
            continue
        out = to_image(model, tmp_path / f"{abs(hash(name))}.png", dpi=60)
        assert out.exists(), name


def test_core_import_does_not_pull_in_matplotlib():
    """Computing must not require a drawing dependency."""
    import subprocess
    import sys

    code = (
        "import sys; import pathmgr; "
        "assert 'matplotlib' not in sys.modules, sorted(m for m in sys.modules "
        "if 'matplotlib' in m); print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout


# ======================================================================================
# task-20260804-205013: legibility pass, and the correctness rule behind item 1
# ======================================================================================
def test_a_style_flag_never_suppresses_a_highlighted_edge():
    """CORRECTNESS. `show_variances=False` declutters the CONTEXT; it must not hide the chain.

    In the allele-level chain the two `z <-> z` variances are the first and last edges and carry
    the `1/2 * 1/2` that produces the whole `/4` in the result. Hiding them makes the figure
    contradict its own caption and leaves a reader tracing it by hand off by a factor of four.
    """
    from pathmgr.genetics import allele_motif

    motif = allele_motif(n_variants=1)
    chain = pm.WrightTracer(motif.model).trace(
        motif.z("m", "mat", 0), motif.z("p", "mat", 0)
    ).chains[0]
    tidy = DiagramStyle(show_variances=False)

    without_highlight = to_tikz(motif.model, style=tidy)
    assert "loop above" not in without_highlight  # context variances still hidden

    with_highlight = to_tikz(motif.model, style=tidy, highlight=chain)
    loops = [line for line in with_highlight.splitlines() if "loop above" in line]
    assert len(loops) == 2, "the chain's own two z <-> z variances must be drawn"
    highlight_hex = tidy.highlight_colour.lstrip("#").upper()
    assert all(highlight_hex in line for line in loops), "and emphasised"
    # the 1/2 they contribute is visible
    assert all("\\frac{1}{2}" in line for line in loops)


def test_the_suppression_rule_is_stated_once_and_covers_every_filter():
    """Any style flag that can hide an edge must go through draws_variance-style gating.

    Enumerated deliberately: if a new filter is added, this test is where the rule gets applied
    again rather than the bug being rediscovered on a figure.
    """
    style = DiagramStyle(show_variances=False)
    assert style.draws_variance(highlighted=True) is True
    assert style.draws_variance(highlighted=False) is False
    assert DiagramStyle(show_variances=True).draws_variance(highlighted=False) is True
    # show_unit_coefficients only affects LABEL text, never whether an edge is drawn
    model = three_edge_model()
    with_units = to_tikz(model, style=DiagramStyle(show_unit_coefficients=True))
    without = to_tikz(model, style=DiagramStyle(show_unit_coefficients=False))
    assert with_units.count("\\draw") == without.count("\\draw")


def test_nodes_are_sized_by_their_contents_not_a_uniform_footprint():
    style = DiagramStyle()
    small = style.node_size("g", latent=False)
    large = style.node_size(r"z^{(m)}_{m,0}", latent=False)
    assert large[0] > small[0], "a longer label must get a wider box"
    assert small[0] >= style.node_min_width, "but not smaller than the floor"
    # an ellipse needs proportionally more room than a rectangle for the same text
    assert style.node_size("abc", latent=True)[0] > style.node_size("abc", latent=False)[0]
    # and it is all tunable from the style
    roomy = DiagramStyle(rectangle_inset=0.5)
    assert roomy.node_size("g", latent=False)[0] > small[0]


def test_tikz_nodes_use_inner_sep_rather_than_a_uniform_minimum():
    style = DiagramStyle()
    tex = to_tikz(three_edge_model(), style=style)
    assert f"inner sep={style.rectangle_inset}cm" in tex
    assert f"inner sep={style.ellipse_inset}cm" in tex
    # the minimums are a floor for a one-character label, not a target
    assert f"minimum width={style.node_min_width}cm" in tex
    assert style.node_min_width < 0.6


def test_edges_stop_at_the_node_boundary():
    """No arrow may be drawn to a node's centre and then covered by the node."""
    from pathmgr.render.placement import Rect, boundary_point

    rect = Rect(0.0, 0.0, 2.0, 1.0)
    right = boundary_point((0.0, 0.0), (10.0, 0.0), rect, clearance=0.0)
    assert abs(right[0] - 1.0) < 1e-9 and abs(right[1]) < 1e-9
    up = boundary_point((0.0, 0.0), (0.0, 10.0), rect, clearance=0.0)
    assert abs(up[1] - 0.5) < 1e-9
    # a wider node pushes the start further out -- the whole point of not using a constant
    wide = boundary_point((0.0, 0.0), (10.0, 0.0), Rect(0.0, 0.0, 4.0, 1.0), clearance=0.0)
    assert wide[0] > right[0]
    # clearance adds a visible gap on top
    assert boundary_point((0.0, 0.0), (10.0, 0.0), rect, clearance=0.3)[0] > right[0]
    # never overshoot the target
    near = boundary_point((0.0, 0.0), (0.2, 0.0), rect, clearance=0.0)
    assert near[0] <= 0.2 + 1e-9


def test_label_placement_is_deterministic():
    """Identical model in, identical TikZ out -- or figure diffs become unreadable."""
    from battery import half_sibling_pedigree

    model = half_sibling_pedigree()
    first = to_tikz(model)
    for _ in range(3):
        assert to_tikz(model) == first
    # and rebuilding the model from scratch gives the same bytes
    assert to_tikz(half_sibling_pedigree()) == first


def test_simple_diagrams_keep_the_plain_midpoint_label():
    """The midpoint is the first candidate, so an uncluttered diagram is unchanged."""
    model = pm.from_text("y ~ b*x\nx ~~ V_x*x")
    tex = to_tikz(model, layout=Layout({"x": (0, 0), "y": (0, -2)}))
    assert "node[pmLabel] {$b$}" in tex  # no pos= or shift, i.e. still the midpoint


def test_label_placement_moves_a_label_that_would_collide():
    """A label sitting on a node must be nudged off it."""
    from pathmgr.render.placement import labelled_edges, place_labels

    model = pm.Model()
    model.add_vars("a", "b", "middle")
    model.add_path("a", "b", "coefficient_with_a_long_name")
    model.add_variance("a", "V")
    # `middle` sits exactly where the a->b label would go
    layout = Layout({"a": (0, 0), "b": (4, 0), "middle": (2, 0)})
    style = DiagramStyle()
    placed = place_labels(model, layout, style, labelled_edges(model, style))
    placement = placed[("a", "b")]
    assert (placement.position, placement.offset) != (0.5, 0.0), "should have moved off the node"

    # and with avoidance off it stays at the midpoint
    plain = DiagramStyle(avoid_label_collisions=False)
    stubborn = place_labels(model, layout, plain, labelled_edges(model, plain))[("a", "b")]
    assert (stubborn.position, stubborn.offset) == (0.5, 0.0)


def test_both_back_ends_place_labels_identically():
    """They share the placement pass, so a figure looks the same whichever draws it."""
    from pathmgr.render.placement import labelled_edges, place_labels

    from battery import half_sibling_pedigree

    model = half_sibling_pedigree()
    layout = Layout().completed(model)
    style = DiagramStyle()
    once = place_labels(model, layout, style, labelled_edges(model, style))
    twice = place_labels(model, layout, style, labelled_edges(model, style))
    assert {k: (v.position, v.offset) for k, v in once.items()} == {
        k: (v.position, v.offset) for k, v in twice.items()
    }


def test_highlight_caption_shows_the_product_being_formed():
    """Item 5: the figure and the derivation are the same object."""
    from pathmgr.genetics import allele_motif

    motif = allele_motif(n_variants=1)
    chain = pm.WrightTracer(motif.model).trace(
        motif.z("m", "mat", 0), motif.z("p", "mat", 0)
    ).chains[0]

    product = chain.tex_factors()
    assert "\\cdot" in product
    assert "\\frac{1}{2}" in product  # the variances are IN the product
    assert "\\cdot 1 \\cdot" not in product  # unit coefficients dropped, as on the edges

    contribution = chain.tex_contribution()
    assert product in contribution
    assert contribution.count("=") == 1  # <product> = <value>

    tex = to_tikz(motif.model, highlight=chain, caption_name=r"\operatorname{Cov}")
    assert "\\cdot" in tex
    assert "\\operatorname{Cov}" in tex
    assert "align=center" in tex  # the two-line caption


def test_a_chain_of_only_unit_factors_still_shows_a_product():
    model = pm.from_text("y ~ g\ng ~~ g")
    chain = pm.WrightTracer(model).trace("g", "y").chains[0]
    assert chain.tex_factors() == "1"


@pytest.mark.skipif(not HAS_PDFLATEX, reason="pdflatex not available")
def test_the_highlighted_allele_figure_compiles_with_its_two_line_caption(tmp_path):
    from pathmgr.genetics import allele_motif

    motif = allele_motif(n_variants=1)
    chain = pm.WrightTracer(motif.model).trace(
        motif.z("m", "mat", 0), motif.z("p", "mat", 0)
    ).chains[0]
    out = write_pdf(
        motif.model,
        tmp_path / "chain.pdf",
        style=DiagramStyle(show_variances=False),
        highlight=chain,
        caption_name=r"\operatorname{Cov}\left[z^{(m)}_{m}, z^{(m)}_{p}\right]",
    )
    assert out.exists() and out.stat().st_size > 1000


# ======================================================================================
# task-20260804-214554: no edge may be drawn through a third node
# ======================================================================================
def _pedigree(n_generations):
    from pathmgr.genetics import am_pedigree, g_level_model

    unrolled = g_level_model(am_pedigree(n_generations))
    return unrolled.model, unrolled.layout()


@pytest.mark.parametrize("generations", [2, 3, 4, 5])
def test_no_edge_crosses_a_third_node_in_a_pedigree(generations):
    """The geometric check kept as a test, since the count grows with depth.

    An arrow driven through a variable's box reads as a mistake, and a path grazing an ellipse can
    be misread as a doubled border or even a variance self-loop. Nothing about the covariances is
    affected -- this is purely what the figure looks like -- but the pedigree figure is the one
    most likely to end up in the writeup.
    """
    from pathmgr.render.placement import edge_node_crossings, route_edges

    model, layout = _pedigree(generations)
    style = DiagramStyle(show_variances=False)
    bends = route_edges(model, layout, style)
    crossings = edge_node_crossings(
        model, layout, style, margin=style.edge_clearance, bends=bends
    )
    assert crossings == [], "\n".join(str(c) for c in crossings)


def test_the_layout_does_most_of_the_work_and_routing_only_mops_up():
    """If routing has to bend a lot of edges, the layout has regressed."""
    from pathmgr.render.placement import route_edges

    for generations in (2, 3, 4, 5):
        model, layout = _pedigree(generations)
        bends = route_edges(model, layout, DiagramStyle(show_variances=False))
        assert len(bends) <= generations, (
            f"{generations} generations needed {len(bends)} bends; the spacing defaults are "
            f"load-bearing and something has narrowed the transmission corridor"
        )


def test_crossing_detection_actually_detects():
    """A guard on the guard: the check must fail on a layout that really does cross."""
    from pathmgr.render.placement import edge_node_crossings

    model = pm.Model()
    model.add_vars("a", "middle", "b")
    model.add_path("a", "b", "q")
    model.add_variance("a", "V")
    # `middle` sits exactly on the a -> b line
    crossings = edge_node_crossings(
        model, Layout({"a": (0, 0), "middle": (2, 0), "b": (4, 0)}), DiagramStyle()
    )
    assert [c.through for c in crossings] == ["middle"]
    # and moving it off the line clears it
    assert edge_node_crossings(
        model, Layout({"a": (0, 0), "middle": (2, 3), "b": (4, 0)}), DiagramStyle()
    ) == []


def test_routing_is_a_no_op_when_nothing_crosses():
    """Which is what keeps the already-approved small figures byte-identical."""
    from pathmgr.render.placement import route_edges

    model = three_edge_model()
    layout = Layout({"g_m": (0, 0), "e_m": (1.7, 0), "y_m": (0.85, -1.9),
                     "g_p": (5.1, 0), "e_p": (6.8, 0), "y_p": (5.95, -1.9)})
    assert route_edges(model, layout, DiagramStyle()) == {}
    tex = to_tikz(model, layout=layout)
    # every DIRECTED edge stays a plain straight `--`. (Bidirected edges always use `to[bend]`;
    # curvature is part of what distinguishes them, and routing does not touch it.)
    directed = [line for line in tex.splitlines() if "pmDirected," in line]
    assert directed
    assert all(") -- " in line for line in directed)
    assert not any("to[bend" in line for line in directed)


def test_routing_is_deterministic():
    from pathmgr.render.placement import route_edges

    model, layout = _pedigree(3)
    first = route_edges(model, layout, DiagramStyle(show_variances=False))
    for _ in range(3):
        assert route_edges(model, layout, DiagramStyle(show_variances=False)) == first
    assert to_tikz(model, layout=layout) == to_tikz(model, layout=layout)


def test_a_bent_edge_is_emitted_as_a_tikz_bend():
    from pathmgr.render.placement import route_edges

    model, layout = _pedigree(4)
    style = DiagramStyle(show_variances=False)
    bends = route_edges(model, layout, style)
    assert bends, "the four-generation pedigree needs at least one bend"
    tex = to_tikz(model, layout=layout, style=style)
    assert "to[bend" in tex
    assert tex.count("to[bend") >= len(bends)


def test_routing_can_be_switched_off():
    from pathmgr.render.placement import route_edges

    model, layout = _pedigree(4)
    off = DiagramStyle(show_variances=False, route_edges_around_nodes=False)
    assert route_edges(model, layout, DiagramStyle(show_variances=False))
    assert "to[bend" not in to_tikz(model, layout=layout, style=off)


# ======================================================================================
# the caption goes through the style, like every other label (task-20260807-183029)
# ======================================================================================
def _unit_chain_model():
    """A chain carrying unit coefficients, so omit_unit visibly changes the caption."""
    return pm.from_text(
        """
        latent: z1_m, z2_m, z1_p, z2_p, g_m, g_p, e_m, e_p
        positive: V_E, beta_1, beta_2
        g_m ~ beta_1*z1_m + beta_2*z2_m
        g_p ~ beta_1*z1_p + beta_2*z2_p
        y_m ~ g_m + e_m
        y_p ~ g_p + e_p
        z1_m ~~ 1*z1_m
        z2_m ~~ 1*z2_m
        z1_p ~~ 1*z1_p
        z2_p ~~ 1*z2_p
        e_m ~~ V_E*e_m
        e_p ~~ V_E*e_p
        y_m -- (rho_y/(V_E + beta_1**2 + beta_2**2))*y_p
        """
    )


def test_caption_and_diagram_show_the_same_factors():
    """The defect was the MISMATCH, so the test asserts agreement, not either convention.

    With ``show_unit_coefficients=True`` the diagram draws every coefficient of 1 while the caption
    used to drop them: seven factors drawn against three written, on the one figure whose job is to
    let a reader check the product edge by edge.
    """
    model = _unit_chain_model()
    chain = pm.WrightTracer(model).trace("z1_m", "z1_p").chains[0]

    for show_units in (False, True):
        style = DiagramStyle(show_unit_coefficients=show_units)
        caption = chain.tex_caption(**style.caption_options())
        product = caption.split("\\\\")[1].split(" = ")[0]
        drawn = [
            style.edge_label(edge, coeff)
            for edge, coeff in [((a, b), c) for a, b, c in _chain_edge_coefficients(model, chain)]
        ]
        # a factor of 1 is drawn iff the style says so, and the caption must make the same choice
        caption_has_units = r"1 \cdot" in product or product.endswith(" 1")
        diagram_has_units = any(label == "1" for label in drawn)
        assert caption_has_units == diagram_has_units, (
            f"show_unit_coefficients={show_units}: caption {product!r} vs drawn {drawn!r}"
        )


def _chain_edge_coefficients(model, chain):
    """``(src, dst, coeff)`` for each directed edge the chain traverses."""
    out = []
    for src, dst in chain.directed_edges():
        for edge in model.directed_edges:
            if (edge.src, edge.dst) == (src, dst):
                out.append((src, dst, edge.coeff))
    return out


def test_the_default_caption_is_unchanged_by_going_through_the_style():
    """Existing figures must not move: the default style must reproduce the old default exactly."""
    model = _unit_chain_model()
    chain = pm.WrightTracer(model).trace("z1_m", "z1_p").chains[0]
    assert chain.tex_caption() == chain.tex_caption(**DiagramStyle().caption_options())


def test_a_subexpression_can_be_rendered_under_the_documents_own_name():
    """The writeup calls this sum ``\\VPo``; rendering the sum it came from is unreadable there."""
    model = _unit_chain_model()
    V_E, beta_1, beta_2 = (model.sym(s) for s in ("V_E", "beta_1", "beta_2"))
    V_P0 = V_E + beta_1**2 + beta_2**2
    chain = pm.WrightTracer(model).trace("z1_m", "z1_p").chains[0]

    plain = chain.tex_caption()
    assert "V_{E} + \\beta_{1}^{2}" in plain, plain

    style = DiagramStyle(latex_names={V_P0: r"\VPo"})
    named = chain.tex_caption(**style.caption_options())
    assert r"\VPo" in named
    assert "V_{E} + \\beta_{1}^{2}" not in named, named
    # BOTH caption lines, not just the product
    assert named.count(r"\VPo") == 2, named


@pytest.mark.parametrize("back_end", ["tikz", "raster"])
def test_document_names_reach_both_back_ends(back_end, tmp_path):
    model = _unit_chain_model()
    V_E, beta_1, beta_2 = (model.sym(s) for s in ("V_E", "beta_1", "beta_2"))
    style = DiagramStyle(latex_names={V_E + beta_1**2 + beta_2**2: r"\VPo"})
    layout = Layout({name: (i * 1.6, (i % 3) * 1.4) for i, name in enumerate(model.names)})
    chain = pm.WrightTracer(model).trace("z1_m", "z1_p").chains[0]

    if back_end == "tikz":
        out = to_tikz(model, layout=layout, style=style, highlight=chain)
        assert r"\VPo" in out
        assert "V_{E} + \\beta_{1}^{2}" not in out, "the raw sum survived somewhere in the figure"
    else:
        # the raster back end renders through mathtext, so just check it draws without raising
        path = to_image(model, tmp_path / "named.png", layout=layout, style=style, highlight=chain)
        assert path.exists() and path.stat().st_size > 0


def test_document_names_apply_to_edge_labels_too():
    """A figure calling a sum one thing on an edge and another in its caption is the same defect."""
    model = _unit_chain_model()
    V_E, beta_1, beta_2 = (model.sym(s) for s in ("V_E", "beta_1", "beta_2"))
    style = DiagramStyle(latex_names={V_E + beta_1**2 + beta_2**2: r"\VPo"})
    copath = model.copaths[0]
    assert r"\VPo" in style.copath_label(copath), style.copath_label(copath)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
def test_the_raster_back_end_falls_back_for_names_it_cannot_typeset(tmp_path):
    """A document macro is defined in the DOCUMENT, which matplotlib has never seen.

    ``\\VPo`` renders in TikZ, which is fed to real LaTeX. matplotlib's mathtext knows a fixed set
    of commands and raises on anything else -- mid-savefig, naming a macro the user never typed
    into pathmgr. So the raster preview drops names it cannot render and keeps the ones it can.
    """
    from pathmgr.render.raster import _mathtext_safe_names

    model = _unit_chain_model()
    V_E, beta_1, beta_2 = (model.sym(s) for s in ("V_E", "beta_1", "beta_2"))
    V_P0 = V_E + beta_1**2 + beta_2**2

    names = {V_P0: r"\VPo", beta_1: r"\beta_{1}"}
    safe = _mathtext_safe_names(names)
    assert V_P0 not in safe, "an undefined macro must not reach mathtext"
    assert beta_1 in safe, "a standard command must survive the filter"

    # and the figure still renders rather than raising
    layout = Layout({name: (i * 1.6, (i % 3) * 1.4) for i, name in enumerate(model.names)})
    chain = pm.WrightTracer(model).trace("z1_m", "z1_p").chains[0]
    path = to_image(
        model,
        tmp_path / "fallback.png",
        layout=layout,
        style=DiagramStyle(latex_names=names),
        highlight=chain,
    )
    assert path.exists() and path.stat().st_size > 0
