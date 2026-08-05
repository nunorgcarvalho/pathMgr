"""Produce the figures this package exists to make.

    python examples/make_figures.py [output_dir]

Writes, for each figure, a ``.tikz`` snippet to paste into a writeup, a compiled ``.pdf`` if
pdflatex is available, and a ``.png`` for slides or a quick look. The pedigree figures use
**explicit coordinates**, because that is the reliable path and a pedigree's natural layout
(generations as rows) is obvious; the auto-layout fallback is only there for arbitrary graphs.

The last figure is the one worth the trouble: a single Wright chain highlighted on the diagram,
captioned with the chain itself. That is the diagram and the covariance in one object.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import sympy as sp

import pathmgr as pm
from pathmgr.render import DiagramStyle, Layout, to_image, to_tikz, write_pdf

ROOT = Path(__file__).resolve().parent.parent
HAS_PDFLATEX = shutil.which("pdflatex") is not None


def mated_pair() -> tuple[pm.Model, Layout]:
    """One mated pair assorting on the phenotype, with a co-path."""
    model = pm.from_text(
        """
        latent: g_m, e_m, g_f, e_f
        positive: V_A, V_E
        label: g_m = $g_m$
        label: e_m = $e_m$
        label: y_m = $y_m$
        label: g_f = $g_f$
        label: e_f = $e_f$
        label: y_f = $y_f$
        y_m ~ g_m + e_m
        y_f ~ g_f + e_f
        g_m ~~ V_A*g_m
        e_m ~~ V_E*e_m
        g_f ~~ V_A*g_f
        e_f ~~ V_E*e_f
        y_m -- (rho_y/(V_A + V_E))*y_f
        """,
        name="mated pair",
    )
    layout = Layout(
        {
            "g_m": (0.0, 0.0), "e_m": (1.7, 0.0), "y_m": (0.85, -1.9),
            "g_f": (5.1, 0.0), "e_f": (6.8, 0.0), "y_f": (5.95, -1.9),
        }
    )
    return model, layout


def allele_level_pair() -> tuple[pm.Model, Layout]:
    """The same pair with each g resolved into two allele nodes -- the decisive figure.

    A co-path on the phenotypes induces covariance all the way down to the alleles; a bidirected
    edge there would give zero. Drawing it makes that visible.
    """
    model = pm.from_text(
        """
        latent: z_mat_m, z_pat_m, g_m, e_m, z_mat_f, z_pat_f, g_f, e_f
        positive: beta, V_E
        label: z_mat_m = $z^{(m)}_{m}$
        label: z_pat_m = $z^{(p)}_{m}$
        label: z_mat_f = $z^{(m)}_{f}$
        label: z_pat_f = $z^{(p)}_{f}$
        label: g_m = $g_m$
        label: e_m = $e_m$
        label: y_m = $y_m$
        label: g_f = $g_f$
        label: e_f = $e_f$
        label: y_f = $y_f$
        g_m ~ beta*z_mat_m + beta*z_pat_m
        g_f ~ beta*z_mat_f + beta*z_pat_f
        y_m ~ g_m + e_m
        y_f ~ g_f + e_f
        z_mat_m ~~ 1/2*z_mat_m
        z_pat_m ~~ 1/2*z_pat_m
        z_mat_f ~~ 1/2*z_mat_f
        z_pat_f ~~ 1/2*z_pat_f
        e_m ~~ V_E*e_m
        e_f ~~ V_E*e_f
        y_m -- (rho_y/(beta**2 + V_E))*y_f
        """,
        name="allele level",
    )
    layout = Layout(
        {
            "z_mat_m": (0.0, 1.9), "z_pat_m": (1.8, 1.9),
            "g_m": (0.9, 0.0), "e_m": (2.7, 0.0), "y_m": (1.4, -2.0),
            "z_mat_f": (5.8, 1.9), "z_pat_f": (7.6, 1.9),
            "g_f": (6.7, 0.0), "e_f": (4.9, 0.0), "y_f": (6.2, -2.0),
        }
    )
    return model, layout


def pair_with_two_children() -> tuple[pm.Model, Layout]:
    """The AM transmission unit: a mated pair and two full sibs, generations as rows."""
    model = pm.from_text((ROOT / "examples" / "am_equilibrium.pmg").read_text(), name="AM unit")
    layout = Layout(
        {
            "g_m": (0.0, 0.0), "e_m": (1.6, 0.0), "y_m": (0.8, -1.9),
            "g_f": (5.6, 0.0), "e_f": (7.2, 0.0), "y_f": (6.4, -1.9),
            "s_o1": (0.4, -3.7), "g_o1": (2.2, -3.7), "e_o1": (4.0, -3.7),
            "y_o1": (2.2, -5.5),
            "s_o2": (7.6, -3.7), "g_o2": (5.8, -3.7), "e_o2": (9.2, -3.7),
            "y_o2": (5.8, -5.5),
        }
    )
    return model, layout


def allele_transmission_motif() -> tuple[pm.Model, Layout]:
    """The single-variant allele motif: the figure task-20260804-173344 asks for.

    Makes the design decision visible: BOTH of each parent's alleles feed the child's allele from
    that parent, each at 1/2, and which one was actually transmitted is not represented at all.
    Generations as rows, explicit coordinates.
    """
    from pathmgr.genetics import allele_motif

    motif = allele_motif(n_variants=1)
    m = motif.model
    layout = Layout(
        {
            # generation 0: mother left, father right
            motif.z("m", "mat", 0): (0.0, 3.2), motif.z("m", "pat", 0): (2.0, 3.2),
            motif.x("m", 0): (1.0, 1.7), motif.g("m"): (1.0, 0.3), motif.e("m"): (-1.0, 0.3),
            motif.y("m"): (1.0, -1.1),
            motif.z("f", "mat", 0): (7.0, 3.2), motif.z("f", "pat", 0): (9.0, 3.2),
            motif.x("f", 0): (8.0, 1.7), motif.g("f"): (8.0, 0.3), motif.e("f"): (10.0, 0.3),
            motif.y("f"): (8.0, -1.1),
            # generation 1
            motif.z("o", "mat", 0): (3.4, -3.0), motif.z("o", "pat", 0): (5.6, -3.0),
            motif.s("o", "mat", 0): (1.4, -3.0), motif.s("o", "pat", 0): (7.6, -3.0),
            motif.x("o", 0): (4.5, -4.5), motif.g("o"): (4.5, -5.9), motif.e("o"): (6.9, -5.9),
            motif.y("o"): (4.5, -7.3),
        }
    )
    return m, layout


def emit(name: str, model: pm.Model, layout: Layout, out: Path, style: DiagramStyle, **kwargs):
    (out / f"{name}.tikz").write_text(to_tikz(model, layout=layout, style=style, **kwargs))
    to_image(model, out / f"{name}.png", layout=layout, style=style, dpi=200, **kwargs)
    status = "tikz + png"
    if HAS_PDFLATEX:
        write_pdf(model, out / f"{name}.pdf", layout=layout, style=style, **kwargs)
        status += " + pdf"
    print(f"  {name:34s} {status}")


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plain = DiagramStyle(show_variances=True)
    tidy = DiagramStyle(show_variances=False)

    print(f"writing figures to {out}")

    model, layout = mated_pair()
    emit("mated_pair", model, layout, out, plain)

    allele_model, allele_layout = allele_level_pair()
    emit("allele_level", allele_model, allele_layout, out, tidy)

    unit_model, unit_layout = pair_with_two_children()
    emit("am_unit", unit_model, unit_layout, out, tidy)

    allele_transmission, allele_transmission_layout = allele_transmission_motif()
    emit("allele_transmission", allele_transmission, allele_transmission_layout, out, tidy)

    # -- the figure this project exists to make ---------------------------------------
    tracer = pm.WrightTracer(allele_model)
    decomposition = tracer.trace("z_mat_m", "z_mat_f")
    chain = decomposition.chains[0]
    # `tidy` hides the CONTEXT variances; the chain's own z <-> z loops are drawn regardless,
    # because they carry the 1/2 * 1/2 that produces the /4 in the result
    name = r"\operatorname{Cov}\left[z^{(m)}_{m}, z^{(m)}_{f}\right]" if len(decomposition) == 1 else None
    emit(
        "allele_chain_highlighted", allele_model, allele_layout, out, tidy,
        highlight=chain, caption_name=name,
    )

    beta, V_E, rho_y = (allele_model.sym(s) for s in ("beta", "V_E", "rho_y"))
    expected = beta**2 * rho_y / (4 * (beta**2 + V_E))
    assert sp.simplify(decomposition.total - expected) == 0
    (out / "allele_chain.tex").write_text(decomposition.to_latex())
    print(f"  {'allele_chain.tex':34s} the matching decomposition, as align*")
    print()
    print("  the highlighted chain is:")
    print(f"    {chain.path_string()}")
    print(f"    = {sp.factor(decomposition.total)}")
    print()
    print("  which is Cov[z_mat_m, z_mat_f] = beta^2 rho_y / (4 V_P) -- a co-path reaching")
    print("  the alleles, where a bidirected edge would give exactly 0.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(ROOT / "docs" / "figures"),
        help="directory to write the figures into",
    )
    # argparse rejects an unknown flag outright, so a mistyped `--outdir DIR` errors instead of
    # silently creating a directory literally named "--outdir".
    main(Path(parser.parse_args().output_dir))
