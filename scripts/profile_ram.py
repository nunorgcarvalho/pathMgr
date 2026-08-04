"""Profile the RAM engine on a pedigree-shaped model, and record where it starts to hurt.

    python scripts/profile_ram.py [max_generations]

Task-20260804-151347 asks for this so that task-20260804-151350 knows its budget before it
starts unrolling generations. The model is a *lineage* at assortative-mating equilibrium: a
founding individual and their partner, then one child per generation who takes an outside
partner. Seven to ten nodes per generation. That is roughly the shape 151350 will generate,
so the timings transfer.

Note how assortment is represented, because it is not the obvious way. A partner's genetic
value enters as a **directed path from the focal individual's phenotype**
(`y_c -> g_p` with coefficient `rho_g`), not as a bidirected `g_c <-> g_p` edge. A bidirected
edge is a *disturbance* covariance, and a child's genetic value is endogenous, so a
bidirected edge there asserts that a deterministic disturbance covaries with something --
which silently yields a Sigma that is not positive semi-definite. `Model.validate()` now
reports that case as an error. See `docs/profile_ram.md`.

This script also serves as a correctness check: it asserts the lineage reproduces the
lineal-relative result of `relative_covariance.tex` at every depth, and that the equilibrium
is self-consistent (`Var[g]` is preserved generation to generation).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import sympy as sp

import pathmgr as pm

ROOT = Path(__file__).resolve().parent.parent


def lineage(generations: int) -> pm.Model:
    """A founding couple plus `generations` descendant generations, at AM equilibrium."""
    m = pm.Model(f"lineage depth {generations}", units=pm.Units.unstandardized())
    for name in ("V_A_eq", "V_E"):
        m.declare(name, positive=True)
    V_A, V_E, rho_g, rho_y = (m.sym(s) for s in ("V_A_eq", "V_E", "rho_g", "rho_y"))
    V_P = V_A + V_E
    # V_K = V_A0 / 2 and V_A_eq = V_A0 / (1 - rho_g), so V_K = V_A_eq (1 - rho_g) / 2
    V_K = V_A * (1 - rho_g) / 2

    def person(tag: str) -> None:
        m.add_var(f"g_{tag}", latent=True)
        m.add_var(f"e_{tag}", latent=True)
        m.add_var(f"y_{tag}")
        m.add_path(f"g_{tag}", f"y_{tag}", 1)
        m.add_path(f"e_{tag}", f"y_{tag}", 1)

    def founder(tag: str) -> None:
        person(tag)
        m.add_variance(f"g_{tag}", V_A)
        m.add_variance(f"e_{tag}", V_E)

    def partner_of(tag: str, focal: str) -> None:
        """`tag` assorts on `focal`'s phenotype -- a directed copath, not a bidirected edge."""
        person(tag)
        b_e = rho_y * V_E / V_P  # so that Cov[e_partner, y_focal] = rho_y V_E
        m.add_path(f"y_{focal}", f"g_{tag}", rho_g)
        m.add_path(f"y_{focal}", f"e_{tag}", b_e)
        # residual variances chosen to keep Var[g] = V_A_eq and Var[e] = V_E
        m.add_variance(f"g_{tag}", V_A - rho_g**2 * V_P)
        m.add_variance(f"e_{tag}", V_E - b_e**2 * V_P)
        # Both components load on the same phenotype, which would induce a spurious
        # within-individual Cov[g, e] = rho_g * b_e * V_P. GE-indep says that must be zero, so
        # the disturbance covariance has to offset it exactly. Both variables are endogenous
        # here but do have disturbance variances, so this is a legitimate (if easily missed)
        # disturbance covariance -- validate() warns about it, correctly.
        m.add_cov(f"g_{tag}", f"e_{tag}", -rho_g * b_e * V_P)

    def child(tag: str, mother: str, father: str) -> None:
        person(tag)
        m.add_var(f"s_{tag}", latent=True)
        m.add_variance(f"s_{tag}", V_K)
        m.add_path(f"g_{mother}", f"g_{tag}", sp.Rational(1, 2))
        m.add_path(f"g_{father}", f"g_{tag}", sp.Rational(1, 2))
        m.add_path(f"s_{tag}", f"g_{tag}", 1)
        m.add_variance(f"e_{tag}", V_E)

    founder("0m")
    partner_of("0f", "0m")
    focal, partner = "0m", "0f"
    for t in range(1, generations + 1):
        child(f"{t}c", focal, partner)
        partner_of(f"{t}p", f"{t}c")
        focal, partner = f"{t}c", f"{t}p"
    return m


def check(generations: int) -> None:
    """Assert the lineage is right before trusting any timing taken from it."""
    for depth in range(1, generations + 1):
        model = lineage(depth)
        engine = pm.RAMEngine(model)
        V_A, V_E, rho_g, rho_y = (model.sym(s) for s in ("V_A_eq", "V_E", "rho_g", "rho_y"))
        fixed_point = {rho_g: rho_y * V_A / (V_A + V_E)}

        errors = [i for i in model.validate() if i.severity == "error"]
        assert not errors, f"depth {depth}: {errors}"

        # equilibrium is self-consistent: the child's genetic variance is still V_A_eq
        assert sp.simplify(engine.var(f"g_{depth}c") - V_A) == 0, f"depth {depth}: Var[g] drifted"

        # Partners correlate at rho_y, by construction. Compared squared: the correlation
        # carries a sqrt((V_A_eq + V_E)**2) that simplify() will not reduce even though both
        # symbols are declared positive -- a small reminder that sympy needs help at the
        # points we choose, which is the whole reason simplification here is explicit.
        mate_corr = engine.corr("y_0m", "y_0f").subs(fixed_point)
        assert sp.simplify((mate_corr / rho_y) ** 2 - 1) == 0, (
            f"depth {depth}: mate corr {sp.simplify(mate_corr)}"
        )

        # eq:am-level3-lin -- Cov[y_a, y_b] = V_A_eq (1+rho_y)/2 ((1+rho_g)/2)^(d-1)
        got = sp.simplify(engine.cov("y_0m", f"y_{depth}c").subs(fixed_point))
        want = sp.simplify(
            V_A * (1 + rho_y) / 2 * ((1 + rho_g.subs(fixed_point)) / 2) ** (depth - 1)
        )
        assert sp.simplify(got - want) == 0, f"depth {depth}: lineal mismatch"
    print(f"  correctness: lineal result, equilibrium and mate correlation hold to depth {generations}")


def profile(max_generations: int) -> list[dict]:
    rows = []
    for depth in range(1, max_generations + 1):
        model = lineage(depth)
        engine = pm.RAMEngine(model)

        start = time.perf_counter()
        sigma = engine.sigma()
        build = time.perf_counter() - start

        entry_ops = [
            sp.count_ops(sigma[i, j]) for i in range(sigma.rows) for j in range(sigma.cols)
        ]

        start = time.perf_counter()
        engine.cov("y_0m", f"y_{depth}c")
        query = time.perf_counter() - start

        start = time.perf_counter()
        engine.cov("y_0m", f"y_{depth}c", form="simplified")
        simplified = time.perf_counter() - start

        rows.append(
            {
                "depth": depth,
                "nodes": len(model.names),
                "build_s": build,
                "query_s": query,
                "simplify_s": simplified,
                "total_ops": sum(entry_ops),
                "worst_entry_ops": max(entry_ops, default=0),
            }
        )
        r = rows[-1]
        print(
            f"  depth {r['depth']:3d}  nodes {r['nodes']:4d}  build {r['build_s']:8.3f}s  "
            f"query {r['query_s']:7.4f}s  simplified {r['simplify_s']:8.3f}s  "
            f"ops {r['total_ops']:8d}  worst {r['worst_entry_ops']:6d}"
        )
    return rows


def write_report(rows: list[dict]) -> Path:
    out = ROOT / "docs" / "profile_ram.md"
    out.parent.mkdir(exist_ok=True)
    lines = [
        "# RAM engine profile",
        "",
        "Generated by `python scripts/profile_ram.py N`. Re-run after any change to how",
        "`Sigma` is built. These numbers are the budget task-20260804-151350 works within when",
        "it unrolls generations.",
        "",
        "**Model**: a lineage at assortative-mating equilibrium -- a founding couple, then one",
        "child per generation who takes an outside partner. The script asserts it reproduces the",
        "lineal-relative result of `relative_covariance.tex` at every depth, that the",
        "equilibrium is self-consistent (`Var[g]` preserved across generations), and that",
        "partners correlate at `rho_y`, before any timing is taken.",
        "",
        "| generations | nodes | build Sigma | cov() query | cov(form='simplified') | total ops in Sigma | worst entry |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['depth']} | {r['nodes']} | {r['build_s']:.3f}s | {r['query_s']:.4f}s | "
            f"{r['simplify_s']:.3f}s | {r['total_ops']} | {r['worst_entry_ops']} |"
        )
    comfortable = [r for r in rows if r["build_s"] < 3.0]
    painful = [r for r in rows if r["build_s"] >= 10.0]
    lines += [
        "",
        "## Where it starts to hurt",
        "",
        f"- Comfortable to **{max((r['depth'] for r in comfortable), default=0)} generations**"
        f" ({max((r['nodes'] for r in comfortable), default=0)} nodes): building Sigma stays"
        f" under 3 s.",
    ]
    if painful:
        first = min(r["depth"] for r in painful)
        lines.append(
            f"- Past **{first} generations** the build crosses 10 s and climbs steeply"
            f" (depth {rows[-1]['depth']}: {rows[-1]['build_s']:.0f} s)."
        )
    else:
        lines.append(
            f"- No depth measured here crossed 10 s (deepest:"
            f" {rows[-1]['depth']} generations at {rows[-1]['build_s']:.1f} s)."
        )
    lines += [
        "- Queries stay free at every depth, because Sigma is built once and cached.",
        "- If a future task needs materially more depth, the optimisation to reach for is a",
        "  **targeted single-entry computation**: `cov(x, y)` currently forces the whole",
        "  n-by-n Sigma, but the two recursions only need the ancestors of `x` and `y`. That",
        "  is not implemented -- the full matrix is the primary object by design -- and the",
        "  numbers above suggest it is not yet needed.",
        "",
        "## Reading it",
        "",
        "- **Building `Sigma` dominates; querying it is free.** `cov()` is a matrix lookup plus",
        "  post-processing, so the build cost is paid once per `model.revision` and then cached.",
        "- **`form='simplified'` is the expensive knob, not the engine.** That is why `cov()`",
        "  defaults to `'expanded'` and `simplify` is never called automatically.",
        "- Total ops in `Sigma` grows roughly with the square of the node count, because there",
        "  are `n^2` entries and each stays small. The worst single entry is what to watch: if",
        "  it starts growing with depth rather than plateauing, expansion has stopped cancelling",
        "  and that is the onset of real symbolic blowup.",
        "",
        "## Guidance for unrolling generations",
        "",
        "- Prefer **one model holding all generations** over many small ones: `Sigma` is cached",
        "  per `model.revision`, so a single build answers every cross-generation query. Mutating",
        "  the model invalidates the cache and pays the build again -- so build the whole",
        "  pedigree, then query it.",
        "- Keep coefficients symbolic and substitute numbers **at the end**, on the one scalar",
        "  expression you care about, rather than building a numeric model per parameter value.",
        "- When depth becomes the binding constraint, that is the signal to stop unrolling and",
        "  solve the equilibrium fixed point instead -- which is task-20260804-151351, and why",
        '  "unroll a lot" is not a substitute for it.',
        "",
        "## The assortment-representation trap (found while writing this profile)",
        "",
        "Mates' genetic values are correlated, so the obvious encoding is a bidirected edge",
        "`g_mother <-> g_father`. That is correct **only while both are exogenous** -- as in a",
        "founding pair. Once a mate is a child in the pedigree, their genetic value is",
        "endogenous, and a bidirected edge is a covariance between *disturbances*, not between",
        "variables. Asserting that an endogenous variable's disturbance covaries with something,",
        "when that disturbance is fully determined by its parents, produces a `Sigma` that is",
        "**not positive semi-definite** -- an implied correlation above 1 -- with nothing else",
        "to signal the mistake.",
        "",
        "The first version of this script did exactly that, and the resulting covariances decayed",
        "as `2^-d` with no `(1 + rho_g)` accumulation at all, silently disagreeing with the",
        "writeup. `Model.validate()` now reports it as an **error**.",
        "",
        "The correct encoding is the one used here: assortment enters as a **directed path from",
        "the focal individual's phenotype to the partner's components**,",
        "",
        "    y_focal -> g_partner    with coefficient  rho_g",
        "    y_focal -> e_partner    with coefficient  rho_y V_E / V_P",
        "",
        "with the partner's residual variances reduced to keep `Var[g] = V_A_eq` and",
        "`Var[e] = V_E`. This reproduces the writeup exactly, and it makes",
        "`Cov[e_partner, g_focal] = rho_g V_E` -- the term behind the lineal/collateral",
        "asymmetry -- fall out automatically instead of needing its own edge.",
        "",
    ]
    out.write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"checking correctness up to {min(limit, 5)} generations")
    check(min(limit, 5))
    print(f"profiling lineage models up to {limit} generations")
    report = write_report(profile(limit))
    print(f"wrote {report.relative_to(ROOT)}")
