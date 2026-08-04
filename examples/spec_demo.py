"""A runnable tour of the pathMgr specification API.

    python examples/spec_demo.py

Four sections: the specification object on three models, then the text front-end, the RAM
engine, the Wright tracer, and co-paths.

Three models of increasing awkwardness, chosen to exercise every part of the spec object:

1. bivariate regression      -- the standard SEM smoke test; all observed
2. relative covariance, S1   -- latents, and a covariance between two individuals' latents
3. one AM transmission unit  -- rational + symbolic-expression coefficients, and a
                                covariance between one person's environment and another's
                                genes

Each is built twice -- with the python builder and with the text front-end -- and the two
are checked to agree, since that is the contract between them. See also
``examples/am_equilibrium.pmg`` for the AM model as a standalone text file.

Nothing here computes a covariance; that is the engine (task-20260804-151347). This file
only shows what the model object can *say*.
"""

from pathlib import Path

import sympy as sp

import pathmgr as pm

HERE = Path(__file__).resolve().parent


def bivariate_regression() -> pm.Model:
    m = pm.Model("bivariate regression")
    for v in ("V_1", "V_2", "V_r"):
        m.declare(v, positive=True)
    m.add_vars("x1", "x2", "y")
    m.add_path("x1", "y", "b1")
    m.add_path("x2", "y", "b2")
    m.add_variance("x1", "V_1")
    m.add_variance("x2", "V_2")
    m.add_variance("y", "V_r")  # residual variance, not total
    m.add_cov("x1", "x2", "c12")
    return m


def relative_covariance_section1() -> pm.Model:
    """y_i = g_i + e_i for two individuals, with Cov[g_i, g_j] = V_A * pi_ij."""
    m = pm.Model("relative covariance S1", units=pm.Units.unstandardized())
    for v in ("V_A", "V_E"):
        m.declare(v, positive=True)
    for i in ("i", "j"):
        m.add_var(f"g_{i}", latent=True, label=rf"$g_{i}$")
        m.add_var(f"e_{i}", latent=True, label=rf"$e_{i}$")
        m.add_var(f"y_{i}", label=rf"$y_{i}$")
        m.add_path(f"g_{i}", f"y_{i}", 1)
        m.add_path(f"e_{i}", f"y_{i}", 1)
        m.add_variance(f"g_{i}", "V_A")
        m.add_variance(f"e_{i}", "V_E")
    m.add_cov("g_i", "g_j", "V_A * pi_ij")
    m.assume("V_A + V_E", 1)  # V_P = 1 in the base population
    return m


def am_transmission_unit() -> pm.Model:
    """One mated pair and one child, under phenotypic assortative mating.

    The parts worth staring at:
      - transmission coefficients are exact rationals, not floats;
      - the assortment edge is an *expression*, rho_g * V_A_eq;
      - Cov[e_m, g_f] = rho_g * V_E is an edge between one individual's environment and
        the other's genes -- the term that makes lineal relatives differ from collateral
        ones, and the reason bidirected edges must be allowed between arbitrary latents;
      - rho_g = rho_y * h2_eq and V_A_eq = V_A0 / (1 - rho_g) are recorded as *assumptions*,
        not edges: they are the equilibrium fixed point, to be solved, not traced.
    """
    m = pm.Model("AM: one transmission", units=pm.Units.unstandardized())
    for v in ("V_A_eq", "V_E", "V_K", "V_A0"):
        m.declare(v, positive=True)

    for i in ("m", "f", "o"):  # mother, father, offspring
        m.add_var(f"g_{i}", latent=True, label=rf"$g_{i}$")
        m.add_var(f"e_{i}", latent=True, label=rf"$e_{i}$")
        m.add_var(f"y_{i}", label=rf"$y_{i}$")
        m.add_path(f"g_{i}", f"y_{i}", 1)
        m.add_path(f"e_{i}", f"y_{i}", 1)
        m.add_variance(f"e_{i}", "V_E")

    # parents: exogenous genetic values at the equilibrium scale, correlated by assortment
    for i in ("m", "f"):
        m.add_variance(f"g_{i}", "V_A_eq")
    m.add_cov("g_m", "g_f", "rho_g * V_A_eq")

    # transmission: g_o = (g_m + g_f)/2 + s_o
    m.add_var("s_o", latent=True, label=r"$s_o$")
    m.add_path("g_m", "g_o", sp.Rational(1, 2))
    m.add_path("g_f", "g_o", sp.Rational(1, 2))
    m.add_path("s_o", "g_o", 1)
    m.add_variance("s_o", "V_K")

    # each parent's environment is correlated with the *other* parent's genes
    m.add_cov("e_m", "g_f", "rho_g * V_E")
    m.add_cov("e_f", "g_m", "rho_g * V_E")

    m.assume("rho_g", "rho_y * h2_eq")
    m.assume("h2_eq", "V_A_eq / (V_A_eq + V_E)")
    m.assume("V_A_eq", "V_A0 / (1 - rho_g)")
    m.assume("V_K", "V_A0 / 2")
    return m


def show(m: pm.Model) -> None:
    print("=" * 78)
    print(m.describe())
    issues = m.validate()
    print(f"  validate: {'clean' if not issues else ''}")
    for i in issues:
        print(f"    {i}")
    A, S, F, order = m.ram()
    print(f"  RAM order: {order}")
    print(f"  A ({A.shape[0]}x{A.shape[1]}), S, F ({F.shape[0]}x{F.shape[1]})")
    print("  A =")
    print(sp.pretty(A, use_unicode=False))
    print("  S =")
    print(sp.pretty(S, use_unicode=False))
    print()
    print("  the same model in the text grammar (m.to_text()):")
    for line in m.to_text().splitlines():
        print(f"    {line}")
    print()


def text_front_end_tour() -> None:
    """The text grammar: terser for a hand-written model, and it round-trips."""
    print("=" * 78)
    print("TEXT FRONT-END")
    print()

    # the bivariate regression, as text
    m = pm.from_text(
        """
        positive: V_1, V_2, V_r
        y ~ b1*x1 + b2*x2
        x1 ~~ V_1*x1
        x2 ~~ V_2*x2
        y  ~~ V_r*y
        x1 ~~ c12*x2
        """,
        name="bivariate regression (from text)",
    )
    print("  parsed:", m)
    print("  matches the builder version:", _same(m, bivariate_regression()))
    print()

    # loaded from a file, and round-tripped. The builder twin below is the superseded
    # hand-written-covariance encoding, so it is the *handwritten* fixture that matches it;
    # am_equilibrium.pmg now uses a co-path (see the CO-PATHS section).
    from_file = pm.from_text(
        (HERE / "am_equilibrium_handwritten.pmg").read_text(), name="AM (from file)"
    )
    print("  loaded examples/am_equilibrium_handwritten.pmg:", from_file)
    print("  matches the builder version:", _same(from_file, am_transmission_unit_pair()))
    print("  survives to_text -> from_text:", _same(pm.from_text(from_file.to_text()), from_file))
    copath_file = pm.from_text((HERE / "am_equilibrium.pmg").read_text(), name="AM (co-path)")
    print("  loaded examples/am_equilibrium.pmg (co-path):", copath_file)
    print("  survives to_text -> from_text:",
          _same(pm.from_text(copath_file.to_text()), copath_file))
    print()

    # errors point at the line
    for bad, why in [
        ("g ~~ (V_A*pi_ij)", "no trailing variable"),
        ("units: standardized\ny ~ x", "no reference population"),
        ("y ~ b1*x\ny ~ b2*x", "duplicate edge"),
    ]:
        try:
            pm.from_text(bad)
        except (pm.TextSyntaxError, ValueError) as exc:
            print(f"  rejected ({why}): {str(exc).splitlines()[0]}")
    print()


def am_transmission_unit_pair() -> pm.Model:
    """The `.pmg` file's model, built with builder calls, for the equivalence check."""
    m = pm.Model("AM (builder)", units=pm.Units.unstandardized())
    for v in ("V_A_eq", "V_E", "V_K", "V_A0"):
        m.declare(v, positive=True)
    for i in ("m", "f", "o1", "o2"):
        m.add_var(f"g_{i}", latent=True, label=rf"$g_{{{i}}}$")
        m.add_var(f"e_{i}", latent=True, label=rf"$e_{{{i}}}$")
        m.add_var(f"y_{i}", label=rf"$y_{{{i}}}$")
        m.add_path(f"g_{i}", f"y_{i}", 1)
        m.add_path(f"e_{i}", f"y_{i}", 1)
        m.add_variance(f"e_{i}", "V_E")
    for i in ("m", "f"):
        m.add_variance(f"g_{i}", "V_A_eq")
    m.add_cov("g_m", "g_f", "rho_g * V_A_eq")
    m.add_cov("e_m", "g_f", "rho_g * V_E")
    m.add_cov("e_f", "g_m", "rho_g * V_E")
    for o in ("o1", "o2"):
        m.add_var(f"s_{o}", latent=True, label=rf"$s_{{{o}}}$")
        m.add_path("g_m", f"g_{o}", sp.Rational(1, 2))
        m.add_path("g_f", f"g_{o}", sp.Rational(1, 2))
        m.add_path(f"s_{o}", f"g_{o}", 1)
        m.add_variance(f"s_{o}", "V_K")
    m.assume("rho_g", "rho_y * h2_eq")
    m.assume("h2_eq", "V_A_eq / (V_A_eq + V_E)")
    m.assume("V_A_eq", "V_A0 / (1 - rho_g)")
    m.assume("V_K", "V_A0 / 2")
    return m


def _same(a: pm.Model, b: pm.Model) -> bool:
    """Order- and name-insensitive model comparison."""

    def key(m: pm.Model):
        return (
            m.units,
            frozenset(m.observed),
            frozenset(m.latent),
            frozenset((e.src, e.dst, sp.srepr(e.coeff)) for e in m.directed_edges),
            frozenset((e.a, e.b, sp.srepr(e.value)) for e in m.bidirected_edges),
            frozenset((c.a, c.b, c.process, sp.srepr(c.coefficient)) for c in m.copaths),
            frozenset(sp.srepr(eq) for eq in m.assumptions),
        )

    return key(a) == key(b)


def engine_tour() -> None:
    """The RAM engine: covariances between any two variables, latents included."""
    print("=" * 78)
    print("RAM ENGINE")
    print()

    m = relative_covariance_section1()
    e = pm.RAMEngine(m)
    print(f"  {len(m.names)} nodes, {len(m.observed)} observed; used inverse: {e.used_inverse}")
    print()
    print("  Section 1 of relative_covariance.tex, derived rather than recorded:")
    for x, y in [("y_i", "y_j"), ("g_i", "g_j"), ("g_i", "y_i"), ("e_i", "e_j"), ("g_i", "e_j")]:
        kind = "/".join("latent" if m.var(v).latent else "observed" for v in (x, y))
        print(f"    Cov[{x:>3}, {y:>3}] = {str(e.cov(x, y)):<16} ({kind})")
    print(f"    Var[y_i]         = {e.var('y_i')}")
    print(f"    Corr[y_i, y_j]   = {e.corr('y_i', 'y_j')}")
    print()
    print("  the boxed Level-2 result E[y_i y_j | pi_ij] = V_A * pi_ij:",
          sp.simplify(e.cov("y_i", "y_j") - m.sym("V_A") * m.sym("pi_ij")) == 0)
    print("  and Cov[y_i,y_j] == Cov[g_i,g_j] (eq:reduce):",
          sp.simplify(e.cov("y_i", "y_j") - e.cov("g_i", "g_j")) == 0)
    print()

    print("  a feedback loop -- infinitely many Wright chains, summed in closed form:")
    cyc = pm.Model("feedback")
    cyc.add_vars("x", "y", "z")
    cyc.add_path("x", "y", "a")
    cyc.add_path("y", "z", "b")
    cyc.add_path("z", "y", "d")
    for v in ("x", "y", "z"):
        cyc.add_variance(v, f"S_{v}")
    ec = pm.RAMEngine(cyc)
    print(f"    recursive: {cyc.is_recursive}; used inverse: {ec.used_inverse}")
    print(f"    Cov[x, y] = {sp.factor(ec.cov('x', 'y'))}")
    print()

    print("  the disturbance-covariance trap that validate() now catches:")
    trap = pm.Model("trap")
    trap.add_vars("x", "y", "z")
    trap.add_path("x", "y", "a")
    trap.add_variance("x", "V_x")
    trap.add_variance("z", "V_z")
    trap.add_cov("y", "z", "c")  # y is endogenous with no disturbance variance
    for issue in trap.validate():
        print(f"    {str(issue)[:100]}...")
    print()


def tracer_tour() -> None:
    """The Wright tracer: the decomposition, which is the part that goes into a writeup."""
    print("=" * 78)
    print("WRIGHT TRACER")
    print()

    m = relative_covariance_section1()
    print("  Section 1 -- why the phenotypic covariance IS the genetic covariance:")
    print(_indent(pm.WrightTracer(m).trace("y_i", "y_j")))
    print()

    print("  a mediation chain, direct and indirect routes itemized separately:")
    med = pm.from_text(
        """
        positive: V_x, V_m, V_y
        m ~ a*x
        y ~ b*m + c*x
        x ~~ V_x*x
        m ~~ V_m*m
        y ~~ V_y*y
        """
    )
    print(_indent(pm.WrightTracer(med).trace("x", "y")))
    print()

    print("  and the same decomposition as LaTeX, ready to paste into a writeup:")
    print(_indent(pm.WrightTracer(med).trace("x", "y").to_latex()))
    print()

    print("  the two engines agree -- pathMgr's standing correctness property:")
    engine, tracer = pm.RAMEngine(m), pm.WrightTracer(m)
    for x, y in [("y_i", "y_j"), ("g_i", "g_j"), ("y_i", "y_i")]:
        agree = sp.simplify(tracer.cov(x, y) - engine.cov(x, y)) == 0
        print(f"    Cov[{x}, {y}]: traced == matrix -> {agree}")
    print()

    print("  a chain may visit a node in BOTH legs, and must for correctness:")
    tp = pm.from_text(
        """
        positive: V_b, V_c, V_w, V_x, V_y
        w ~ p_b*b + p_c*c
        x ~ q*w
        y ~ r*w
        b ~~ V_b*b
        c ~~ V_c*c
        b ~~ C_bc*c
        w ~~ V_w*w
        x ~~ V_x*x
        y ~~ V_y*y
        """
    )
    print(_indent(pm.WrightTracer(tp).trace("x", "y")))
    print("    (Cov[x,y] = q*r*Var[w]; the w-revisiting chains carry Var[w]'s ancestral part)")
    print()

    print("  a feedback loop cannot be enumerated, and says so:")
    cyc = pm.Model("feedback")
    cyc.add_vars("x", "y", "z")
    cyc.add_path("x", "y", "a")
    cyc.add_path("y", "z", "b")
    cyc.add_path("z", "y", "d")
    cyc.add_variance("x", "S_x")
    try:
        pm.WrightTracer(cyc).trace("x", "y")
    except pm.UntraceableModelError as exc:
        print(f"    {str(exc)[:150]}...")
    print()


def _indent(block: object, pad: str = "    ") -> str:
    return "\n".join(pad + line for line in str(block).splitlines())


def copath_tour() -> None:
    """Co-paths: covariance from matching, which a bidirected edge cannot express."""
    print("=" * 78)
    print("CO-PATHS  (assortative mating)")
    print()

    m = pm.from_text(
        """
        latent: g_m, e_m, g_f, e_f
        positive: V_A, V_E
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
    e = pm.RAMEngine(m)
    V_A, V_E, rho_y = (m.sym(s) for s in ("V_A", "V_E", "rho_y"))
    V_P = V_A + V_E
    print("  ONE co-path on the phenotypes induces covariance among ALL the causes:")
    for x, y in [("y_m", "y_f"), ("g_m", "g_f"), ("e_m", "g_f"), ("e_m", "e_f")]:
        print(f"    Cov[{x}, {y}] = {sp.factor(sp.simplify(e.cov(x, y)))}")
    print(f"    ... and Cov[y_m,y_f] == rho_y*V_P: "
          f"{sp.simplify(e.cov('y_m', 'y_f') - rho_y * V_P) == 0}")
    print("    none of the cause-level covariances were specified.")
    print()
    print("  the decomposition is exactly Sunde's Eq. (2):")
    print(_indent(pm.WrightTracer(m).trace("y_m", "y_f")))
    print()

    print("  a co-path induces covariance WITHOUT causing variance:")
    without = m.copy()
    without.remove_copath("y_m", "y_f")
    same = all(
        sp.simplify(e.var(n) - pm.RAMEngine(without).var(n)) == 0 for n in m.names
    )
    print(f"    every variance unchanged by adding the co-path: {same}")
    print()

    print("  the decisive contrast -- split g into alleles, g = beta*(z_mat + z_pat):")
    allele = pm.from_text(
        """
        latent: z_mat_m, z_pat_m, g_m, e_m, z_mat_f, z_pat_f, g_f, e_f
        positive: beta, V_E
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
    ea = pm.RAMEngine(allele)
    beta, V_E2, ry2 = (allele.sym(s) for s in ("beta", "V_E", "rho_y"))
    print(f"    co-path      -> Cov[z_mat_m, z_mat_f] = "
          f"{sp.factor(sp.simplify(ea.cov('z_mat_m', 'z_mat_f')))}")
    faked = allele.copy()
    faked.remove_copath("y_m", "y_f")
    faked.add_cov("y_m", "y_f", ry2 * (beta**2 + V_E2))
    print(f"    bidirected   -> Cov[z_mat_m, z_mat_f] = "
          f"{pm.RAMEngine(faked).cov('z_mat_m', 'z_mat_f')}   <- the whole point")
    print()

    print("  chains may cross co-paths from DIFFERENT mating processes (but not the same one):")
    shared = pm.from_text(
        """
        latent: g_1, e_1, g_P, e_P, g_2, e_2
        positive: V_A, V_E
        y_1 ~ g_1 + e_1
        y_P ~ g_P + e_P
        y_2 ~ g_2 + e_2
        g_1 ~~ V_A*g_1
        g_P ~~ V_A*g_P
        g_2 ~~ V_A*g_2
        e_1 ~~ V_E*e_1
        e_P ~~ V_E*e_P
        e_2 ~~ V_E*e_2
        y_1 -- (rho_y/(V_A + V_E))*y_P [couple_1P]
        y_2 -- (rho_y/(V_A + V_E))*y_P [couple_2P]
        """,
        name="shared partner",
    )
    print(_indent(pm.WrightTracer(shared).trace("g_1", "g_2")))
    print("    (two people who each mated with the same third person are correlated,")
    print("     with no shared ancestry at all -- and this is why half-sibs break (1+rho_g)^d)")
    print()


if __name__ == "__main__":
    for model in (
        bivariate_regression(),
        relative_covariance_section1(),
        am_transmission_unit(),
    ):
        show(model)
    text_front_end_tour()
    engine_tour()
    tracer_tour()
    copath_tour()
