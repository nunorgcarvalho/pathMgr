"""A runnable tour of the pathMgr specification API.

    python examples/spec_demo.py

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

    # loaded from a file, and round-tripped
    from_file = pm.from_text((HERE / "am_equilibrium.pmg").read_text(), name="AM (from file)")
    print("  loaded examples/am_equilibrium.pmg:", from_file)
    print("  matches the builder version:", _same(from_file, am_transmission_unit_pair()))
    print("  survives to_text -> from_text:", _same(pm.from_text(from_file.to_text()), from_file))
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


if __name__ == "__main__":
    for model in (
        bivariate_regression(),
        relative_covariance_section1(),
        am_transmission_unit(),
    ):
        show(model)
    text_front_end_tour()
    engine_tour()
