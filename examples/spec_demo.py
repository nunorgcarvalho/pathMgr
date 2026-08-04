"""A runnable tour of the pathMgr specification API.

    python examples/spec_demo.py

Three models of increasing awkwardness, chosen to exercise every part of the spec object:

1. bivariate regression      -- the standard SEM smoke test; all observed
2. relative covariance, S1   -- latents, and a covariance between two individuals' latents
3. one AM transmission unit  -- rational + symbolic-expression coefficients, and a
                                covariance between one person's environment and another's
                                genes

Nothing here computes a covariance; that is the engine (task-20260804-151347). This file
only shows what the model object can *say*.
"""

import sympy as sp

import pathmgr as pm


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


if __name__ == "__main__":
    for model in (
        bivariate_regression(),
        relative_covariance_section1(),
        am_transmission_unit(),
    ):
        show(model)
