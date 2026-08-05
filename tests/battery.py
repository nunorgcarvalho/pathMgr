"""The shared model battery.

One place where every model worth testing against lives, so a property that should hold for
all models -- above all "the two engines agree" -- can be stated once and picked up
automatically by anything added later. Add a model here and every battery-wide test covers it.

Models come from two places: written directly below, and imported from the task-specific test
modules that introduced them.

The dependency runs **outward from here**: ``scripts/scale_ram.py`` imports :func:`lineage`
from this module, not the other way round. It used to be reversed, which meant the registry
reached into a developer script for a model definition and the definition itself was never
reachable from the test suite alone.
"""

from __future__ import annotations

import random
from pathlib import Path

import sympy as sp

import pathmgr as pm

ROOT = Path(__file__).resolve().parent.parent


# ======================================================================================
# hand-checkable models
# ======================================================================================
def mediation_chain() -> pm.Model:
    """x -> m -> y plus a direct x -> y, so direct and indirect routes are separable."""
    return pm.from_text(
        """
        positive: V_x, V_m, V_y
        m ~ a*x
        y ~ b*m + c*x
        x ~~ V_x*x
        m ~~ V_m*m
        y ~~ V_y*y
        """,
        name="mediation chain",
    )


def confounded_pair() -> pm.Model:
    """Two predictors of y that covary -- the classic confounded-predictor case."""
    return pm.from_text(
        """
        positive: V_1, V_2, V_r
        y ~ b1*x1 + b2*x2
        x1 ~~ V_1*x1
        x2 ~~ V_2*x2
        x1 ~~ c12*x2
        y  ~~ V_r*y
        """,
        name="confounded pair",
    )


def latent_confounder() -> pm.Model:
    """An unobserved common cause of two observed variables."""
    return pm.from_text(
        """
        latent: u
        positive: V_u, V_1, V_2
        x ~ p*u
        y ~ q*u
        u ~~ V_u*u
        x ~~ V_1*x
        y ~~ V_2*y
        """,
        name="latent confounder",
    )


def common_factor() -> pm.Model:
    """One latent factor with three indicators."""
    return pm.from_text(
        """
        latent: f
        positive: V_f, V_1, V_2, V_3
        y1 ~ l1*f
        y2 ~ l2*f
        y3 ~ l3*f
        f  ~~ V_f*f
        y1 ~~ V_1*y1
        y2 ~~ V_2*y2
        y3 ~~ V_3*y3
        """,
        name="common factor",
    )


def siblings_sharing_a_genetic_factor() -> pm.Model:
    """Two sibs whose genetic values are each half their parents', plus segregation.

    The acceptance case for task-20260804-151348: the shared latent genetic factors are the
    parents, so the sib covariance must come out as chains through them.
    """
    return pm.from_text(
        """
        latent: g_m, g_p, g_1, g_2, s_1, s_2, e_1, e_2
        positive: V_A, V_E, V_K
        g_1 ~ 1/2*g_m + 1/2*g_p + s_1
        g_2 ~ 1/2*g_m + 1/2*g_p + s_2
        y_1 ~ g_1 + e_1
        y_2 ~ g_2 + e_2
        g_m ~~ V_A*g_m
        g_p ~~ V_A*g_p
        s_1 ~~ V_K*s_1
        s_2 ~~ V_K*s_2
        e_1 ~~ V_E*e_1
        e_2 ~~ V_E*e_2
        """,
        name="siblings sharing a genetic factor",
    )


def turning_point_with_ancestors() -> pm.Model:
    """w has two correlated parents and two children.

    The model that shows why the classical "no variable twice" rule cannot be enforced in the
    unstandardized RAM formulation: Cov[x, y] = q r Var[w], and Var[w]'s ancestral part is
    carried by chains that pass through w in both legs.
    """
    return pm.from_text(
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
        """,
        name="turning point with ancestors",
    )


def diamond() -> pm.Model:
    """Two distinct directed routes from x to y -- enumeration must find both."""
    return pm.from_text(
        """
        positive: V_x
        a ~ p1*x
        b ~ p2*x
        y ~ q1*a + q2*b
        x ~~ V_x*x
        a ~~ V_a*a
        b ~~ V_b*b
        y ~~ V_y*y
        """,
        name="diamond",
    )


def standardized_regression() -> pm.Model:
    """A standardized model, to check units are honoured rather than assumed."""
    return pm.from_text(
        """
        units: standardized to base generation (gen 0)
        y ~ b*x
        x ~~ x
        y ~~ (1 - b**2)*y
        """,
        name="standardized regression",
    )


def random_recursive(seed: int, n: int = 7) -> pm.Model:
    """A random DAG with latents, mixed directed and bidirected edges."""
    rng = random.Random(seed)
    m = pm.Model(f"random recursive (seed {seed})")
    names = [f"v{i}" for i in range(n)]
    for i, name in enumerate(names):
        m.add_var(name, latent=(i % 3 == 0))
    for j in range(n):
        for i in range(j):
            if rng.random() < 0.35:
                m.add_path(names[i], names[j], f"a{i}_{j}")
    for i, name in enumerate(names):
        m.add_variance(name, f"S{i}")
    for i in range(n):
        for j in range(i + 1, n):
            # only between exogenous variables: a bidirected edge touching an endogenous
            # variable is a disturbance covariance and can imply a non-PSD Sigma
            if rng.random() < 0.2 and not m.parents(names[i]) and not m.parents(names[j]):
                m.add_cov(names[i], names[j], f"C{i}_{j}")
    return m


# ======================================================================================
# the assortative-mating lineage
# ======================================================================================
def lineage(generations: int) -> pm.Model:
    """A founding couple plus `generations` descendant generations, at AM equilibrium.

    The **directed** encoding of assortment, and the reason it lives here rather than in the
    script that profiles it: how assortment is represented is not the obvious way, and getting
    it wrong is silent. A partner's genetic value enters as a directed path *from the focal
    individual's phenotype* (`y_c -> g_p` with coefficient `rho_g`), not as a bidirected
    `g_c <-> g_p` edge -- a bidirected edge is a *disturbance* covariance, and a child's genetic
    value is endogenous, so a bidirected edge there asserts that a fully determined disturbance
    covaries with something, which yields a Sigma that is not positive semi-definite.
    ``Model.validate()`` now reports that case as an error. The full account is in
    ``docs/assortment_representation_trap.md``.

    Being in the battery gets it two-engine agreement coverage; its substantive properties (the
    lineal-relative formula, equilibrium self-consistency, the mate correlation) are asserted in
    ``tests/test_am_lineage.py``. ``scripts/scale_ram.py`` imports it from here to profile it.
    """
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

    def child(tag: str, maternal: str, paternal: str) -> None:
        person(tag)
        m.add_var(f"s_{tag}", latent=True)
        m.add_variance(f"s_{tag}", V_K)
        m.add_path(f"g_{maternal}", f"g_{tag}", sp.Rational(1, 2))
        m.add_path(f"g_{paternal}", f"g_{tag}", sp.Rational(1, 2))
        m.add_path(f"s_{tag}", f"g_{tag}", 1)
        m.add_variance(f"e_{tag}", V_E)

    founder("0m")
    partner_of("0p", "0m")
    focal, partner = "0m", "0p"
    for t in range(1, generations + 1):
        child(f"{t}c", focal, partner)
        partner_of(f"{t}p", f"{t}c")
        focal, partner = f"{t}c", f"{t}p"
    return m


# ======================================================================================
# the battery
# ======================================================================================
def _imported_models() -> dict[str, pm.Model]:
    """Models defined by the tasks that introduced them, so there is one definition each."""
    from test_am_spec_spike import am_pair_with_two_children
    from test_copath import allele_level_pair, mated_pair, shared_partner

    from pathmgr.genetics import allele_motif, am_pedigree, g_level_model
    from test_validation_models import bivariate_regression, relative_covariance_section1

    root = Path(__file__).resolve().parent.parent
    return {
        "bivariate regression": bivariate_regression(),
        "relative covariance S1": relative_covariance_section1(),
        "AM pair + two sibs": am_pair_with_two_children(),
        "AM lineage depth 1": lineage(1),
        "AM lineage depth 2": lineage(2),
        # co-paths (task-20260804-173343)
        "co-path mated pair": mated_pair(),
        "co-path allele level": allele_level_pair(),
        "co-path shared partner": shared_partner(),
        # the same mated pair declared by CORRELATION rather than raw mu -- so the agreement
        # sweep covers the resolution path, not just the raw one
        "co-path mated pair (standardized)": pm.from_text(
            """
            latent: g_m, e_m, g_p, e_p
            positive: V_A, V_E
            y_m ~ g_m + e_m
            y_p ~ g_p + e_p
            g_m ~~ V_A*g_m
            e_m ~~ V_E*e_m
            g_p ~~ V_A*g_p
            e_p ~~ V_E*e_p
            y_m -- [rho_y]*y_p
            """,
            name="mated pair (standardized co-path)",
        ),
        "co-path AM example": pm.from_text(
            (root / "examples" / "am_equilibrium.pmg").read_text(), name="AM co-path"
        ),
        # the allele-level motif (task-20260804-173344); M=1 keeps the agreement sweep quick
        "allele motif M=1": allele_motif(n_variants=1).model,
        "allele motif M=2 two children": allele_motif(n_variants=2, n_children=2).model,
        # the unrolled pedigree (task-20260804-151350); depth 1 keeps the sweep quick
        "AM pedigree depth 1": g_level_model(am_pedigree(1)).model,
        "AM pedigree half-sibs": g_level_model(am_pedigree(1, half_sib_at=0)).model,
        "AM handwritten (superseded)": pm.from_text(
            (root / "tests" / "fixtures" / "am_equilibrium_handwritten.pmg").read_text(),
            name="AM handwritten",
        ),
    }


def copath_chain_of_three() -> pm.Model:
    """Three couples in a row, so chains must cross up to three distinct mating processes."""
    m = pm.Model("co-path chain of three", units=pm.Units.unstandardized())
    for v in ("V_A", "V_E"):
        m.declare(v, positive=True)
    V_A, V_E, rho_y = (m.sym(s) for s in ("V_A", "V_E", "rho_y"))
    mu = rho_y / (V_A + V_E)
    people = ("a", "b", "c", "d")
    for who in people:
        m.add_var(f"g_{who}", latent=True)
        m.add_var(f"e_{who}", latent=True)
        m.add_var(f"y_{who}")
        m.add_path(f"g_{who}", f"y_{who}", 1)
        m.add_path(f"e_{who}", f"y_{who}", 1)
        m.add_variance(f"g_{who}", V_A)
        m.add_variance(f"e_{who}", V_E)
    for i, (left, right) in enumerate(zip(people, people[1:])):
        m.add_copath(f"y_{left}", f"y_{right}", mu, process=f"couple{i}")
    return m


def couple_relatedness_cycle() -> pm.Model:
    """A x B mated; A also has child C, B also has child D; then C x D mate.

    Couple 2 has one member related to each member of couple 1, so the *couple-relatedness*
    graph has a cycle. This is the pedigree that separates a correct co-path implementation
    from a sequential rank-one update against the running Sigma: that construction lets a
    chain cross couple 1's co-path on BOTH legs of couple 2's update, which Sunde's
    one-co-path-per-mating-process rule forbids. It shows up as order dependence.
    See tests/test_copath.py::test_sequential_rank_one_updates_reuse_a_copath.
    """
    m = pm.Model("couple-relatedness cycle", units=pm.Units.unstandardized())
    for v in ("V_A", "V_E", "V_K"):
        m.declare(v, positive=True)
    V_A, V_E, V_K, rho_y = (m.sym(s) for s in ("V_A", "V_E", "V_K", "rho_y"))
    founders = ("A", "B", "P1", "P2")
    for who in founders + ("C", "D"):
        m.add_var(f"g_{who}", latent=True)
        m.add_var(f"e_{who}", latent=True)
        m.add_var(f"y_{who}")
        m.add_path(f"g_{who}", f"y_{who}", 1)
        m.add_path(f"e_{who}", f"y_{who}", 1)
        m.add_variance(f"e_{who}", V_E)
    for who in founders:
        m.add_variance(f"g_{who}", V_A)
    for child, first, second in [("C", "A", "P1"), ("D", "B", "P2")]:
        m.add_var(f"s_{child}", latent=True)
        m.add_variance(f"s_{child}", V_K)
        m.add_path(f"g_{first}", f"g_{child}", sp.Rational(1, 2))
        m.add_path(f"g_{second}", f"g_{child}", sp.Rational(1, 2))
        m.add_path(f"s_{child}", f"g_{child}", 1)
    mu = rho_y / (V_A + V_E)
    m.add_copath("y_A", "y_B", mu, process="c1")
    m.add_copath("y_C", "y_D", mu, process="c2")
    return m


def half_sibling_pedigree() -> pm.Model:
    """A x B -> full sibs E1, E2;  A x B2 -> H.  So E1 and H are half-sibs through A.

    The coordinator's validation pedigree for task-20260804-173343: two mating processes
    sharing an individual. Carries the in-law covariance rho_g^2 V_P between B and B2, who have
    no common ancestor at all, and the half-sib value that does NOT follow ((1+rho_g)/2)^d.
    """
    m = pm.Model("half-sibling pedigree", units=pm.Units.unstandardized())
    for v in ("V_A", "V_E", "V_K"):
        m.declare(v, positive=True)
    V_A, V_E, V_K, rho_y = (m.sym(s) for s in ("V_A", "V_E", "V_K", "rho_y"))
    for who in ("A", "B", "B2", "E1", "E2", "H"):
        m.add_var(f"g_{who}", latent=True)
        m.add_var(f"e_{who}", latent=True)
        m.add_var(f"y_{who}")
        m.add_path(f"g_{who}", f"y_{who}", 1)
        m.add_path(f"e_{who}", f"y_{who}", 1)
        m.add_variance(f"e_{who}", V_E)
    for who in ("A", "B", "B2"):
        m.add_variance(f"g_{who}", V_A)
    for child, other in [("E1", "B"), ("E2", "B"), ("H", "B2")]:
        m.add_var(f"s_{child}", latent=True)
        m.add_variance(f"s_{child}", V_K)
        m.add_path("g_A", f"g_{child}", sp.Rational(1, 2))
        m.add_path(f"g_{other}", f"g_{child}", sp.Rational(1, 2))
        m.add_path(f"s_{child}", f"g_{child}", 1)
    mu = rho_y / (V_A + V_E)
    m.add_copath("y_A", "y_B", mu, process="couple_AB")
    m.add_copath("y_A", "y_B2", mu, process="couple_AB2")
    return m


def all_models() -> dict[str, pm.Model]:
    """Every battery model, keyed by a readable name (used as the pytest test id)."""
    models: dict[str, pm.Model] = {
        "mediation chain": mediation_chain(),
        "confounded pair": confounded_pair(),
        "latent confounder": latent_confounder(),
        "common factor": common_factor(),
        "siblings": siblings_sharing_a_genetic_factor(),
        "turning point": turning_point_with_ancestors(),
        "diamond": diamond(),
        "standardized regression": standardized_regression(),
        "co-path chain of three": copath_chain_of_three(),
        "half-sibling pedigree": half_sibling_pedigree(),
        "couple-relatedness cycle": couple_relatedness_cycle(),
    }
    models.update(_imported_models())
    for seed in range(4):
        models[f"random recursive {seed}"] = random_recursive(seed)
    return models


def pairs(model: pm.Model, limit: int = 40) -> list[tuple[str, str]]:
    """Variable pairs to check, including self-pairs (variances). Capped for runtime."""
    names = model.names
    out = [(x, y) for i, x in enumerate(names) for y in names[i:]]
    if len(out) <= limit:
        return out
    # deterministic thinning that keeps every self-pair
    self_pairs = [(n, n) for n in names]
    cross = [p for p in out if p[0] != p[1]]
    step = max(1, len(cross) // max(1, limit - len(self_pairs)))
    return self_pairs + cross[::step]
