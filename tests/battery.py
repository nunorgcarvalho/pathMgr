"""The shared model battery.

One place where every model worth testing against lives, so a property that should hold for
all models -- above all "the two engines agree" -- can be stated once and picked up
automatically by anything added later. Add a model here and every battery-wide test covers it.

Models come from three places: written directly below, imported from the task-specific test
modules that introduced them, and (for the assortative-mating lineage, whose encoding is
subtle) imported from ``scripts/profile_ram.py`` so there is exactly one definition of it.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import sympy as sp

import pathmgr as pm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


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
        latent: g_m, g_f, g_1, g_2, s_1, s_2, e_1, e_2
        positive: V_A, V_E, V_K
        g_1 ~ 1/2*g_m + 1/2*g_f + s_1
        g_2 ~ 1/2*g_m + 1/2*g_f + s_2
        y_1 ~ g_1 + e_1
        y_2 ~ g_2 + e_2
        g_m ~~ V_A*g_m
        g_f ~~ V_A*g_f
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
# the battery
# ======================================================================================
def _imported_models() -> dict[str, pm.Model]:
    """Models defined by the tasks that introduced them, so there is one definition each."""
    from profile_ram import lineage  # scripts/profile_ram.py -- the AM copath encoding
    from test_am_spec_spike import am_pair_with_two_children
    from test_validation_models import bivariate_regression, relative_covariance_section1

    return {
        "bivariate regression": bivariate_regression(),
        "relative covariance S1": relative_covariance_section1(),
        "AM pair + two sibs": am_pair_with_two_children(),
        "AM lineage depth 1": lineage(1),
        "AM lineage depth 2": lineage(2),
    }


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
