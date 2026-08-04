"""Shared test helpers.

``ram_sigma`` is a deliberate **spike**, not the engine: four lines of sympy so the
specification tests can check *meaning* rather than only matrix shape, before
task-20260804-151347 exists. It has none of what the real engine needs -- no caching on
``model.revision``, no topological forward substitution, no unit awareness, no controlled
simplification -- and should be deleted once the engine lands, with these tests repointed
at it.
"""

from __future__ import annotations

import sympy as sp

import pathmgr as pm


def ram_sigma(model: pm.Model) -> tuple[sp.Matrix, dict[str, int]]:
    """``(Sigma_observed, name -> row index)`` via ``F (I - A)^-1 S (I - A)^-T F^T``."""
    A, S, F, _ = model.ram()
    IA = (sp.eye(A.rows) - A).inv()
    Sigma = sp.expand(F * IA * S * IA.T * F.T)
    return Sigma, {name: i for i, name in enumerate(model.observed)}


def canonical(model: pm.Model) -> dict:
    """An order-insensitive, comparable summary of a model's content.

    Used to assert that two ways of specifying the same model (builder vs. text, or a
    to_text/from_text round trip) really do agree. Deliberately ignores variable insertion
    order and the model's name, and compares symbols by name plus assumptions so that a
    `positive: V_A` directive is not mistaken for a plain real symbol.
    """
    return {
        "units": model.units,
        "observed": frozenset(model.observed),
        "latent": frozenset(model.latent),
        "labels": {v.name: v.label for v in model.variables if v.label is not None},
        "paths": frozenset((e.src, e.dst, sp.srepr(e.coeff)) for e in model.directed_edges),
        "covs": frozenset((e.a, e.b, sp.srepr(e.value)) for e in model.bidirected_edges),
        "assumptions": frozenset(sp.srepr(eq) for eq in model.assumptions),
    }
