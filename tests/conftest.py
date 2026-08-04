"""Shared test helpers.

``ram_sigma`` was a four-line throwaway spike while task-20260804-151346 designed the
specification API and no engine existed. Task-20260804-151347 landed the real engine, so it
is now a thin wrapper over :class:`pathmgr.RAMEngine` -- meaning the specification tests that
used it now exercise the engine too, which is the point of keeping it rather than deleting it.
"""

from __future__ import annotations

import sympy as sp

import pathmgr as pm


def ram_sigma(model: pm.Model) -> tuple[sp.Matrix, dict[str, int]]:
    """``(Sigma over observed variables, name -> row index)``, from the real RAM engine."""
    sigma, names = pm.RAMEngine(model).sigma_observed()
    return sigma, {name: i for i, name in enumerate(names)}


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
