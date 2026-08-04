"""THE standing correctness property: the two engines agree, symbolically, on every model.

The RAM engine and the Wright tracer reach the same numbers by genuinely different routes --
a matrix identity versus explicit chain enumeration -- so agreement is strong evidence that
neither has a subtle bug. This is the project's principal defense, and it is why pathMgr is
worth writing rather than doing by hand.

Everything here is driven by ``tests/battery.py``. **Add a model there and it is covered by
this property automatically**; nothing needs editing in this file.

Comparison is `simplify(a - b) == 0` on sympy expressions, never string equality.
"""

import pytest
import sympy as sp

import pathmgr as pm

from battery import all_models, pairs

MODELS = all_models()


@pytest.mark.parametrize("name", sorted(MODELS))
def test_engines_agree_on_every_pair(name):
    """For every variable pair in every battery model: tracer total == RAM engine result."""
    model = MODELS[name]
    engine = pm.RAMEngine(model)
    tracer = pm.WrightTracer(model)

    mismatches = []
    for x, y in pairs(model):
        traced = tracer.cov(x, y)
        matrix = engine.cov(x, y)
        if sp.simplify(traced - matrix) != 0:
            mismatches.append(f"Cov[{x}, {y}]: traced {traced} != RAM {matrix}")
    assert not mismatches, f"{name}: " + "; ".join(mismatches)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_variances_agree(name):
    """Variances are the same enumeration with both endpoints equal -- checked separately."""
    model = MODELS[name]
    engine = pm.RAMEngine(model)
    tracer = pm.WrightTracer(model)
    for node in model.names:
        assert sp.simplify(tracer.var(node) - engine.var(node)) == 0, f"{name}: Var[{node}]"


@pytest.mark.parametrize("name", sorted(MODELS))
def test_decomposition_total_equals_the_sum_of_its_parts(name):
    """The itemized list must actually add up to the total it reports."""
    model = MODELS[name]
    tracer = pm.WrightTracer(model)
    for x, y in pairs(model, limit=12):
        decomposition = tracer.trace(x, y)
        by_hand = sp.expand(sum((c.contribution for c in decomposition), sp.Integer(0)))
        assert sp.simplify(decomposition.total - by_hand) == 0, f"{name}: Cov[{x}, {y}]"


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_battery_model_is_structurally_valid(name):
    """No battery model may carry a validate() error -- otherwise it proves nothing."""
    errors = [i for i in MODELS[name].validate() if i.severity == "error"]
    assert not errors, f"{name}: {[str(e) for e in errors]}"


@pytest.mark.parametrize("name", sorted(MODELS))
def test_chain_factors_multiply_to_the_contribution(name):
    """Each chain's itemized factors must multiply back to what it claims to contribute."""
    model = MODELS[name]
    tracer = pm.WrightTracer(model)
    for x, y in pairs(model, limit=8):
        for chain in tracer.trace(x, y):
            product = sp.expand(sp.Mul(*chain.factors))
            assert sp.simplify(product - chain.contribution) == 0, (
                f"{name}: {chain.path_string()}"
            )


def test_the_battery_is_not_accidentally_empty():
    """A guard on the guard: this property is worthless if the battery silently shrinks."""
    assert len(MODELS) >= 20
    assert any("AM" in name for name in MODELS)
    assert any(model.latent for model in MODELS.values())
    assert any(model.units.is_standardized for model in MODELS.values())
    # co-paths must be represented, and by more than one mating process, or the
    # multi-co-path composition rule would go unexercised by this property
    assert any(model.has_copaths for model in MODELS.values())
    assert any(len(model.mating_processes) >= 2 for model in MODELS.values())
