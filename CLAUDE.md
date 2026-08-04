# pathMgr

**Path Manager** — a python package for **symbolic path analysis**. Three jobs:

1. **Specify** a model of dependencies between variables: directed paths and bidirected
   covariances, with coefficients that are sympy symbols, numbers, or expressions.
2. **Compute** the covariance or correlation between any two variables — including latent
   and intermediate ones — on demand, symbolically.
3. **Draw** the corresponding path diagram, exportable as TikZ (for LaTeX writeups) and as
   a raster image.

## Scope boundary — read this before adding a feature

pathMgr is a **symbolic derivation and visualization tool**. It does **not** fit models to
data: no estimation, no optimization, no fit statistics, no data input at all. Do not drift
into re-implementing lavaan or OpenMx. If a proposed feature needs a dataset, it does not
belong here.

The other boundary is internal and one-directional: `pathmgr.core` is a **generic**
path-analysis engine that knows nothing about genetics, and `pathmgr.genetics` is a layer on
top of it. `genetics` imports from `core`; `core` never imports from `genetics`, and no
genetics concept (`V_A`, `rho_g`, transmission, pedigree) may leak into `core`.
`render` likewise stays out of `core`'s computation path.

## Setup and test

```bash
source ~/alkes_nuno/.venv_py13/bin/activate
cd /n/groups/price/nuno/pathMgr
pip install -e ".[dev]"      # add ",render" once rendering lands
pytest                       # from the repo root
```

Core dependency is **sympy**. Import name is `pathmgr` (lowercase, PEP 8); the project and
repo are `pathMgr`.

## Module layout

```
src/pathmgr/
  core/                 generic path analysis — no genetics, no rendering
    model.py            Model, Variable, DirectedEdge, BidirectedEdge  ← the spec object
    units.py            Units: the scale a model is stated on + its reference population
    symbols.py          SymbolRegistry: symbol identity + safe expression parsing
    ram.py              closed-form covariance engine            (task-…-151347)
    tracing.py          Wright chain-enumeration engine          (task-…-151348)
  render/               diagram output — never imported by core
    tikz.py             TikZ export                              (task-…-151349)
    raster.py           PNG/SVG export                           (task-…-151349)
  genetics/             the genetics layer, built on core
    pedigree.py         unroll generations into a path model     (task-…-151350)
    am.py               AM dynamics + equilibrium fixed point    (task-…-151351)
tests/
  test_model.py               specification API unit tests
  test_validation_models.py   two hand-encoded models, checked to round-trip
examples/
  spec_demo.py                runnable tour of the specification API
```

## The specification API

```python
import pathmgr as pm

m = pm.Model("bivariate regression")
m.declare("V_1", positive=True)          # declare assumptions BEFORE first use
m.add_vars("x1", "x2", "y")              # observed
m.add_var("f", latent=True)              # latent
m.add_path("x1", "y", "b1")              # directed: y regresses on x1 with coeff b1
m.add_variance("x1", "V_1")              # bidirected self-edge
m.add_cov("x1", "x2", "c12")             # bidirected
m.assume("V_A + V_E", 1)                 # side relation, not an edge
A, S, F, order = m.ram()                 # symbolic RAM matrices
print(m.describe())
```

Conventions, all load-bearing:

- **Directed edge** `a -> b` with coefficient `c` means *`b` regresses on `a`*, and becomes
  `A[b, a] = c`.
- **Bidirected edge** `a <-> b` is a **disturbance** covariance, `S[a, b] = S[b, a]`. On an
  exogenous variable `a <-> a` is its variance; on an **endogenous** variable it is the
  *residual* variance — never the total, which the model implies rather than states. This
  catches people out; say so in docstrings.
- **Latent vs observed** matters twice: `F` selects observed rows, and diagrams draw latents
  as circles, observed variables as boxes. Latents remain legitimate query targets.
- `Model` is a **mutable builder** with `.copy()`. It is not persistent/immutable because
  the driving use case accumulates one growing graph (a pedigree, generation by generation).
  `model.revision` increments on every structural change — **engines must key their caches
  on it**, since the symbolic `(I - A)^-1` is the expensive step.
- `m.assume(...)` records side relations (`rho_g = rho_y * h2_eq`, `V_A + V_E = 1`) that are
  *not* edges. Engines may substitute them in, but only opt-in, never silently.

## Gotchas that shaped the design

**Symbol names collide with sympy builtins, silently.** The notation of statistical
genetics is a minefield: plain `sympify` turns `pi` (realized relatedness) into 3.14159…,
`E` (environment) into Euler's number, `beta` and `gamma` into special functions, `I` into
the imaginary unit; `S`, `N`, `O`, `Q` are singletons. `SymbolRegistry.parse` therefore
parses with a curated global namespace of constructors and functions only, so every bare
identifier becomes a plain `Symbol`. The reserved names are exactly the keys of
`symbols._SAFE_GLOBALS`.

**Symbol identity depends on assumptions.** In sympy `Symbol('V_A') != Symbol('V_A',
positive=True)` — distinct objects whose expressions will not cancel. Each `Model` owns a
registry so one name is always one Symbol. `declare()` must precede first use; re-declaring
differently raises rather than silently bifurcating. New names get `real=True`.

**"Standardized" is not a complete statement.** Wright's classic tracing rules assume
unit-variance variables, but the genetics is written in unstandardized components (`V_A`,
`V_E`, `V_K`) on a scale that *shifts every generation* under assortative mating. So
`Units.standardized()` refuses to construct without a named reference population, and every
result must carry the model's units so a returned expression is never scale-ambiguous.

**Infinite ancestral regress.** At AM equilibrium the ancestral graph extends back forever.
A naive tracer will not terminate or will silently truncate. Finite-generation unrolling
terminates by construction; equilibrium must come from an **explicit fixed-point solve**,
never from "unroll a lot". A tracer must fail loudly rather than truncate.

**Symbolic blowup.** sympy expressions grow fast with pedigree depth. Simplify deliberately
at defined points; do not assume `simplify()` is cheap. For a recursive (acyclic) model,
prefer forward substitution in topological order over a general symbolic matrix inverse.

## Correctness properties

**The two-engine property (standing, non-negotiable).** Every covariance is computed two
ways — the RAM identity `Sigma = F (I - A)^-1 S (I - A)^-T F^T`, and explicit Wright chain
enumeration — and they must agree **symbolically on every model**. Keep this a property
test over a corpus of models. It is the project's main defense against subtle tracing bugs,
and it is why the package is worth writing rather than doing by hand.

**Ground truth already exists.** `popstatgenwriteups`'
`writeups/statistical_genetics/relative_covariance/relative_covariance.tex` carries
hand-derived boxed results to validate against — notably `rho_g = rho_y h2_eq`,
`V_A_eq = V_A0 / (1 - rho_g)`, the collateral result `V_A_eq ((1+rho_g)/2)^d`, and the
lineal result `V_A_eq ((1+rho_y)/2)((1+rho_g)/2)^(d-1)`. Published cross-checks: Sunde et
al. 2024/2025 Nat Commun, in `~/thesisMgr/corpus/literature/` (read-only). **If pathMgr and
the writeup disagree, the disagreement is a finding** — surface it to the user rather than
silently conforming to either.

## Conventions

- Tests live in `tests/`, run with `pytest` from the repo root, and stay green.
- Project-repo commits do **not** reference thesisMgr task IDs.
- Tracked as thesis work in `~/thesisMgr` (project `pathMgr`); see
  `~/thesisMgr/.claude/profiles/worker-pathMgr.md`.
