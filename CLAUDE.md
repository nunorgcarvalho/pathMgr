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
    text.py             text front-end: from_text / to_text, a thin layer over the builder
    ram.py              closed-form covariance engine  ← RAMEngine
    tracing.py          Wright chain-enumeration engine ← WrightTracer, Decomposition
                        (also the SPECIFICATION for co-path semantics)
  render/               diagram output — never imported by core
    layout.py           node placement: explicit coords + layered auto fallback
    style.py            the drawing conventions, shared by both back ends
    tikz.py             TikZ export + pdflatex compilation
    raster.py           PNG/SVG/PDF via matplotlib (imported lazily)
  genetics/             the genetics layer, built on core
    pedigree.py         unroll generations into a path model     (task-…-151350)
    am.py               AM dynamics + equilibrium fixed point    (task-…-151351)
tests/
  conftest.py                 shared helpers: ram_sigma wrapper, canonical() comparison
  battery.py                  THE shared model battery — add models here
  test_agreement.py           the standing property: RAM engine == Wright tracer
  test_model.py               specification API unit tests
  test_text.py                text front-end: grammar, errors, builder equivalence
  test_ram.py                 RAM engine: battery, queries, units, robustness
  test_tracing.py             tracer: enumeration, decomposition, LaTeX, limits
  test_validation_models.py   two hand-encoded models, checked to round-trip
  test_am_spec_spike.py       AM unit vs. the writeup's boxed results
  test_copath.py              co-paths: Sunde's rules, the allele-level decisive test
  test_relative_covariance_section1.py   Section 1 derived by the engine
scripts/
  profile_ram.py              RAM engine profile + the AM copath reference model
examples/
  spec_demo.py                runnable tour: both front-ends, engine, tracer, co-paths
  make_figures.py             produces docs/figures/ — including the highlighted chain
  am_equilibrium.pmg          the AM model as text, using a co-path
  am_equilibrium_handwritten.pmg   superseded encoding, kept as a regression fixture
docs/
  profile_ram.md              measured timings + the assortment-representation trap
  figures/                    generated diagrams (tikz + png + pdf)
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
- **Co-path** `a -- b` is covariance from **matching**, a third edge type — see the co-path
  section below. Not interchangeable with a bidirected edge in either direction.
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

## The text front-end

`pm.from_text(...)` / `m.to_text()` (or `Model.from_text`). It is a **thin layer over the
builder** — the parser makes builder calls and nothing else, so the two front-ends cannot
diverge in meaning. Keep it that way: no feature may exist in text that the builder lacks.
`tests/test_am_spec_spike.py::test_text_front_end_produces_the_identical_model` is the
guard, comparing the full AM model built both ways.

```
units: unstandardized                  # or: standardized to base generation (gen 0)
latent: g_i, e_i                       # everything else is observed
observed: z                            # only needed for isolated nodes
positive: V_A, V_E                     # sympy assumptions on SYMBOLS
label: g_i = $g_i$                     # rendering label
assume: V_A + V_E = 1                  # side relation, not an edge

y_i ~ g_i + e_i                        # directed; coefficient 1 implied
y   ~ b1*x1 + b2*x2
g_o ~ 1/2*g_m + 1/2*g_f + s_o          # exact rationals, not floats
g_c ~ ((1 + rho_g)/2)*g_p              # parenthesise a compound coefficient

x1  ~~ V_1*x1                          # a variance
g_i ~~ (V_A*pi_ij)*g_j                 # a covariance, expression value

y_m -- mu*y_f                          # a co-path (matching); see the co-path section
S_m -- mu2*Sx_p [couple0]              # [name] names the mating process
```

Directives may appear in any order (they are all processed before the equations), so a
`latent:` line can sit at the bottom. Variables are created on first use in a variable
position. Every parse error carries its line number and the offending line
(`TextSyntaxError`).

**The one rule to internalise: every right-hand-side term must end in a variable name.**
In `(V_A*pi_ij)*g_j` the trailing identifier is the variable and everything before the final
`*` is the coefficient. There is deliberately **no** bare-variance shorthand, because
`g_i ~~ V_A` cannot be disambiguated — is `V_A` the variance of `g_i`, or a variable it
covaries with? Write `g_i ~~ V_A*g_i`. As a backstop, a name used both as a variable and as
a coefficient symbol is rejected outright.

## The RAM engine

```python
eng = pm.RAMEngine(model)
eng.sigma()                       # FULL Sigma over all nodes — the primary object
eng.sigma_observed()              # (F Sigma F^T, observed names) — a view, not primary
eng.cov("g_i", "g_j")             # any pair: latent-latent, latent-observed, observed-observed
eng.var("y_i"); eng.corr("y_i", "y_j")
eng.explain("y_i", "y_j")         # covariance + both variances + UNITS
eng.check_standardization()        # variables whose implied variance isn't 1
```

- **The full matrix is primary.** The whole point of pathMgr is covariances between *any* two
  variables, so `F` is applied only when asked. Never bake the observed filter into the core.
- **Recursive models never form `(I - A)^-1`.** Two topological sweeps build `T = S (I-A)^-T`
  and then `Sigma = (I-A)^-1 T` entry by entry, exploiting Σ's symmetry to halve the work.
  Cyclic models fall back to the explicit inverse — a genuine capability the chain tracer in
  [tracing.py](src/pathmgr/core/tracing.py) cannot match, since a feedback loop has
  infinitely many Wright chains but a closed-form geometric sum. A structurally singular
  `(I - A)` raises `SingularModelError`.
- **`Sigma` is cached against `model.revision`** and rebuilt on any structural change. Build
  the whole pedigree first, *then* query — mutating mid-query pays the build again.
- **Simplification is explicit, never reflexive.** The only automatic step is `expand` per
  entry as it is built (cheap, and what makes terms cancel). `cov` defaults to
  `form="expanded"` — canonical, and the form to compare term-by-term against the tracer.
  `corr` defaults to `"simplified"` because an unsimplified ratio is unreadable. `"raw"` and
  `"factored"` are also available.
- **Side relations are opt-in.** `apply_assumptions=True` substitutes only unambiguous
  `Symbol = expr` relations. A relation like `V_A + V_E = 1` could be solved either way, so it
  is skipped unless you name the symbol to eliminate: `apply_assumptions=["V_E"]`.
- See [docs/profile_ram.md](docs/profile_ram.md) for measured timings and the budget for
  unrolling generations. Regenerate with `python scripts/profile_ram.py N`.

## The Wright tracer

```python
d = pm.WrightTracer(model).trace("y_i", "y_j")
print(d)                          # itemized chains + total
d.total                           # same expression the RAM engine returns
d.to_latex()                      # align* for a writeup; style="tabular" also available
[c.directed_edges() for c in d]   # for highlighting a chain on a diagram
```

A chain is `x <- ... <- u  <->  v -> ... -> y`: backward to an ancestor, **exactly one**
bidirected edge (`u == v` is a variance), then forward. Its value is the product of the
directed coefficients and the bidirected value; the covariance is the sum over chains. This is
the RAM identity written out term by term.

**One classical rule is deliberately not enforced: "no variable may appear twice in a chain".**
It belongs to the *standardized* formulation, where a chain may turn around anywhere using
`Var = 1` and tracing stops at exogenous variables. Here `S` entries are **disturbance**
(co)variances, so a turning point is written explicitly as `u <-> u`, and for an endogenous
variable that is only its residual — the rest of its variance arrives via chains continuing
back to its ancestors, which necessarily visit the turning-point variable in *both* legs.
Dropping those would lose `Var[w]`'s ancestral part entirely. A node may repeat *across* legs;
within a leg it cannot, which in a DAG is automatic. `Chain.revisits` reports them, since
that's what a reader hand-checking against a textbook will query.

**Cyclic models cannot be traced** — a feedback loop has infinitely many chains. Raises
`UntraceableModelError` pointing at `RAMEngine`, never silently truncating. Since co-paths compose
with the inverse fallback too, **`RAMEngine` strictly dominates `WrightTracer` in capability**: any
model the tracer handles, the engine handles, and it also handles cyclic and cyclic-with-co-path
models the tracer cannot. What the tracer alone gives is the *decomposition*. Enumeration is also
capped by `max_chains` (`ChainLimitError`) so a wide lattice fails loudly instead of hanging.

## Co-paths (assortative mating)

A **third edge type**, distinct from both arrows, following Sunde et al. 2025 Nat Commun
(Supplementary Notes 1 and 3 — in `~/thesisMgr/corpus/literature/`, read-only, and the authority
here). A co-path denotes covariance attributable to **matching**, inducing covariance *without
causing variance*.

```python
m.add_copath("y_m", "y_f", "mu")                      # process defaults to the pair
m.add_copath("S_m", "Sx_f", "mu_prime", process="couple0")   # cross-trait: same process
```
```
y_m -- mu*y_f                    # text grammar: an arrowless line
S_m -- mu2*Sx_p [couple0]        # [name] names the mating process
```

**Why it is not a bidirected edge, and cannot be emulated by one.** Matching induces correlation
among *all the causes* of the matched variables, so the association propagates **backward** up
the graph. An `S` entry is a disturbance covariance and does not. So `S[y_m, y_f] = c` gets
`Cov[y_m, y_f]` right and `Cov[g_m, g_f] = 0`. The decisive check, in `tests/test_copath.py`:
split each parent's `g` into allele nodes and confirm
`Cov[z_mat_m, z_mat_f] = beta^2 rho_y / (4 V_P)` **without it being specified**. A bidirected
edge gives 0 there.

**The coefficient is not the correlation.** Sunde Eq. (1) is `Cov[a,b] = mu Var[a] Var[b]`, so a
target phenotypic correlation `rho_y` needs `mu = rho_y / V_P` — **generation-indexed**, because
`V_P` grows under assortment. With `V_P = 1` in the base generation `mu = rho_y` and every
first-generation test passes while later ones are wrong by a power of `V_P`.

**Tracing rules.** A chain is `[segment] -- [segment] -- ...`, each segment a standard valid
chain. A chain cannot start or end with a co-path, and may use **at most one co-path per mating
process**. Co-paths from *different* processes may combine, and must — that is what accumulates
`((1+rho_g)/2)^d` across generations. Implementing "one co-path per chain" would silently
truncate every multi-generation result to first order in `rho_g`.

**Do not replace the enumeration with sequential rank-one updates against a running `Sigma`.**
That construction (`Sigma += mu*(outer(Sigma e_a, Sigma e_b) + transpose)`, once per co-path)
looks equivalent and agrees wherever each couple pairs with an unrelated partner — including on
the half-sibling pedigree. It breaks once the **couple-relatedness graph has a cycle**: using the
running `Sigma` on *both* legs lets one mating process be crossed twice in a chain. The symptom
is order dependence, so it cannot be right in both orders.
`test_sequential_rank_one_updates_reuse_a_copath` pins this on `A x B` mated, `A` and `B` each
having a child by someone else, and those two children mating.

**The RAM form.** Bundling each leg into a `Sigma_0` entry, a co-path sequence contributes a
scalar times an **outer product** of one `Sigma_0` column and one `Sigma_0` row — which is
exactly why it reaches the causes, where a bidirected edge contributes `B` columns instead. The
engine sums over sequences of distinct-process co-paths, pruning wherever the connecting
`Sigma_0` entry is zero. **`Sigma_0 (I - C Sigma_0)^-1` is NOT the closed form**: it re-uses the
same co-path, adding `rho_y^3 V_P + rho_y^5 V_P + ...` on a single mated pair
(0.3495 against the correct 0.3180). `test_the_geometric_series_form_overcounts` pins that, so
nobody "simplifies" the enumeration into a matrix inverse later. Restricted to distinct
co-paths the sum is a simple-walk enumeration, which has no closed form in general — hence
`CoPathLimitError` rather than a silent truncation.

## Rendering

```python
from pathmgr.render import to_tikz, to_standalone, write_pdf, to_image, Layout, DiagramStyle

to_tikz(model, layout=Layout({"y_m": (0, 0), ...}))   # snippet to paste into a writeup
write_pdf(model, "fig.pdf")                            # compiles it (pdflatex)
to_image(model, "fig.png", legend=True)                # matplotlib: png / svg / pdf
to_tikz(model, highlight=chain)                        # ONE traced chain, emphasised + captioned
```

**The three edge types must be visually unmistakable**, and that is a correctness concern, not a
cosmetic one: a reader who takes a co-path for a covariance arrow applies the wrong tracing rules
and gets a wrong answer by hand. So the co-path differs on three axes at once and stays
distinguishable in greyscale:

| edge | shape | arrowheads | weight |
|---|---|---|---|
| directed `a -> b` | straight | one | thin |
| bidirected `a <-> b` | **curved** | **two** | thin |
| co-path `a -- b` | straight | **none** | **thick, own colour** |

`to_image(..., legend=True)` draws a key. A variance is a self-loop, suppressible with
`DiagramStyle(show_variances=False)`.

**Highlighting a traced chain is the figure this project exists to produce** — the diagram and
the covariance in one object. Pass any `Chain` from the tracer; its edges are emphasised, the rest
faded, and the chain's own `tex_path()` becomes the caption. `examples/make_figures.py` produces
it for the allele-level model, where the highlighted chain *is* the proof that a co-path reaches
the causes.

**Layout**: explicit coordinates are the reliable path and what pedigrees should use
(`pedigree_layout` takes a generation per individual). The layered auto fallback exists so an
arbitrary model renders at all — its bar is "legible and correct", and it does one barycentre pass
to cut edge crossings. Do not sink time into making it beautiful.

**Dependencies, deliberately minimal.** The TikZ output needs only `tikz` + `shapes.geometric` +
`xcolor`. It does **not** use `arrows.meta` or `standalone.cls`, both of which are absent from a
plain TinyTeX install — arrowheads are TikZ's built-in `->`/`<->` and `to_standalone` defaults to
`article` with the page sized to the drawing. Colours are emitted as `\definecolor` declarations,
because a raw `#RRGGBB` in TikZ trips `Illegal parameter number` (`#` is a TeX special).
`write_pdf` calls `pdflatex` directly, not `latexmk`, since the system perl on compute nodes is
incomplete and breaks it. matplotlib is an optional extra (`pip install -e ".[render]"`) imported
lazily, so `import pathmgr` never needs a drawing dependency — a test asserts that.

## The correctness property — do not let this rot

`tests/test_agreement.py` asserts the two engines agree symbolically (`simplify(a - b) == 0`)
on every variable pair of every model in `tests/battery.py`. **Add a model to the battery and
it is covered automatically** — nothing in the agreement test needs editing. This is the
project's principal defense against subtle tracing bugs and the reason the package is worth
writing rather than doing by hand. When you add a feature or a model, add it to the battery.

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
Also: sympy will not always reduce what looks obvious — `sqrt((V_A + V_E)**2)` survives
`simplify()` even with both symbols declared positive. Compare squared, or `factor` inside the
root.

**Bidirected edges on endogenous variables are a trap.** A bidirected edge is a covariance
between *disturbances*. For two exogenous variables that coincides with their covariance,
which is why the distinction is easy to miss. For an endogenous variable it does not, and if
that variable has no disturbance variance of its own its disturbance is identically zero and
cannot covary with anything — so asserting that it does yields a `Sigma` that is **not
positive semi-definite**, with an implied correlation above 1 and no other complaint.
`Model.validate()` reports this as an error, and the warning case (endogenous but with a
disturbance variance) too. This bit while writing the RAM profile: encoding assortative mating
as `g_mother <-> g_father` is right for a founding couple and wrong the moment a mate is
someone's child. The fix is to make assortment a **directed path from the partner's
phenotype** — see the last section of [docs/profile_ram.md](docs/profile_ram.md), which
matters for [pedigree.py](src/pathmgr/genetics/pedigree.py).

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
