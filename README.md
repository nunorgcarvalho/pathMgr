# pathMgr

**Symbolic path analysis.** Write down a model of how variables depend on each other; get the
covariance or correlation between *any* two of them — including latent and intermediate ones — as an
algebraic expression, and get the path diagram to go with it.

```python
import pathmgr as pm

m = pm.from_text("""
    latent: u
    positive: V_u, V_x
    x ~ p*u
    y ~ c*x + q*u
    u ~~ V_u*u
    x ~~ V_x*x
    y ~~ V_y*y
""")

pm.RAMEngine(m).cov("x", "y")
# V_u*c*p**2 + V_u*p*q + V_x*c

print(pm.WrightTracer(m).trace("x", "y"))
# Cov[x, y]  (3 chains, unstandardized)
#   x <-> x -> y            =  V_x * c
#   x <- u <-> u -> y       =  p * V_u * q
#   x <- u <-> u -> x -> y  =  p * V_u * p * c   [revisits x]
#   total                   =  V_u*c*p**2 + V_u*p*q + V_x*c
```

The second output is the reason this package exists: not the covariance, but the **decomposition** —
which routes contribute, and how much each one carries. Any two variables, any depth, exact algebra.

## Scope

pathMgr is a **derivation and visualization** tool. It does **not** fit models to data: no
estimation, no optimizer, no fit statistics, no way to hand it a dataset. If a feature would need a
dataset, it does not belong here.

What it does have:

- **Two front-ends** — a builder API and a text syntax, the second a thin layer over the first.
- **Two independent engines** — a closed-form RAM engine ($\Sigma = F(I-A)^{-1}S(I-A)^{-\top}F^\top$)
  and a Wright chain-enumerating tracer. They share no code, and that they agree on every model in
  the test battery is the project's principal correctness check. The tracer gives you the
  decomposition; the engine handles cycles and scales further.
- **Three edge types** — directed, bidirected, and **co-paths**, the last for covariance arising
  from *matching* (assortative mating). A co-path is not a bidirected edge and the difference is not
  cosmetic; see [`docs/assortment_representation_trap.md`](docs/assortment_representation_trap.md).
- **Diagrams** — TikZ for a LaTeX writeup, PNG/SVG/PDF via matplotlib, and the figure the package
  exists to make: a single Wright chain highlighted on the diagram, captioned with its own algebra.
- **A genetics layer** on top of the generic core: allele-level transmission, pedigrees under
  assortative mating, generation dynamics, and the equilibrium fixed point solved symbolically.

## Install

```bash
pip install -e ".[render,notebook,dev]"     # core needs only sympy
pytest
```

`render` is matplotlib, `notebook` is the Jupyter tooling, `dev` is pytest. The core is importable
with none of them.

## Learn it here

→ **[`examples/pathmgr_tour.ipynb`](examples/pathmgr_tour.ipynb)** ←

The notebook is the documentation. It works one example through end to end in four acts: the generic
core on a model small enough to **check by hand**, then co-paths and the mistake they prevent, then
a pedigree under assortative mating where hand-checking is hopeless, then getting a figure into your
own document. It is committed with its outputs, so GitHub renders it without you running anything.

Also worth knowing about:

| | |
|---|---|
| [`examples/spec_demo.py`](examples/spec_demo.py) | runnable tour of the whole API, wider than the notebook, no narrative |
| [`examples/make_figures.py`](examples/make_figures.py) | regenerates [`examples/figures/`](examples/figures/) |
| [`docs/assortment_representation_trap.md`](docs/assortment_representation_trap.md) | the project's central lesson — read before encoding assortment |
| `docs/scale_*.md` | measured limits: pedigree depth, variant count, how long $\Sigma$ takes |
| [`tests/battery.py`](tests/battery.py) | the shared model registry; add a model and the agreement sweep picks it up |
| [`CLAUDE.md`](CLAUDE.md) | the conventions, and why each is load-bearing |

Import name is `pathmgr` (lowercase); the project and repo are `pathMgr`.
