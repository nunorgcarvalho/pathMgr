# The assortment-representation trap

**If you read one document in this repository, read this one.** It is the most expensive thing
this project learned, it is silent when you get it wrong, and it has bitten the same construction
more than once.

It used to be the last section of a *generated* timing report (`docs/profile_ram.md`, now
`docs/scale_ram.md`), which meant the project's central piece of knowledge lived inside a
profiling script's string literals and was filed under a name that reads like a benchmark. It is
hand-written prose and belongs in a hand-written file.

## The trap

Mates' genetic values are correlated, so the obvious encoding is a bidirected edge
`g_m <-> g_p`. That is correct **only while both are exogenous** — as in a founding
pair. Once a mate is a child in the pedigree, their genetic value is endogenous, and a bidirected
edge is a covariance between *disturbances*, not between variables. Asserting that an endogenous
variable's disturbance covaries with something, when that disturbance is fully determined by its
parents, produces a `Sigma` that is **not positive semi-definite** — an implied correlation above
1 — with nothing else to signal the mistake.

The first version of `scripts/scale_ram.py` did exactly that, and the resulting covariances decayed
as `2^-d` with no `(1 + rho_g)` accumulation at all, silently disagreeing with the writeup.
`Model.validate()` now reports it as an **error**.

## Encoding 1: the directed representation

`tests/battery.py::lineage` uses it. Assortment enters as a **directed path from the focal
individual's phenotype to the partner's components**,

    y_focal -> g_partner    with coefficient  rho_g
    y_focal -> e_partner    with coefficient  rho_y V_E / V_P

with the partner's residual variances reduced to keep `Var[g] = V_A_eq` and `Var[e] = V_E`. This
reproduces the writeup exactly, and it makes `Cov[e_partner, g_focal] = rho_g V_E` — the term
behind the lineal/collateral asymmetry — fall out automatically instead of needing its own edge.

Its cost is that it is an *equilibrium-only* device: the residual variances are chosen against a
fixed `V_A_eq`, and the direction of the paths is an artefact of the encoding rather than a causal
claim. It also breaks the symmetry between partners, which is fine for a lineage and awkward for a
general pedigree.

## Encoding 2: the co-path (preferred)

The general answer, and why `CoPath` exists as a third edge type. A co-path is covariance
attributable to **matching**: it induces covariance without causing variance, and — the decisive
property — it propagates **backward to the causes** of the matched variables, which is exactly
what a bidirected edge cannot do. See `src/pathmgr/core/model.py`, Sunde et al. 2025
(Nat Commun, Supplementary Notes 1 and 3), and `tests/test_copath.py`, whose decisive case is
`test_copath_reaches_the_causes_but_a_bidirected_edge_does_not`.

`Cov[a, b] = mu * Var[a] * Var[b]`, so `mu_t = rho_y / V_P(t)` — **generation-indexed**. That was
the second half of the same trap: a co-path coefficient copied between generations is wrong wherever
`V_P` has moved.

**That half of the trap is now closed by construction.** Declare a co-path by the correlation it
induces — `y_m -- [rho_y]*y_p` — and the engines derive `mu` themselves, resolving co-paths in
dependency order so each one is computed against the variances that actually obtain at its own
depth. Nobody writes `mu_t` any more, so nobody can copy it from the wrong generation. Getting there
needed one non-obvious fact: a co-path does *not* change the variance of the pair it matches, but it
*does* change their descendants' — which is the entire content of the AM dynamics — so a downstream
co-path resolved against co-path-free variances is understated by the accumulated assortment gain.
Ordering by graph depth makes every dependency already known when a co-path's turn comes.

## The three encodings side by side

| | exogenous mates | endogenous mates | reaches the causes | valid off equilibrium |
|---|---|---|---|---|
| bidirected `g_m <-> g_p` | correct | **non-PSD, silently** | no | n/a |
| directed from `y_focal` | correct | correct | yes | no (equilibrium-only) |
| co-path `y_m -- [rho_y]*y_p` | correct | correct | yes | yes, and `mu` is derived per generation |
| co-path with raw `mu` | correct | correct | yes | only if you index `mu` by generation yourself |

Both correct encodings are kept, and both are in the battery, so the two-engine agreement sweep
covers them. `tests/fixtures/am_equilibrium_handwritten.pmg` retains the **superseded** encoding
on purpose: `tests/test_copath.py::test_the_superseded_encoding_is_wrong_in_exactly_the_documented_way`
pins the size and shape of the error, so this account cannot rot into folklore.

## How to not fall in

- Before adding a bidirected edge, ask whether either endpoint has parents. If it does, stop.
- A covariance you can *derive* should not be *specified*. If the model already implies it, adding
  an edge for it double-counts.
- `Model.validate()` catches this specific case. Run it. It is cheap and it is the whole point.
