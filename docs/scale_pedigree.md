# Pedigree unroller scale

Measured on the standard lineage: a founding couple, two children per couple, the first of each
taking an outside partner and reproducing. One co-path per couple, `mu_t = rho_y/V_P(t)`.

**Headline: symbolic closed forms are limited to about 6 generations; numeric trajectories reach
12+.** That split is the thing to know, and it is not obvious from the model — see below.

## Targeted query, symbolic parameters

`cov(g_founder, g_deepest)` with `V_A(0)`, `V_E`, `rho_y` left as symbols.

| generations | nodes | couples | `cov(g, g)` | `cov(y, y)` | ops in result |
|---|---|---|---|---|---|
| 1 | 14 | 1 | 0.005s | 0.000s | 8 |
| 2 | 25 | 2 | 0.036s | 0.002s | 32 |
| 3 | 36 | 3 | 0.041s | 0.009s | 110 |
| 4 | 47 | 4 | 0.114s | 0.041s | 360 |
| 5 | 58 | 5 | 0.476s | 0.222s | 1158 |
| 6 | 69 | 6 | 2.470s | 1.367s | 3700 |
| 8 | 91 | 8 | 131s | 71s | 37424 |

**Comfortable to 6 generations.** The result's operation count roughly triples per generation, and
that growth — not the node count — is what stops it.

## Targeted query, numeric parameters

`AMParameters(values={"V_A0": 0.4, "V_E": 0.6, "rho_y": 0.3})` resolves the per-generation recursion
at build time, so every coefficient is a number.

| generations | `cov(g, g)` | value |
|---|---|---|
| 6 | 0.093s | 0.01275215 |
| 8 | 0.175s | 0.00406939 |
| 10 | 0.597s | 0.00129913 |
| 12 | 12.2s | 0.00041480 |

**Comfortable past 10 generations.** The co-path *sequence count* grows with depth either way — that
part is inherent, since each generation's couple is linked to the next by real covariance. What
changes is the cost of each term: a numeric term costs nothing to accumulate, a symbolic one grows.

## What this means for task-20260804-151351

**Not binding.** Trajectories are computed numerically, and the numeric ceiling comfortably covers
the six-to-ten generations Sunde et al. report and the coordinator's measurement confirms (within 1%
of equilibrium by generation 6, still 1e-3 off at 8). Use `AMParameters(values=...)` for trajectory
work.

**Do keep symbolic work shallow.** A symbolic closed form for a relative pair is practical to about
6 generations. That is not a real constraint either, because the interesting symbolic statements —
partners, parent–offspring, siblings, grandparents, half-sibs — all live within two or three
generations. Depth is for watching convergence, and convergence is numeric.

## Query the entry, do not build the matrix

The single largest factor. Building the **whole** `Sigma` pays an `O(n^2)` outer product per co-path
sequence; a **single entry** pays `O(1)`. Measured on this same lineage, before the targeted path
existed and after:

| generations | full `Sigma` | targeted `cov()` |
|---|---|---|
| 3 | 1.3s | 0.041s |
| 4 | 13.8s | 0.114s |
| 5 | 145s | 0.476s |

A 300× difference at five generations. `RAMEngine.cov` therefore computes the entry directly and
never materialises `Sigma` when co-paths are present. `sigma()` remains the primary object and is
still correct — it is just the wrong tool at depth. **Ask for the covariances you want.**

## Keep results in per-generation symbols

`V_A(t)` and `mu_t = rho_y/V_P(t)` are carried as symbols rather than substituted forward, so every
co-path coefficient stays one ratio of two symbols however deep the pedigree is. Resolving them back
to `V_A(0)` — `UnrolledModel.recursion_substitutions()` — is where the expression swell actually
lives, and it is deferred until a caller asks, on the one scalar they care about. Same discipline as
keeping coefficients symbolic and substituting numbers last.
