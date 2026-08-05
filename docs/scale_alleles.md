# Allele-motif scale

Measured, not assumed. Node count is `3M + 3` per individual plus `2M` segregation residuals for
the child, i.e. `11M + 9` for the one-child motif.

| variants M | nodes | build Sigma | cov() query | ops in Sigma |
|---|---|---|---|---|
| 1 | 20 | 0.027s | 0.0003s | 1544 |
| 2 | 31 | 0.085s | 0.0011s | 6935 |
| 3 | 42 | 0.170s | 0.0020s | 18236 |
| 4 | 53 | 0.294s | 0.0037s | 37621 |
| 6 | 75 | 0.752s | 0.0098s | 109259 |
| 8 | 97 | 1.477s | 0.0227s | 239081 |
| 10 | 119 | 2.567s | 0.0412s | 444319 |
| 12 | 141 | 4.107s | 0.0656s | 742205 |

## The practical limit

**Comfortable to M = 12** (Sigma under 5 s). This is more generous than
task-20260804-173344 assumed when it said "keep M small -- 2 or 3": the co-path enumeration adds
only one mating process regardless of M, so the cost grows with the node count rather than
combinatorially. Queries stay effectively free, since Sigma is cached per `model.revision`.

**But M = 2 is still the right size to work at**, for a reason that has nothing to do with speed:
every result in the validation table, including both cross-variant rows, is already visible at
M = 2, and the symbolic expressions stay small enough to read. Large M makes the same expressions
longer without making them say anything new -- and the `1/M` share result is exactly the argument
that the aggregate behaviour belongs in the `g`-level model instead.

Use M = 3 when you want a cross-variant sum with more than one term in it (the coordinator's
oracle uses M = 3); go higher only to confirm an M-dependence, as
`test_per_variant_inflation_is_one_over_M_of_the_total` does at M = 2, 3, 4.
