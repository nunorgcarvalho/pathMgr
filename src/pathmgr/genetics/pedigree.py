"""Pedigree scaffolding, and unrolling an assortative-mating model forward from a random base.

Rather than reasoning about an infinitely-deep ancestral graph, **unroll a finite number of
generations**. That terminates by construction and sidesteps the infinite regress entirely.
Equilibrium is a separate, explicit fixed-point solve (task-20260804-151351) -- never "unroll a lot
and hope".

Structure vs content
--------------------
:class:`Pedigree` is *only* the scaffolding -- who exists, who is mated to whom, who descends from
whom, and which generation each individual is in. It knows nothing about genetics. The builders
then walk it: :func:`g_level_model` puts ``y = g + e`` on each individual, and
:func:`allele_level_model` puts the per-variant machinery of :mod:`pathmgr.genetics.alleles` on the
same scaffolding. Neither duplicates the pedigree logic.

How assortment is encoded
-------------------------
**One co-path per couple, between the partners' phenotypes**, and nothing else. ``rho_g`` is
*derived*, never asserted. The pre-co-path workaround -- writing induced covariances directly
between ``g`` and ``e`` -- only holds while those are root nodes, which stops being true from
generation 1 onward.

**The co-path coefficient is generation-indexed**: ``mu_t = rho_y / V_P(t)``, because
``Cov[y_m, y_f] = mu V_P^2`` and ``V_P`` grows every generation under assortment. Holding ``mu``
fixed instead of ``rho_y`` is a *different model*; Sunde et al. 2024 (Supp. Note 2.1) hold
``rho_y`` constant and note the choice is open, so :class:`AMParameters` makes it an explicit
option rather than an accident.

``V_P(t)`` and ``V_A(t)`` are carried as **symbols per generation** rather than substituted in.
That keeps every expression small -- ``mu_t`` is one ratio of two symbols -- and, more importantly,
it makes the generation indexing *visible in the model* instead of buried in a build-time
computation. The recursion relating them is recorded via :meth:`Model.assume`, so it can be checked
against what the engine derives rather than being assumed true.

What the finite unroll does NOT satisfy
---------------------------------------
``((1 + rho_g)/2)^d`` is an **equilibrium-only** form. On a finite unroll from a randomly mating
base it does not hold, and a mismatch must not be reconciled by adjusting it. Two exact statements
during disequilibrium, both indexed to the **parents'** generation:

    Cov[full sibs]   = Cov[g_parent, g_offspring] = V_A(t) (1 + rho_g(t)) / 2
    Cov[y_parent, y_offspring]                    = V_A(t) (1 + rho_y) / 2

For a lineal pair spanning several generations it is a **chained product using each generation's
own** ``rho_g``, not a power of a single one. At equilibrium ``V_A`` stops changing, the generation
index becomes invisible, and the writeup's boxed forms are recovered. Using the *offspring's*
``V_A`` instead of the parents' is wrong by ~4e-4 -- small enough to look like numerical noise,
which is why ``tests/test_pedigree.py`` pins the indexing at low ``t`` where the error is largest,
not near equilibrium where it would pass either way.

``V_K`` is held constant at ``V_A(0)/2``
---------------------------------------
An explicit parameter, not an implicit assumption. This is the model Sunde et al. 2024 (Supp. Note
2.2) and Section 2 of ``relative_covariance.tex`` both use, and it reproduces
``V_A(t+1) = V_A(0)/2 + V_A(t)(1 + rho_g(t))/2`` exactly. The allele-level refinement
``Var(s) = 1/4 - c/2`` (see :mod:`pathmgr.genetics.alleles`) is a *departure* from that model
rather than a correction to an error in it, of size ``O(rho_y V_A^2 / M)`` -- noted, not built for.
:attr:`AMParameters.segregation_variance` is where it would go.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from ..core.model import Model
from ..core.units import Units
from ..render.layout import Layout

__all__ = [
    "AMParameters",
    "Couple",
    "Individual",
    "Pedigree",
    "am_pedigree",
    "g_level_model",
]


# ======================================================================================
# scaffolding -- structure only, no genetics
# ======================================================================================
@dataclass(frozen=True)
class Individual:
    """One person: which generation, and who their parents are (None for a founder)."""

    key: str
    generation: int
    mother: str | None = None
    father: str | None = None
    #: a short human tag for diagrams and error messages, e.g. "child of the founding couple"
    role: str = ""

    @property
    def is_founder(self) -> bool:
        return self.mother is None and self.father is None

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(p for p in (self.mother, self.father) if p is not None)


@dataclass(frozen=True)
class Couple:
    """A mating pair. ``generation`` is the PARENTS' generation, which is the index that matters."""

    mother: str
    father: str
    generation: int

    @property
    def key(self) -> str:
        return f"couple_{self.mother}_{self.father}"


@dataclass
class Pedigree:
    """Who exists and how they are related. No genetics here at all."""

    individuals: dict[str, Individual] = field(default_factory=dict)
    couples: list[Couple] = field(default_factory=list)

    # -- construction ------------------------------------------------------------------
    def add(self, key: str, generation: int, mother=None, father=None, role="") -> str:
        if key in self.individuals:
            raise ValueError(f"individual {key!r} already in the pedigree")
        self.individuals[key] = Individual(key, generation, mother, father, role)
        return key

    def mate(self, mother: str, father: str) -> Couple:
        for who in (mother, father):
            if who not in self.individuals:
                raise KeyError(f"unknown individual {who!r}")
        generation = self.individuals[mother].generation
        if self.individuals[father].generation != generation:
            raise ValueError(
                f"{mother!r} and {father!r} are in different generations; a couple spans one"
            )
        couple = Couple(mother, father, generation)
        self.couples.append(couple)
        return couple

    # -- queries -----------------------------------------------------------------------
    @property
    def n_generations(self) -> int:
        return max((i.generation for i in self.individuals.values()), default=-1) + 1

    def generation(self, t: int) -> tuple[str, ...]:
        return tuple(k for k, i in self.individuals.items() if i.generation == t)

    def children_of(self, couple: Couple) -> tuple[str, ...]:
        return tuple(
            k
            for k, i in self.individuals.items()
            if i.mother == couple.mother and i.father == couple.father
        )

    def partners_of(self, who: str) -> tuple[str, ...]:
        out = []
        for couple in self.couples:
            if couple.mother == who:
                out.append(couple.father)
            elif couple.father == who:
                out.append(couple.mother)
        return tuple(out)

    def ancestors_of(self, who: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.individuals[who].parents)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.individuals[current].parents)
        return seen

    def relationship(self, a: str, b: str) -> str:
        """A name for the relationship, or ``"unrelated"``.

        Exposed **by pedigree position**, computed from the structure -- never stored. Degree alone
        is not sufficient under assortment (a grandparent-grandchild pair and an avuncular pair are
        both degree 2 and have different covariances), and half-siblings are a genuine third case,
        so the three that must never be conflated -- lineal, collateral, half-sib -- are each
        derived from an explicit structural test rather than from a distance count.
        """
        if a == b:
            return "self"
        if b in self.partners_of(a):
            return "partners"
        ancestors_a, ancestors_b = self.ancestors_of(a), self.ancestors_of(b)
        if b in ancestors_a or a in ancestors_b:
            return "lineal"
        shared_parents = set(self.individuals[a].parents) & set(self.individuals[b].parents)
        if len(shared_parents) == 2:
            return "full siblings"
        if len(shared_parents) == 1:
            return "half siblings"
        if ancestors_a & ancestors_b:
            return "collateral"
        # no common ancestor, but assortment can still correlate them: two people who each had
        # children with the same third person, for instance
        if set(self.partners_of(a)) & set(self.partners_of(b)):
            return "co-parents-in-law"
        return "unrelated"

    def generation_order(self, t: int) -> tuple[str, ...]:
        """Individuals of generation ``t``, ordered so that **partners are adjacent**.

        Without this a non-breeding sibling can sit between a couple, and the co-path joining
        them is then drawn straight through that sibling's node. Deterministic: couples in the
        order they were mated, each partner pair together, then everyone else.
        """
        members = set(self.generation(t))
        ordered: list[str] = []
        for couple in self.couples:
            if couple.generation != t:
                continue
            for who in (couple.mother, couple.father):
                if who in members:
                    members.discard(who)
                    ordered.append(who)
        return tuple(ordered + sorted(members))

    def layout(self, x_gap: float = 2.6, y_gap: float = 3.4) -> Layout:
        """Generations as rows, individuals as columns -- the natural pedigree layout."""
        positions: dict[str, tuple[float, float]] = {}
        for t in range(self.n_generations):
            members = self.generation_order(t)
            offset = (len(members) - 1) / 2.0
            for column, key in enumerate(members):
                positions[key] = ((column - offset) * x_gap, -t * y_gap)
        return Layout(positions, x_gap=x_gap, y_gap=y_gap)

    def describe(self) -> str:
        lines = [f"Pedigree: {len(self.individuals)} individuals, {len(self.couples)} couples"]
        for t in range(self.n_generations):
            members = sorted(self.generation(t))
            lines.append(f"  generation {t}: {', '.join(members)}")
        for couple in self.couples:
            children = ", ".join(sorted(self.children_of(couple))) or "-"
            lines.append(f"  {couple.mother} x {couple.father} (gen {couple.generation}) -> {children}")
        return "\n".join(lines)


def am_pedigree(
    n_generations: int,
    children_per_couple: int = 2,
    breeding_children: int = 1,
    half_sib_at: int | None = None,
) -> Pedigree:
    """A lineage under monogamous assortative mating, unrolled ``n_generations`` deep.

    Generation 0 is a founding couple. Each generation, the first ``breeding_children`` children of
    each couple take an unrelated outside partner and reproduce; the rest are leaves, which is what
    provides sibling, avuncular and cousin pairs to query.

    ``half_sib_at`` gives one individual in that generation a **second** partner and a child by
    them, deliberately violating monogamy. Half-siblings are a third case that follows neither the
    lineal nor the collateral formula, and they are the sharpest check that a chain can cross
    co-paths from two different mating processes.

    >>> pedigree = am_pedigree(2)
    >>> pedigree.n_generations
    3
    >>> pedigree.relationship("i0_0", "i1_0")
    'lineal'
    """
    if n_generations < 1:
        raise ValueError(f"need at least one descendant generation, got {n_generations}")
    if children_per_couple < 1:
        raise ValueError(f"need at least one child per couple, got {children_per_couple}")
    if breeding_children < 1 or breeding_children > children_per_couple:
        raise ValueError(
            f"breeding_children must be between 1 and children_per_couple, got {breeding_children}"
        )

    pedigree = Pedigree()
    counters: dict[int, int] = {}

    def new_key(generation: int) -> str:
        index = counters.get(generation, 0)
        counters[generation] = index + 1
        return f"i{generation}_{index}"

    mother = pedigree.add(new_key(0), 0, role="founder")
    father = pedigree.add(new_key(0), 0, role="founder")
    active = [pedigree.mate(mother, father)]

    for t in range(1, n_generations + 1):
        next_active: list[Couple] = []
        for couple in active:
            children = [
                pedigree.add(new_key(t), t, couple.mother, couple.father, role="child")
                for _ in range(children_per_couple)
            ]
            if t < n_generations:
                for child in children[:breeding_children]:
                    partner = pedigree.add(new_key(t), t, role="outside partner")
                    next_active.append(pedigree.mate(child, partner))
        active = next_active

    if half_sib_at is not None:
        if not 0 <= half_sib_at < n_generations:
            raise ValueError(
                f"half_sib_at must name a generation with children, 0..{n_generations - 1}"
            )
        # give the first breeding individual of that generation a second partner and a child
        parent = sorted(k for k in pedigree.generation(half_sib_at) if pedigree.partners_of(k))[0]
        second = pedigree.add(new_key(half_sib_at), half_sib_at, role="second partner")
        couple = pedigree.mate(parent, second)
        pedigree.add(
            new_key(half_sib_at + 1),
            half_sib_at + 1,
            couple.mother,
            couple.father,
            role="half sibling",
        )
    return pedigree


# ======================================================================================
# the g-level model
# ======================================================================================
@dataclass
class AMParameters:
    """The assortative-mating parameters, and the modelling choices that are easy to hide.

    ``hold`` says which quantity is held constant across generations. ``"rho_y"`` follows Sunde et
    al. 2024 (Supp. Note 2.1) and the writeup: the phenotypic correlation between partners is the
    thing that stays fixed, so the co-path coefficient ``mu_t = rho_y / V_P(t)`` must be
    recomputed every generation as ``V_P`` grows. ``"mu"`` holds the co-path coefficient fixed
    instead, which is a **different model** -- available so the difference can be demonstrated
    rather than stumbled into.
    """

    #: which of `rho_y` / `mu` is constant across generations
    hold: str = "rho_y"
    #: numeric values for ``V_A(0)``, ``V_E`` and ``rho_y``. Supplying them resolves the
    #: per-generation recursion at build time so every coefficient is a number.
    #: **This is what makes deep pedigrees tractable** -- see ``docs/profile_pedigree.md``: the
    #: co-path sequence count grows with depth either way, but a numeric term costs nothing to
    #: accumulate while a symbolic one grows. Symbolic closed forms are limited to a few
    #: generations; numeric trajectories are not.
    values: dict | None = None
    #: segregation variance. Held constant at V_A(0)/2 -- see the module docstring for why that is
    #: the model being validated against rather than an approximation being smuggled in.
    segregation_variance: sp.Expr | None = None
    base_additive_variance: str = "V_A0"
    environmental_variance: str = "V_E"
    phenotypic_correlation: str = "rho_y"

    def __post_init__(self) -> None:
        if self.hold not in ("rho_y", "mu"):
            raise ValueError(f"hold must be 'rho_y' or 'mu', got {self.hold!r}")


@dataclass
class UnrolledModel:
    """A built pedigree model, plus the per-generation symbols needed to ask about it."""

    model: Model
    pedigree: Pedigree
    parameters: AMParameters
    #: ``V_A(t)`` per generation, as symbols
    V_A: tuple[sp.Symbol, ...]
    #: ``V_P(t) = V_A(t) + V_E`` per generation
    V_P: tuple[sp.Expr, ...]
    #: ``rho_g(t) = rho_y V_A(t) / V_P(t)`` per generation
    rho_g: tuple[sp.Expr, ...]
    #: ``mu_t`` per generation, the co-path coefficient actually used
    mu: tuple[sp.Expr, ...]
    rho_y: sp.Symbol
    V_E: sp.Symbol
    V_A0: sp.Symbol
    V_K: sp.Expr

    def g(self, who: str) -> str:
        return f"g_{who}"

    def e(self, who: str) -> str:
        return f"e_{who}"

    def y(self, who: str) -> str:
        return f"y_{who}"

    def s(self, who: str) -> str:
        return f"s_{who}"

    def recursion(self, t: int) -> sp.Eq:
        """``V_A(t+1) = V_A(0)/2 + V_A(t)(1 + rho_g(t))/2`` -- the relation to check against."""
        return sp.Eq(self.V_A[t + 1], self.V_K + self.V_A[t] * (1 + self.rho_g[t]) / 2)

    def recursion_substitutions(self, upto: int | None = None) -> dict[sp.Symbol, sp.Expr]:
        """``V_A(t) -> its value in terms of V_A(0)``, resolved forward from the base population.

        Applying these turns a result expressed in per-generation symbols into one in the base
        parameters. Expression size grows quickly with ``t``, which is the cost this task warns
        about -- see ``docs/profile_pedigree.md``.
        """
        if self.parameters.values:
            return {}  # already resolved at build time
        upto = len(self.V_A) - 1 if upto is None else upto
        values: dict[sp.Symbol, sp.Expr] = {self.V_A[0]: self.V_A0}
        for t in range(upto):
            current = self.V_A[t].subs(values) if t else self.V_A0
            rho_g = self.rho_y * current / (current + self.V_E)
            values[self.V_A[t + 1]] = sp.together(self.V_K + current * (1 + rho_g) / 2)
        return values

    def layout(self, **kwargs) -> Layout:
        """Diagram coordinates: generations as rows, with each individual's g/e/y stacked."""
        base = self.pedigree.layout(**kwargs)
        positions: dict[str, tuple[float, float]] = {}
        for key, (x, y) in base.positions.items():
            positions[self.g(key)] = (x - 0.55, y + 1.05)
            positions[self.e(key)] = (x + 0.75, y + 1.05)
            positions[self.y(key)] = (x, y)
            if self.model.has_var(self.s(key)):
                positions[self.s(key)] = (x - 1.75, y + 1.05)
        return Layout(positions, x_gap=base.x_gap, y_gap=base.y_gap)


def g_level_model(
    pedigree: Pedigree,
    parameters: AMParameters | None = None,
    name: str | None = None,
) -> UnrolledModel:
    """Put ``y = g + e`` on every individual of ``pedigree``, with assortment as co-paths.

    Everything about the assortment is one co-path per couple. ``rho_g``, the induced correlation
    between partners' genetic values, is derived by the engine and never written into the model.
    """
    parameters = parameters or AMParameters()
    model = Model(
        name or f"AM pedigree, {pedigree.n_generations - 1} descendant generation(s)",
        units=Units.unstandardized(),
    )
    V_A0 = model.declare(parameters.base_additive_variance, positive=True)
    V_E = model.declare(parameters.environmental_variance, positive=True)
    rho_y = model.declare(parameters.phenotypic_correlation, real=True)

    n = pedigree.n_generations
    # per-generation symbols: keeps every mu_t a single ratio, and keeps the generation index
    # visible in the model instead of buried in a build-time computation
    V_A = tuple(
        V_A0 if t == 0 else model.declare(f"V_A_{t}", positive=True) for t in range(n)
    )
    V_K = (
        parameters.segregation_variance
        if parameters.segregation_variance is not None
        else V_A0 / 2
    )

    if parameters.values:
        # resolve the recursion forward to numbers, so every coefficient is numeric
        base = {V_A0: sp.nsimplify(parameters.values.get("V_A0", V_A0), rational=True),
                V_E: sp.nsimplify(parameters.values.get("V_E", V_E), rational=True),
                rho_y: sp.nsimplify(parameters.values.get("rho_y", rho_y), rational=True)}
        resolved = [base[V_A0]]
        segregation = sp.nsimplify(V_K.subs(base), rational=True)
        for _ in range(n - 1):
            current = resolved[-1]
            phenotypic = current + base[V_E]
            induced = base[rho_y] * current / phenotypic
            resolved.append(sp.nsimplify(segregation + current * (1 + induced) / 2, rational=True))
        V_A = tuple(resolved)
        V_E = base[V_E]
        rho_y = base[rho_y]
        V_A0 = base[V_A0]
        V_K = segregation

    V_P = tuple(V_A[t] + V_E for t in range(n))
    rho_g = tuple(rho_y * V_A[t] / V_P[t] for t in range(n))
    if parameters.hold == "rho_y":
        mu = tuple(rho_y / V_P[t] for t in range(n))
    else:  # a DIFFERENT model: the co-path coefficient, not the correlation, is what is fixed
        fixed = model.declare("mu", real=True) if not parameters.values else (
            sp.nsimplify(parameters.values.get("mu", rho_y / V_P[0]), rational=True)
        )
        mu = tuple(fixed for _ in range(n))

    for key, person in pedigree.individuals.items():
        t = person.generation
        model.add_var(f"g_{key}", latent=True, label=f"$g_{{{key[1:]}}}$")
        model.add_var(f"e_{key}", latent=True, label=f"$e_{{{key[1:]}}}$")
        model.add_var(f"y_{key}", label=f"$y_{{{key[1:]}}}$")
        model.add_path(f"g_{key}", f"y_{key}", 1)
        model.add_path(f"e_{key}", f"y_{key}", 1)
        model.add_variance(f"e_{key}", V_E)
        if person.is_founder:
            # an outside partner joining at generation t comes from that generation's population
            model.add_variance(f"g_{key}", V_A[t])
        else:
            # transmission: g_o = (g_m + g_f)/2 + s_o, with V_K held constant
            model.add_var(f"s_{key}", latent=True, label=f"$s_{{{key[1:]}}}$")
            model.add_variance(f"s_{key}", V_K)
            model.add_path(f"g_{person.mother}", f"g_{key}", sp.Rational(1, 2))
            model.add_path(f"g_{person.father}", f"g_{key}", sp.Rational(1, 2))
            model.add_path(f"s_{key}", f"g_{key}", 1)

    # THE ONLY cross-couple statement: one co-path per couple, at that couple's generation
    for couple in pedigree.couples:
        model.add_copath(
            f"y_{couple.mother}",
            f"y_{couple.father}",
            mu[couple.generation],
            process=couple.key,
        )

    unrolled = UnrolledModel(
        model=model,
        pedigree=pedigree,
        parameters=parameters,
        V_A=V_A,
        V_P=V_P,
        rho_g=rho_g,
        mu=mu,
        rho_y=rho_y,
        V_E=V_E,
        V_A0=V_A0,
        V_K=V_K,
    )
    if not parameters.values:
        for t in range(n - 1):
            model.assume(unrolled.recursion(t))
    return unrolled
