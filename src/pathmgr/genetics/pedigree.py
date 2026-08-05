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
``Cov[y_m, y_p] = mu V_P^2`` and ``V_P`` grows every generation under assortment. Holding ``mu``
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
    maternal: str | None = None
    paternal: str | None = None
    #: a short human tag for diagrams and error messages, e.g. "child of the founding couple"
    role: str = ""

    @property
    def is_founder(self) -> bool:
        return self.maternal is None and self.paternal is None

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(p for p in (self.maternal, self.paternal) if p is not None)


@dataclass(frozen=True)
class Couple:
    """A mating pair. ``generation`` is the PARENTS' generation, which is the index that matters."""

    maternal: str
    paternal: str
    generation: int

    @property
    def key(self) -> str:
        return f"couple_{self.maternal}_{self.paternal}"


@dataclass
class Pedigree:
    """Who exists and how they are related. No genetics here at all."""

    individuals: dict[str, Individual] = field(default_factory=dict)
    couples: list[Couple] = field(default_factory=list)

    # -- construction ------------------------------------------------------------------
    def add(self, key: str, generation: int, maternal=None, paternal=None, role="") -> str:
        if key in self.individuals:
            raise ValueError(f"individual {key!r} already in the pedigree")
        self.individuals[key] = Individual(key, generation, maternal, paternal, role)
        return key

    def mate(self, maternal: str, paternal: str) -> Couple:
        for who in (maternal, paternal):
            if who not in self.individuals:
                raise KeyError(f"unknown individual {who!r}")
        generation = self.individuals[maternal].generation
        if self.individuals[paternal].generation != generation:
            raise ValueError(
                f"{maternal!r} and {paternal!r} are in different generations; a couple spans one"
            )
        couple = Couple(maternal, paternal, generation)
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
            if i.maternal == couple.maternal and i.paternal == couple.paternal
        )

    def partners_of(self, who: str) -> tuple[str, ...]:
        out = []
        for couple in self.couples:
            if couple.maternal == who:
                out.append(couple.paternal)
            elif couple.paternal == who:
                out.append(couple.maternal)
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
        """Individuals of generation ``t``, ordered for drawing. Deterministic.

        Two properties, both of which stop edges being drawn through unrelated nodes:

        **Children sit near their parents.** Each individual is keyed by the mean position of its
        parents in the row above, so a transmission edge is short and near-vertical instead of
        spanning the figure. Without this, a child could be placed at the far end of its row from
        its own parents, and the resulting diagonal crossed everything in between -- no amount of
        edge bending can rescue that, because there is nowhere clear to bend to.

        **Partners are adjacent.** Otherwise a non-breeding sibling sits between a couple and the
        co-path joining them is drawn straight through that sibling.

        An individual with no parents in the model (an outside partner marrying in) has no
        ancestral position of its own, so it inherits its partner's.
        """
        members = list(self.generation(t))
        if not members:
            return ()

        above = {key: i for i, key in enumerate(self.generation_order(t - 1))} if t else {}

        def ancestral_key(who: str) -> float:
            parents = [above[p] for p in self.individuals[who].parents if p in above]
            if parents:
                return sum(parents) / len(parents)
            partners = [
                p for p in self.partners_of(who) if self.individuals[p].generation == t
            ]
            for partner in partners:  # marry-in: take the position of whoever they married
                inherited = [above[p] for p in self.individuals[partner].parents if p in above]
                if inherited:
                    return sum(inherited) / len(inherited) + 0.5
            return float("inf")  # unplaceable: park at the end, deterministically

        members.sort(key=lambda who: (ancestral_key(who), who))

        # now pull each partner next to their mate, keeping the ancestral order otherwise
        ordered: list[str] = []
        placed: set[str] = set()
        for who in members:
            if who in placed:
                continue
            ordered.append(who)
            placed.add(who)
            for partner in sorted(self.partners_of(who)):
                if partner not in placed and self.individuals[partner].generation == t:
                    ordered.append(partner)
                    placed.add(partner)
        return tuple(ordered)

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
            lines.append(f"  {couple.maternal} x {couple.paternal} (gen {couple.generation}) -> {children}")
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

    maternal = pedigree.add(new_key(0), 0, role="founder")
    paternal = pedigree.add(new_key(0), 0, role="founder")
    active = [pedigree.mate(maternal, paternal)]

    for t in range(1, n_generations + 1):
        next_active: list[Couple] = []
        for couple in active:
            children = [
                pedigree.add(new_key(t), t, couple.maternal, couple.paternal, role="child")
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
            couple.maternal,
            couple.paternal,
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
    #: **This is what makes deep pedigrees tractable** -- see ``docs/scale_pedigree.md``: the
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
        about -- see ``docs/scale_pedigree.md``.
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

    #: half-width of one individual's block of nodes, in cm. Generation spacing must exceed twice
    #: this or neighbouring people's nodes interleave -- which is how an edge belonging to one
    #: person ends up drawn through a *different* person's node.
    BLOCK_HALF_WIDTH = 1.35
    #: vertical offset of ``g`` above the individual's ``y``
    BLOCK_HEIGHT = 1.7
    #: vertical offset of the ``s``/``e`` pair. Equal to ``BLOCK_HEIGHT``, i.e. level with
    #: ``g`` -- a flat row works once the block is wide enough and the generations are far
    #: enough apart; dropping them below ``g`` only narrows the transmission corridor.
    FLANK_HEIGHT = 1.7

    def layout(self, x_gap: float = 4.6, y_gap: float = 5.5) -> Layout:
        """Diagram coordinates: generations as rows, each individual a self-contained block.

        Two things about the block are load-bearing, and both were found by measuring crossings
        rather than by eye.

        **It is narrower than the spacing between people**, so no edge belonging to one individual
        traverses another's nodes. The previous layout put ``s`` at ``x - 1.75`` while the
        neighbour's ``e`` sat at ``x - 1.85`` -- 0.1 cm apart, and each one's edge ran through the
        other's node.

        **The generations are far enough apart that the transmission corridor is clear.**
        Transmission is ``g -> g`` between generations, so those edges must descend past the
        parent's own row. Making that descent steep is what keeps them off the parent's ``s`` and
        ``e``; the defaults here were found by measuring crossings over a grid of spacings, and at
        ``x_gap=4.6, y_gap=5.5`` the layout is crossing-free with a single edge needing a bend.

            s     g     e
                  |
                  y

        Tightening the spacing does not merely look worse -- it reintroduces crossings, so the
        defaults are load-bearing rather than taste. ``tests/test_render.py`` asserts the count is
        zero at three, four and five generations.
        """
        base = self.pedigree.layout(x_gap=x_gap, y_gap=y_gap)
        half, top, flank = self.BLOCK_HALF_WIDTH, self.BLOCK_HEIGHT, self.FLANK_HEIGHT
        positions: dict[str, tuple[float, float]] = {}
        for key, (x, y) in base.positions.items():
            positions[self.y(key)] = (x, y)
            positions[self.g(key)] = (x, y + top)
            positions[self.e(key)] = (x + half, y + flank)
            if self.model.has_var(self.s(key)):
                positions[self.s(key)] = (x - half, y + flank)
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
        # canonicalise through the model, so `unrolled.V_K` is the symbol the MODEL registered.
        # A symbol handed in by a caller carries its own sympy assumptions; the registry gives it
        # the model's, and the two do not compare equal -- so returning the caller's object would
        # make `subs(unrolled.V_K, ...)` silently do nothing.
        model.expr(parameters.segregation_variance)
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
            # transmission: g_o = (g_m + g_p)/2 + s_o, with V_K held constant
            model.add_var(f"s_{key}", latent=True, label=f"$s_{{{key[1:]}}}$")
            model.add_variance(f"s_{key}", V_K)
            model.add_path(f"g_{person.maternal}", f"g_{key}", sp.Rational(1, 2))
            model.add_path(f"g_{person.paternal}", f"g_{key}", sp.Rational(1, 2))
            model.add_path(f"s_{key}", f"g_{key}", 1)

    # THE ONLY cross-couple statement: one co-path per couple, at that couple's generation
    for couple in pedigree.couples:
        model.add_copath(
            f"y_{couple.maternal}",
            f"y_{couple.paternal}",
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
