"""Symbol registry and safe expression parsing.

Two sympy landmines this module exists to defuse.

**1. Name collisions with sympy builtins.** Plain ``sympify`` resolves bare identifiers
against sympy's namespace, and the notation of statistical genetics collides with it
badly: ``pi`` (realized relatedness) becomes the number 3.14159..., ``E`` (environment)
becomes Euler's number, ``beta`` and ``gamma`` become special functions, ``I`` becomes the
imaginary unit, ``S``/``N``/``O``/``Q`` are sympy singletons. Silently. So we parse with a
curated global namespace containing *only* functions -- every bare identifier becomes a
plain ``Symbol``.

**2. Symbol identity depends on assumptions.** In sympy ``Symbol('V_A') !=
Symbol('V_A', positive=True)``: they are distinct objects and expressions built from both
will not cancel. Every model therefore owns a registry, so one name always maps to exactly
one Symbol. Assumptions must be declared before first use, and re-declaring differently is
an error rather than a silent bifurcation.
"""

from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

# Constructors and functions only -- deliberately no constants (pi, E, I, oo) and no special
# functions (beta, gamma, zeta), so that every bare identifier in a user expression parses to
# a plain Symbol. `Symbol`/`Integer`/`Float`/`Rational` must be present because the
# auto_symbol / auto_number transformations emit calls to them. The keys below are therefore
# the only reserved names in a coefficient expression; extend as real needs appear.
_SAFE_GLOBALS: dict[str, object] = {
    "Symbol": sp.Symbol,
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "log": sp.log,
    "Abs": sp.Abs,
    "Rational": sp.Rational,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Sum": sp.Sum,
    "binomial": sp.binomial,
    "factorial": sp.factorial,
}

_TRANSFORMS = standard_transformations


class SymbolRegistry:
    """Name -> sympy Symbol, with assumptions fixed at declaration time."""

    #: assumptions given to symbols created implicitly (by appearing in an expression)
    DEFAULT_ASSUMPTIONS: dict[str, bool] = {"real": True}

    def __init__(self) -> None:
        self._symbols: dict[str, sp.Symbol] = {}
        self._declared: dict[str, dict[str, bool]] = {}

    # -- declaration ------------------------------------------------------------------
    def declare(self, name: str, **assumptions: bool) -> sp.Symbol:
        """Create ``name`` with explicit sympy assumptions (e.g. ``positive=True``).

        Must happen before the name is first used, because changing a symbol's assumptions
        changes its identity and would split it into two non-cancelling symbols.
        """
        if name in self._symbols:
            existing = self._declared.get(name, self.DEFAULT_ASSUMPTIONS)
            if existing != assumptions:
                raise ValueError(
                    f"symbol {name!r} already exists with assumptions {existing}; "
                    f"cannot re-declare it as {assumptions}. Declare symbols before "
                    f"first use -- in sympy, differing assumptions make differing symbols."
                )
            return self._symbols[name]
        sym = sp.Symbol(name, **assumptions)
        self._symbols[name] = sym
        self._declared[name] = dict(assumptions)
        return sym

    def get(self, name: str) -> sp.Symbol:
        """Return the Symbol for ``name``, creating it with default assumptions."""
        if name not in self._symbols:
            return self.declare(name, **self.DEFAULT_ASSUMPTIONS)
        return self._symbols[name]

    # -- parsing ----------------------------------------------------------------------
    def parse(self, value) -> sp.Expr:
        """Coerce ``value`` to a sympy expression using this registry's symbols.

        Accepts an int/float/Fraction, a sympy expression, or a string. Strings are parsed
        so that every bare identifier becomes a Symbol from this registry -- never a sympy
        constant or special function (see the module docstring).
        """
        if isinstance(value, sp.Basic):
            expr = value
        elif isinstance(value, str):
            expr = parse_expr(
                value,
                local_dict=dict(self._symbols),
                global_dict=dict(_SAFE_GLOBALS),
                transformations=_TRANSFORMS,
                evaluate=True,
            )
        else:
            expr = sp.sympify(value, rational=True)
        # Register any symbol the parse invented, and canonicalise symbols that arrived
        # inside a pre-built sympy expression to this registry's versions where names match.
        return self._canonicalise(expr)

    def _canonicalise(self, expr: sp.Expr) -> sp.Expr:
        """Give every free symbol this registry's assumptions for that name.

        A newly seen name is registered with :attr:`DEFAULT_ASSUMPTIONS`, whatever assumptions
        it arrived with. That matters: the parser's ``auto_symbol`` emits bare ``Symbol(name)``
        with no assumptions, so without this a symbol's assumptions would depend on whether it
        first appeared in a string or via :meth:`get` -- and two same-named symbols with
        differing assumptions are *unequal* in sympy and will not cancel.

        Note what this does and does not promise. It normalises **assumptions**, not object
        identity: when the incoming symbol already equals the registered one, sympy's ``subs``
        skips equal old/new pairs and returns the original object. That is harmless, because
        equal symbols are interchangeable everywhere in sympy. Do not write code (or tests)
        that depends on ``is`` -- sympy's Symbol constructor is LRU-cached at size 1000, so
        equal symbols become distinct objects once a process has created enough of them.
        """
        subs = {}
        for free in expr.free_symbols:
            known = self._symbols.get(free.name)
            if known is None:
                known = self.declare(free.name, **self.DEFAULT_ASSUMPTIONS)
            if known is not free:
                subs[free] = known
        return expr.subs(subs) if subs else expr

    # -- introspection ----------------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return name in self._symbols

    def __getitem__(self, name: str) -> sp.Symbol:
        return self._symbols[name]

    def __iter__(self):
        return iter(self._symbols)

    def __len__(self) -> int:
        return len(self._symbols)

    def as_dict(self) -> dict[str, sp.Symbol]:
        return dict(self._symbols)

    def copy(self) -> "SymbolRegistry":
        new = SymbolRegistry()
        new._symbols = dict(self._symbols)
        new._declared = {k: dict(v) for k, v in self._declared.items()}
        return new

    def __repr__(self) -> str:
        return f"SymbolRegistry({sorted(self._symbols)})"
