"""A terse text front-end for the specification API: parse text into a :class:`Model`, and
write a Model back out as text.

This is a thin layer over the builder -- ``from_text`` makes builder calls and nothing else,
so the two front-ends cannot diverge in meaning. It exists because hand-written models (and
models quoted into a LaTeX writeup) read better as text than as a sequence of calls.

Grammar
-------
One statement per line. ``#`` starts a comment; blank lines are ignored. Directive lines
(``key: ...``) may appear in any order and are all processed before the equations, so a
``latent:`` or ``positive:`` line can sit at the bottom of the block.

Directives::

    units: unstandardized                      # the default
    units: standardized to base generation     # reference is mandatory
    latent: g_i, e_i, g_j, e_j                 # everything else is observed
    observed: y_i, y_j                         # optional; only needed for isolated nodes
    positive: V_A, V_E                         # sympy assumptions on SYMBOLS
    real: b1, b2                               # (the default for new symbols)
    label: g_i = $g_i$                         # rendering label
    assume: V_A + V_E = 1                      # side relation, not an edge
    assume: rho_g = rho_y * h2_eq

Directed paths -- ``dst ~ terms``, read as "dst regresses on ..."::

    y_i ~ g_i + e_i                            # coefficient 1 is implied
    y   ~ b1*x1 + b2*x2
    g_o ~ 1/2*g_m + 1/2*g_p + s_o              # exact rationals, not floats
    g_c ~ ((1 + rho_g)/2)*g_p                  # parenthesise a compound coefficient

Co-paths -- ``a -- terms``, covariance from matching (Sunde's arrowless line)::

    y_m -- mu*y_p                              # a co-path; mu is NOT the correlation
    S_m -- mu*S_p [couple0]                    # name the mating process explicitly
    S_m -- (mu_prime)*Sx_p [couple0]           # a second co-path on the SAME process

The trailing ``[name]`` names the mating process, defaulting to the pair itself. It only
matters when one couple carries several co-paths (cross-trait assortment), because a chain may
not use two co-paths from the same process.

Bidirected covariances -- ``a ~~ terms``, disturbance covariances (see
:mod:`pathmgr.core.model` for why these are residual, not total, on endogenous variables)::

    x1 ~~ V_1*x1                               # a variance
    x1 ~~ c12*x2                               # a covariance
    g_i ~~ (V_A*pi_ij)*g_j                     # expression value
    e_m ~~ (rho_g*V_E)*g_p                     # one person's environment, another's genes

The one rule worth internalising
--------------------------------
**Every term on a right-hand side must end in a variable name.** In ``(V_A*pi_ij)*g_j`` the
trailing identifier ``g_j`` is the variable and everything before the final ``*`` is the
coefficient. There is deliberately **no** bare-variance shorthand, because ``g_i ~~ V_A``
is unresolvably ambiguous -- is ``V_A`` the variance of ``g_i``, or a variable that ``g_i``
covaries with? Write ``g_i ~~ V_A*g_i``. As a backstop, a name used both as a variable and
as a coefficient symbol is rejected outright rather than silently conflated.

Variables are created on first use in a variable position; ``latent:`` only changes whether
a variable is latent, so it never has to duplicate the list of names.
"""

from __future__ import annotations

import re

import sympy as sp

from .model import Model
from .units import Units

__all__ = ["from_text", "to_text", "TextSyntaxError"]

_DIRECTIVES = ("units", "latent", "observed", "positive", "real", "integer", "label", "assume")
_ASSUMPTION_DIRECTIVES = ("positive", "real", "integer")
_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")
_PROCESS_RE = re.compile(r"\[(?P<process>[^\]]*)\]\s*$")
_TERM_RE = re.compile(r"^\s*(?P<coeff>.*\*)?\s*(?P<var>[A-Za-z_]\w*)\s*$")


class TextSyntaxError(ValueError):
    """A malformed model description, reported with its line number and the line itself."""

    def __init__(self, lineno: int, line: str, message: str):
        self.lineno = lineno
        self.line = line
        super().__init__(f"line {lineno}: {message}\n    {line.strip()}")


# ======================================================================================
# parsing
# ======================================================================================
def from_text(text: str, name: str | None = None) -> Model:
    """Build a :class:`Model` from a text description. See the module docstring for the grammar."""
    statements = _statements(text)

    units = Units.unstandardized()
    latent: set[str] = set()
    declared: list[tuple[str, str]] = []  # (variable name, "latent" | "observed")
    sym_assumptions: list[tuple[str, str, int, str]] = []  # (name, kind, lineno, line)
    labels: dict[str, str] = {}
    assumes: list[tuple[str, str, int, str]] = []
    equations: list[tuple[str, str, str, int, str]] = []  # (op, lhs, rhs, lineno, line)

    # -- pass 1: directives ------------------------------------------------------------
    for lineno, line in statements:
        key, _, rest = line.partition(":")
        key = key.strip().lower()
        if _is_directive(line, key):
            rest = rest.strip()
            if key == "units":
                units = _parse_units(rest, lineno, line)
            elif key in ("latent", "observed"):
                for n in _names(rest, lineno, line):
                    declared.append((n, key))
                    if key == "latent":
                        latent.add(n)
            elif key in _ASSUMPTION_DIRECTIVES:
                for n in _names(rest, lineno, line):
                    sym_assumptions.append((n, key, lineno, line))
            elif key == "label":
                target, eq, value = rest.partition("=")
                if not eq:
                    raise TextSyntaxError(lineno, line, "label needs 'name = text'")
                labels[target.strip()] = value.strip().strip('"').strip("'")
            elif key == "assume":
                lhs, eq, rhs = rest.partition("=")
                if not eq:
                    raise TextSyntaxError(lineno, line, "assume needs 'lhs = rhs'")
                assumes.append((lhs.strip(), rhs.strip(), lineno, line))
            continue

        op, lhs, rhs = _split_equation(line, lineno)
        equations.append((op, lhs, rhs, lineno, line))

    model = Model(name, units)

    # symbol assumptions must be declared before any expression is parsed
    for sym_name, kind, lineno, line in sym_assumptions:
        if not _NAME_RE.match(sym_name):
            raise TextSyntaxError(lineno, line, f"{sym_name!r} is not a valid symbol name")
        try:
            model.declare(sym_name, **{kind: True})
        except ValueError as exc:
            raise TextSyntaxError(lineno, line, str(exc)) from exc

    for n, kind in declared:
        _ensure_var(model, n, latent, labels)

    # -- pass 2: equations -------------------------------------------------------------
    coefficient_symbols: set[str] = set()
    for op, lhs, rhs, lineno, line in equations:
        if not _NAME_RE.match(lhs):
            raise TextSyntaxError(
                lineno, line, f"left-hand side {lhs!r} must be a single variable name"
            )
        _ensure_var(model, lhs, latent, labels)
        process = None
        if op == "--":
            match = _PROCESS_RE.search(rhs)
            if match:
                process = match.group("process").strip()
                rhs = rhs[: match.start()].strip()
                if not process:
                    raise TextSyntaxError(lineno, line, "empty mating-process name in [...]")
        for term in _split_terms(rhs, lineno, line):
            coeff_text, var_name = _split_term(term, lineno, line)
            _ensure_var(model, var_name, latent, labels)
            try:
                coeff = model.expr(coeff_text) if coeff_text is not None else sp.Integer(1)
            except (SyntaxError, TypeError, sp.SympifyError) as exc:
                raise TextSyntaxError(
                    lineno, line, f"could not parse coefficient {coeff_text!r}: {exc}"
                ) from exc
            coefficient_symbols.update(s.name for s in coeff.free_symbols)
            try:
                if op == "~":
                    model.add_path(var_name, lhs, coeff)
                elif op == "~~":
                    model.add_cov(lhs, var_name, coeff)
                else:
                    model.add_copath(lhs, var_name, coeff, process=process)
            except ValueError as exc:
                raise TextSyntaxError(lineno, line, str(exc)) from exc

    # -- pass 3: side relations --------------------------------------------------------
    for lhs, rhs, lineno, line in assumes:
        try:
            model.assume(lhs, rhs)
        except (SyntaxError, TypeError, sp.SympifyError) as exc:
            raise TextSyntaxError(lineno, line, f"could not parse assumption: {exc}") from exc
        for eq in model.assumptions[-1:]:
            coefficient_symbols.update(s.name for s in eq.free_symbols)

    # -- backstop: a name cannot be both a variable and a coefficient symbol -----------
    clash = sorted(coefficient_symbols & set(model.names))
    if clash:
        raise ValueError(
            f"name(s) {clash} are used both as a variable and as a coefficient symbol. "
            f"Most often this means a bare-variance shorthand was written: 'a ~~ V' makes "
            f"V a variable. Write 'a ~~ V*a' instead."
        )
    return model


def _is_directive(line: str, key: str) -> bool:
    """A directive is ``key: ...`` for a known key -- and equations never contain ':'."""
    return key in _DIRECTIVES and ":" in line


def _statements(text: str) -> list[tuple[int, str]]:
    out = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append((lineno, line))
    return out


def _parse_units(rest: str, lineno: int, line: str) -> Units:
    low = rest.lower()
    if low in ("unstandardized", "unstandardised"):
        return Units.unstandardized()
    for prefix in ("standardized", "standardised"):
        if low.startswith(prefix):
            ref = rest[len(prefix):].strip()
            for lead in ("to", "in"):
                if ref.lower().startswith(lead + " "):
                    ref = ref[len(lead) + 1:].strip()
            ref = ref.strip('"').strip("'").lstrip(":").strip()
            if not ref:
                raise TextSyntaxError(
                    lineno,
                    line,
                    "a standardized model must name its reference population, e.g. "
                    "'units: standardized to base generation (gen 0)'",
                )
            return Units.standardized(ref)
    raise TextSyntaxError(
        lineno, line, f"unknown units {rest!r}; expected 'unstandardized' or 'standardized to <ref>'"
    )


def _names(rest: str, lineno: int, line: str) -> list[str]:
    names = [n.strip() for n in rest.replace(",", " ").split()]
    for n in names:
        if not _NAME_RE.match(n):
            raise TextSyntaxError(lineno, line, f"{n!r} is not a valid name")
    if not names:
        raise TextSyntaxError(lineno, line, "expected at least one name")
    return names


def _split_equation(line: str, lineno: int) -> tuple[str, str, str]:
    """Return ``(op, lhs, rhs)`` where op is ``'~~'`` or ``'~'``. ``~~`` is checked first."""
    # Pick the operator that appears EARLIEST, so a coefficient containing '--' (e.g.
    # `y ~ (a--b)*x`) cannot be mistaken for a co-path line. At the same position, the longer
    # operator wins, so '~~' beats '~'.
    candidates = [(line.find(op), -len(op), op) for op in ("--", "~~", "~") if op in line]
    if candidates:
        _, _, op = min(candidates)
        lhs, _, rhs = line.partition(op)
        lhs, rhs = lhs.strip(), rhs.strip()
        if not lhs or not rhs:
            raise TextSyntaxError(lineno, line, f"'{op}' needs both a left and right side")
        return op, lhs, rhs
    raise TextSyntaxError(
        lineno,
        line,
        "not a directive and not an equation; expected 'key: ...', 'dst ~ ...', 'a ~~ ...' "
        "or 'a -- ...' (co-path)",
    )


def _split_terms(rhs: str, lineno: int, line: str) -> list[str]:
    """Split on top-level ``+``/``-``, keeping the sign with its term."""
    terms: list[str] = []
    depth = 0
    cur = ""
    for i, ch in enumerate(rhs):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                raise TextSyntaxError(lineno, line, "unbalanced parentheses")
        if depth == 0 and ch in "+-" and cur.strip() and not _continues(cur):
            terms.append(cur)
            cur = ch
            continue
        cur += ch
    if depth != 0:
        raise TextSyntaxError(lineno, line, "unbalanced parentheses")
    if cur.strip():
        terms.append(cur)
    if not terms:
        raise TextSyntaxError(lineno, line, "right-hand side has no terms")
    return [t.strip() for t in terms]


def _continues(cur: str) -> bool:
    """True if a ``+``/``-`` here is part of the expression, not a term separator."""
    stripped = cur.rstrip()
    if stripped.endswith(("*", "/", "(", "+", "-", "^", ",")):
        return True
    # scientific notation: 1e-5
    return len(stripped) >= 2 and stripped[-1] in "eE" and (
        stripped[-2].isdigit() or stripped[-2] == "."
    )


def _split_term(term: str, lineno: int, line: str) -> tuple[str | None, str]:
    """``'(V_A*pi_ij)*g_j'`` -> ``('(V_A*pi_ij)', 'g_j')``; ``'g_i'`` -> ``(None, 'g_i')``."""
    sign = ""
    t = term.strip()
    while t[:1] in ("+", "-"):
        if t[0] == "-":
            sign = "-" if sign == "" else ""
        t = t[1:].strip()
    match = _TERM_RE.match(t)
    if not match:
        raise TextSyntaxError(
            lineno,
            line,
            f"term {term.strip()!r} must end in a variable name, e.g. 'b1*x1' or "
            f"'(V_A*pi_ij)*g_j'. There is no bare-variance shorthand: write 'a ~~ V*a'.",
        )
    coeff = match.group("coeff")
    var_name = match.group("var")
    if coeff is None:
        return (sign + "1" if sign else None), var_name
    return sign + "(" + coeff.rstrip().rstrip("*").strip() + ")", var_name


def _ensure_var(model: Model, name: str, latent: set[str], labels: dict[str, str]) -> None:
    if not model.has_var(name):
        model.add_var(name, latent=name in latent, label=labels.get(name))


# ======================================================================================
# writing
# ======================================================================================
def to_text(model: Model, include_name: bool = True) -> str:
    """Render ``model`` back out in the :func:`from_text` grammar. Round-trips."""
    lines: list[str] = []
    if include_name and model.name:
        lines.append(f"# {model.name}")
    lines.append(f"units: {_units_text(model.units)}")

    if model.latent:
        lines.append("latent: " + ", ".join(model.latent))
    isolated = [
        n for n in model.observed
        if not model.parents(n) and not model.children(n)
    ]
    if isolated:
        lines.append("observed: " + ", ".join(isolated))

    # `real` is the default for new symbols, so it is never worth emitting. Each symbol is
    # listed under at most one directive: the grammar allows one kind per name, and emitting
    # a name twice would make the text fail to re-parse.
    by_kind: dict[str, list[str]] = {"positive": [], "integer": []}
    for symbol in model.symbols.as_dict().values():
        if symbol.is_positive:
            by_kind["positive"].append(symbol.name)
        elif symbol.is_integer:
            by_kind["integer"].append(symbol.name)
    for kind, named in by_kind.items():
        if named:
            lines.append(f"{kind}: " + ", ".join(sorted(named)))

    for var in model.variables:
        if var.label is not None and var.label != var.name:
            lines.append(f"label: {var.name} = {var.label}")

    directed: dict[str, list[str]] = {}
    for edge in model.directed_edges:
        directed.setdefault(edge.dst, []).append(_term_text(edge.coeff, edge.src))
    if directed:
        lines.append("")
        for dst, terms in directed.items():
            lines.append(f"{dst} ~ " + " + ".join(terms))

    if model.bidirected_edges:
        lines.append("")
        for edge in model.bidirected_edges:
            lines.append(f"{edge.a} ~~ " + _term_text(edge.value, edge.b))

    if model.copaths:
        lines.append("")
        for copath in model.copaths:
            default = f"{copath.a}--{copath.b}"
            suffix = "" if copath.process == default else f" [{copath.process}]"
            lines.append(
                f"{copath.a} -- " + _term_text(copath.coefficient, copath.b) + suffix
            )

    if model.assumptions:
        lines.append("")
        for eq in model.assumptions:
            lines.append(f"assume: {sp.sstr(eq.lhs)} = {sp.sstr(eq.rhs)}")

    return "\n".join(lines) + "\n"


def _units_text(units: Units) -> str:
    return f"standardized to {units.reference}" if units.is_standardized else "unstandardized"


def _term_text(coeff: sp.Expr, var_name: str) -> str:
    if coeff == 1:
        return var_name
    text = sp.sstr(coeff)
    if isinstance(coeff, sp.Add) or coeff.could_extract_minus_sign():
        text = f"({text})"
    return f"{text}*{var_name}"
