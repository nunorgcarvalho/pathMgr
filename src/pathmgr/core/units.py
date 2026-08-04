"""Units of a path model.

A path model's numbers only mean something on a stated scale. Classic Wright tracing
rules assume every variable has unit variance, but the genetics is naturally written in
unstandardized components (V_A, V_E, V_K) on a scale that *shifts across generations*
under assortative mating. So "standardized" is not a complete statement -- it must always
answer "standardized to which reference population?".

This module makes that non-negotiable: `Units.standardized()` refuses to be constructed
without a reference label.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Units:
    """The scale on which a model's coefficients and (co)variances are stated.

    Two kinds:

    - ``unstandardized`` -- edges and covariances are in variance units; a variable's
      variance is whatever the model implies. This is how the genetics is written.
    - ``standardized`` -- every variable is asserted to have unit variance *in a named
      reference population*, so path coefficients are correlations/standardized betas.
      The reference is mandatory and is carried through into every result.
    """

    kind: str
    reference: str | None = None

    _KINDS = ("unstandardized", "standardized")

    def __post_init__(self) -> None:
        if self.kind not in self._KINDS:
            raise ValueError(f"units kind must be one of {self._KINDS}, got {self.kind!r}")
        if self.kind == "standardized" and not self.reference:
            raise ValueError(
                "a standardized model must name its reference population, e.g. "
                "Units.standardized('base generation (gen 0)'). Under assortative mating "
                "the phenotypic scale changes every generation, so 'standardized' alone "
                "is ambiguous."
            )
        if self.kind == "unstandardized" and self.reference:
            raise ValueError("an unstandardized model has no reference population")

    @classmethod
    def unstandardized(cls) -> "Units":
        return cls("unstandardized")

    @classmethod
    def standardized(cls, reference: str) -> "Units":
        return cls("standardized", reference)

    @property
    def is_standardized(self) -> bool:
        return self.kind == "standardized"

    def __str__(self) -> str:
        if self.is_standardized:
            return f"standardized to {self.reference}"
        return "unstandardized"
