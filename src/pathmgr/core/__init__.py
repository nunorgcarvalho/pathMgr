"""Generic path-analysis core. Knows nothing about genetics -- keep it that way.

- :mod:`.model`   -- the specification object every engine and renderer consumes
- :mod:`.units`   -- the scale a model is stated on, and its reference population
- :mod:`.symbols` -- symbol registry + safe parsing of symbolic coefficients
- :mod:`.text`    -- terse text front-end: text <-> Model, a thin layer over the builder
- :mod:`.ram`     -- closed-form covariance engine (task-20260804-151347)
- :mod:`.tracing` -- Wright chain enumeration engine: the covariance DECOMPOSITION
"""

from .model import BidirectedEdge, DirectedEdge, Model, ModelIssue, Variable
from .ram import CovarianceReport, CyclicModelError, RAMEngine, SingularModelError
from .symbols import SymbolRegistry
from .tracing import (
    Chain,
    ChainLimitError,
    Decomposition,
    UntraceableModelError,
    WrightTracer,
)
from .text import TextSyntaxError, from_text, to_text
from .units import Units

__all__ = [
    "BidirectedEdge",
    "Chain",
    "ChainLimitError",
    "CovarianceReport",
    "CyclicModelError",
    "Decomposition",
    "DirectedEdge",
    "Model",
    "ModelIssue",
    "RAMEngine",
    "SingularModelError",
    "SymbolRegistry",
    "TextSyntaxError",
    "UntraceableModelError",
    "WrightTracer",
    "Units",
    "Variable",
    "from_text",
    "to_text",
]
