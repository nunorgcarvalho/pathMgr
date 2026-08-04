"""pathMgr -- symbolic path analysis.

Three jobs: **specify** a model of dependencies between variables (directed paths and
bidirected covariances, with symbolic coefficients); **compute** the covariance or
correlation between any two of them -- latent and intermediate ones included -- symbolically;
and **draw** the corresponding path diagram (TikZ for LaTeX, plus a raster export).

pathMgr does *not* fit models to data. No estimation, no optimization, no fit statistics.
It is a symbolic derivation and visualization tool.

The generic path-analysis core lives in :mod:`pathmgr.core` and knows nothing about
genetics; the genetics (pedigrees, assortative mating) lives in :mod:`pathmgr.genetics`
on top of it.
"""

from .core import (
    BidirectedEdge,
    Chain,
    ChainLimitError,
    CoPath,
    CoPathLimitError,
    CovarianceReport,
    CyclicModelError,
    Decomposition,
    DirectedEdge,
    Model,
    ModelIssue,
    RAMEngine,
    SingularModelError,
    Segment,
    SymbolRegistry,
    TextSyntaxError,
    UntraceableModelError,
    WrightTracer,
    Units,
    Variable,
    from_text,
    to_text,
)

from . import render  # noqa: E402  (subpackage; matplotlib stays lazy inside .raster)

__version__ = "0.0.1"

__all__ = [
    "BidirectedEdge",
    "Chain",
    "ChainLimitError",
    "CoPath",
    "CoPathLimitError",
    "CovarianceReport",
    "CyclicModelError",
    "Decomposition",
    "DirectedEdge",
    "Model",
    "ModelIssue",
    "RAMEngine",
    "SingularModelError",
    "Segment",
    "SymbolRegistry",
    "TextSyntaxError",
    "UntraceableModelError",
    "WrightTracer",
    "Units",
    "Variable",
    "__version__",
    "render",
    "from_text",
    "to_text",
]
