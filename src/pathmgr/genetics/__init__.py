"""The genetics layer: pedigrees and assortative mating, built on the generic core.

This is where pathMgr stops being a generic SEM tool. The boundary is one-directional:
this package imports from :mod:`pathmgr.core`, never the reverse, and no genetics concept
(V_A, rho_g, transmission, pedigree) may leak into the core.

- :mod:`.alleles`  -- the allele-level transmission motif, one generation
- :mod:`.pedigree` -- pedigree scaffolding and the g-level unroller
- :mod:`.am`       -- assortative-mating dynamics and the equilibrium fixed point
  (task-20260804-151351)
"""

from .alleles import AlleleMotif, allele_motif
from .pedigree import (
    AMParameters,
    Couple,
    Individual,
    Pedigree,
    UnrolledModel,
    am_pedigree,
    g_level_model,
)

__all__ = [
    "AMParameters",
    "AlleleMotif",
    "Couple",
    "Individual",
    "Pedigree",
    "UnrolledModel",
    "allele_motif",
    "am_pedigree",
    "g_level_model",
]
