"""The genetics layer: pedigrees and assortative mating, built on the generic core.

This is where pathMgr stops being a generic SEM tool. The boundary is one-directional:
this package imports from :mod:`pathmgr.core`, never the reverse, and no genetics concept
(V_A, rho_g, transmission, pedigree) may leak into the core.

- :mod:`.alleles`  -- the allele-level transmission motif (task-20260804-173344)
- :mod:`.pedigree` -- build/unroll a pedigree as a path model (task-20260804-151350)
- :mod:`.am`       -- assortative-mating dynamics and the equilibrium fixed point
  (task-20260804-151351)
"""

from .alleles import AlleleMotif, allele_motif

__all__ = ["AlleleMotif", "allele_motif"]
