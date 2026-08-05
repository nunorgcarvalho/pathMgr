"""Path-diagram rendering. Kept strictly separate from covariance computation.

- :mod:`.layout` -- node placement: explicit coordinates, with a layered automatic fallback
- :mod:`.style`  -- the drawing conventions, shared by both back ends
- :mod:`.tikz`   -- TikZ export for a LaTeX writeup, plus PDF compilation
- :mod:`.raster` -- PNG/SVG export via matplotlib

Nothing here may be imported by :mod:`pathmgr.core`, and matplotlib is imported lazily inside
:mod:`.raster` so that ``import pathmgr`` never needs a drawing dependency.
"""

from .layout import Layout, layered_layout, pedigree_layout
from .placement import EdgeCrossing, edge_node_crossings, route_edges
from .raster import to_image
from .style import DiagramStyle, coefficient_label
from .tikz import TikzCompileError, to_standalone, to_tikz, write_pdf

__all__ = [
    "DiagramStyle",
    "EdgeCrossing",
    "Layout",
    "TikzCompileError",
    "coefficient_label",
    "edge_node_crossings",
    "layered_layout",
    "pedigree_layout",
    "route_edges",
    "to_image",
    "to_standalone",
    "to_tikz",
    "write_pdf",
]
