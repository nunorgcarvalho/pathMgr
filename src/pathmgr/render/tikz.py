"""TikZ export of a path diagram.

**Not yet implemented -- this is task-20260804-151349.**

Diagram conventions to honour: observed variables in boxes, latents in circles
(``Variable.latent``), single-headed arrows for directed paths labelled with the
coefficient, double-headed arrows for bidirected covariances, and a self-loop or stub for
a variance. ``Variable.label`` carries the LaTeX label to typeset (falling back to the
variable name).

**Stretch goal now unblocked** (noted by task-20260804-151348): highlighting a single traced
Wright chain on the diagram. The tracer provides what is needed --
:meth:`pathmgr.Chain.directed_edges` returns the ``(src, dst)`` edges the chain uses,
:attr:`pathmgr.Chain.pivot` the bidirected edge it passes through, and
:meth:`pathmgr.Chain.tex_path` a ready-made LaTeX rendering of the chain to use as a caption.
So a ``highlight=chain`` argument only has to restyle the edges in those sets;
:meth:`pathmgr.Chain.copath_edges` gives the co-paths it crossed.

**Co-paths need a third edge style** (task-20260804-173343). Sunde draw a co-path as a plain
**arrowless line**, visually distinct from both the single- and double-headed arrows, labelled
with the co-path coefficient. Read them off ``model.copaths``; each carries ``a``, ``b``,
``coefficient`` and a ``process`` identifier, and one couple may carry several co-paths sharing
a process (cross-trait assortment), which is worth grouping visually.
"""

__all__: list[str] = []
