"""TikZ export of a path diagram.

**Not yet implemented -- this is task-20260804-151349.**

Diagram conventions to honour: observed variables in boxes, latents in circles
(``Variable.latent``), single-headed arrows for directed paths labelled with the
coefficient, double-headed arrows for bidirected covariances, and a self-loop or stub for
a variance. ``Variable.label`` carries the LaTeX label to typeset (falling back to the
variable name).
"""

__all__: list[str] = []
