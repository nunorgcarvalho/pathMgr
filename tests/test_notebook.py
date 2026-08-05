"""Re-execute ``examples/pathmgr_tour.ipynb`` and fail if any cell raises.

Without this the notebook rots at the next API change, silently, because a committed notebook keeps
displaying the outputs it was executed with. That makes a stale notebook look *more* trustworthy
than a broken script, not less -- it renders perfectly on GitHub while teaching an API that no
longer exists.

Optional tooling, so it skips cleanly when absent, the same way the ``pdflatex`` tests in
``test_render.py`` do: ``pip install 'pathmgr[notebook]'``.

The execution is done on a **copy in a temp directory** with a fresh kernel, so running the tests
never rewrites the committed outputs -- the notebook in the repo is the executed artifact and this
test must not touch it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / "examples" / "pathmgr_tour.ipynb"

# exc_type is explicit so that a *partially* installed nbconvert -- which raises ImportError rather
# than ModuleNotFoundError -- also skips instead of failing the suite. It is pytest 9.1's default
# behaviour anyway, and passing it silences the deprecation warning on 9.0.
nbformat = pytest.importorskip("nbformat", reason="needs pathmgr[notebook]", exc_type=ImportError)
nbconvert = pytest.importorskip("nbconvert", reason="needs pathmgr[notebook]", exc_type=ImportError)

from nbconvert.preprocessors import ExecutePreprocessor  # noqa: E402

#: generous, because a cold kernel start dominates; a clean run measures well under a minute
TIMEOUT_SECONDS = 600


def test_the_notebook_exists_and_is_committed_with_outputs():
    """A notebook committed with empty outputs is a broken artifact: it must render on GitHub."""
    assert NOTEBOOK.exists(), f"{NOTEBOOK} is missing"
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    code_cells = [c for c in notebook.cells if c.cell_type == "code" and c.source.strip()]
    assert code_cells, "no code cells"
    empty = [i for i, c in enumerate(code_cells) if not c.outputs]
    assert not empty, f"code cells with no stored output: {empty}"

    # and no stored output may be an error traceback
    errors = [
        (i, o.get("ename"))
        for i, c in enumerate(code_cells)
        for o in c.outputs
        if o.output_type == "error"
    ]
    assert not errors, f"stored error outputs: {errors}"


def test_no_display_mode_environment_is_wrapped_in_inline_math():
    """``IPython.display.Math`` wraps its argument in inline ``$...$``; ``align*`` needs display mode.

    KaTeX -- which JupyterLab and nbviewer use -- rejects the combination outright with
    "{align*} can be used only in display mode", so the cell renders as a red error box for every
    reader while executing perfectly for the author. Nothing else catches it: the notebook runs
    clean, the test suite passes, and the failure is purely in the rendering layer.

    The fix is ``display(Latex(...))``, which passes the string through at top level.
    """
    import re

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    display_only = ("align", "align*", "gather", "gather*", "equation", "equation*", "multline")

    broken = []
    for index, cell in enumerate(notebook.cells):
        for output in cell.get("outputs", []):
            latex = output.get("data", {}).get("text/latex", "")
            stripped = latex.strip()
            inline = stripped.startswith("$") and not stripped.startswith("$$")
            if not inline:
                continue
            for environment in re.findall(r"\\begin\{(\w+\*?)\}", latex):
                if environment in display_only:
                    broken.append((index, environment))
    assert not broken, (
        f"display-mode environments inside inline math: {broken}. "
        "Use display(Latex(...)) rather than display(Math(...))."
    )


def test_execution_counts_are_sequential_with_no_gaps():
    """A cell with no execution count renders as ``[ ]`` and offsets every number after it.

    That makes it impossible to refer to "cell [11]" and be understood, which is a real cost for a
    notebook meant to be read and discussed rather than only run.
    """
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    counts = [
        c.execution_count
        for c in notebook.cells
        if c.cell_type == "code" and c.source.strip()
    ]
    assert counts == list(range(1, len(counts) + 1)), f"non-sequential execution counts: {counts}"


def test_every_markdown_link_in_the_notebook_resolves():
    """Relative links break silently when files move; the layout tidy is exactly when that happens."""
    import re

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    root = NOTEBOOK.parent
    missing = []
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        for target in re.findall(r"\]\((\.\.?/[^)]+)\)", cell.source):
            if not (root / target).resolve().exists():
                missing.append(target)
    assert not missing, f"dead relative links: {missing}"


@pytest.mark.skipif(shutil.which("python") is None, reason="no python on PATH for the kernel")
def test_the_notebook_runs_top_to_bottom(tmp_path):
    """The real check: a fresh kernel, every cell, no exceptions.

    Executed on a copy so the committed outputs are never rewritten by a test run.
    """
    working = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, working)
    notebook = nbformat.read(working, as_version=4)

    processor = ExecutePreprocessor(timeout=TIMEOUT_SECONDS, kernel_name="python3")
    # resources path is the notebook's own directory, so any relative path it uses behaves as it
    # does for a reader -- which is what makes "no absolute paths" testable rather than aspirational
    processor.preprocess(notebook, {"metadata": {"path": str(tmp_path)}})

    executed = [c for c in notebook.cells if c.cell_type == "code" and c.source.strip()]
    for index, cell in enumerate(executed):
        errors = [o for o in cell.outputs if o.output_type == "error"]
        assert not errors, f"cell {index} raised {errors[0].get('ename')}: {errors[0].get('evalue')}"
    assert all(cell.outputs for cell in executed), "a cell produced no output on re-execution"
