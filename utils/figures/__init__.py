"""Shared foundation for the handily figure series (see notes/FIGURE_SERIES_PLAN.md).

Run every module in this package as::

    uv run python -m utils.figures.<module>

from the repository root. Modules import each other absolutely
(``from utils.figures.fig_common import ...``); each entry-point module inserts the
repository root on ``sys.path`` so direct script invocation also works.
"""
