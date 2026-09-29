"""Package root — Parkinson's Disease Motor Severity prediction workspace.

Exposes `PROJECT_ROOT`: the absolute path to the project root,
derived from this file's location. Any module that needs to resolve
a project-relative path (data files, reports) imports this constant
instead of hard-coding a CWD-relative string. This lets experiment
scripts run from any CWD without breaking.

Requires an editable install (`pip install -e .`) so `__file__`
points back into `src/parkinson/__init__.py` and `parents[2]` is
the project root.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
