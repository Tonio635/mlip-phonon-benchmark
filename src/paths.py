"""Individuazione della radice del progetto e percorsi standard.

Import tipico da un notebook in notebooks/:

    import sys; sys.path.insert(0, "..")
    from src.paths import ROOT, RESULTS, REFERENCE, FIGURES
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RESULTS = DATA / "results"
REFERENCE = DATA / "reference"
FIGURES = ROOT / "figures"

for _p in (RESULTS, REFERENCE, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)