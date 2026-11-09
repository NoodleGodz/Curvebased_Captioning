"""Shared test setup: put the repo's module folders on sys.path.

Run from anywhere:  python -m pytest -q tests
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
for sub in ("scripts", "tools/cache_data", "evaluation"):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
