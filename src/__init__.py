"""Lifeboat Ledger — data cleaning and EDA on the Titanic dataset.

Prodigy InfoTech Data Science Internship – Task 02.
"""

from pathlib import Path

__version__ = "0.1.0"

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
