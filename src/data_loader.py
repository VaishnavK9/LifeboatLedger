"""Load the raw Titanic CSVs and produce inspection reports.

The three files from the Prodigy InfoTech dataset repository are expected in
``data/raw/``:

* ``train.csv`` — 891 passengers with the ``Survived`` label.
* ``test.csv`` — 418 passengers without the label.
* ``gender_submission.csv`` — Kaggle's "all women survive" example submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import RAW_DIR

RAW_FILES: dict[str, str] = {
    "train": "train.csv",
    "test": "test.csv",
    "gender_submission": "gender_submission.csv",
}

KEY_CATEGORICALS: list[str] = ["Survived", "Pclass", "Sex", "Embarked", "SibSp", "Parch"]


def load_csv(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load one of the raw CSVs by its short name.

    Args:
        name: One of ``"train"``, ``"test"`` or ``"gender_submission"``.
        raw_dir: Folder containing the raw CSVs.

    Returns:
        The file as a DataFrame.

    Raises:
        KeyError: If ``name`` is not a known dataset.
        FileNotFoundError: If the CSV is missing from ``raw_dir``.
    """
    if name not in RAW_FILES:
        raise KeyError(f"Unknown dataset {name!r}; expected one of {list(RAW_FILES)}")
    path = raw_dir / RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Titanic files from "
            "https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%202 "
            f"into {raw_dir}."
        )
    return pd.read_csv(path)


def load_train(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the labelled training set (891 rows)."""
    return load_csv("train", raw_dir)


def load_test(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the unlabelled test set (418 rows)."""
    return load_csv("test", raw_dir)


def load_all(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load every raw CSV into a dict keyed by short name."""
    return {name: load_csv(name, raw_dir) for name in RAW_FILES}


@dataclass
class InspectionReport:
    """Summary of a DataFrame's structure and quality."""

    name: str
    shape: tuple[int, int]
    duplicate_rows: int
    duplicate_ids: int
    columns: pd.DataFrame
    value_counts: dict[str, pd.Series]

    def __str__(self) -> str:
        lines = [
            f"===== {self.name} =====",
            f"shape: {self.shape[0]} rows × {self.shape[1]} columns",
            f"duplicate rows: {self.duplicate_rows} | duplicate PassengerIds: {self.duplicate_ids}",
            "",
            self.columns.to_string(),
        ]
        for col, counts in self.value_counts.items():
            lines += ["", f"-- {col} value counts:", counts.to_string()]
        return "\n".join(lines)


def inspect_dataframe(
    df: pd.DataFrame,
    name: str = "dataframe",
    categorical_cols: list[str] | None = None,
) -> InspectionReport:
    """Summarise shape, dtypes, nulls, duplicates and key value counts.

    Args:
        df: DataFrame to inspect.
        name: Label used in the printed report.
        categorical_cols: Columns to include value counts for. Defaults to
            :data:`KEY_CATEGORICALS` (those present in ``df``).

    Returns:
        An :class:`InspectionReport`; ``print()`` it for a readable summary.
    """
    columns = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "nulls": df.isna().sum(),
            "null_%": (df.isna().mean() * 100).round(1),
            "unique": df.nunique(),
        }
    )
    cats = categorical_cols if categorical_cols is not None else KEY_CATEGORICALS
    value_counts = {
        col: df[col].value_counts(dropna=False) for col in cats if col in df.columns
    }
    duplicate_ids = int(df["PassengerId"].duplicated().sum()) if "PassengerId" in df else 0
    return InspectionReport(
        name=name,
        shape=df.shape,
        duplicate_rows=int(df.duplicated().sum()),
        duplicate_ids=duplicate_ids,
        columns=columns,
        value_counts=value_counts,
    )


def missing_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Side-by-side count and percentage of nulls for several DataFrames.

    Only columns with at least one missing value in any frame are kept.
    """
    parts = []
    for name, df in frames.items():
        parts.append(df.isna().sum().rename(f"{name}_nulls"))
        parts.append((df.isna().mean() * 100).round(1).rename(f"{name}_%"))
    summary = pd.concat(parts, axis=1).fillna(0)
    null_cols = [c for c in summary.columns if c.endswith("_nulls")]
    summary[null_cols] = summary[null_cols].astype(int)
    return summary[summary[null_cols].sum(axis=1) > 0]


if __name__ == "__main__":
    data = load_all()
    for key, frame in data.items():
        print(inspect_dataframe(frame, key), end="\n\n")
    print("===== Missing values (train vs test) =====")
    print(missing_summary({"train": data["train"], "test": data["test"]}).to_string())
