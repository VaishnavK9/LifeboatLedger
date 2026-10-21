"""Feature engineering: Title, FamilySize, IsAlone, AgeGroup, FareBand, Deck.

These features work on the *cleaned* data from :mod:`src.cleaning`. ``Deck``
and ``HasCabin`` are already created there (they replace the raw ``Cabin``
column during cleaning). This module adds the rest and also provides a
numeric encoding used by the correlation heatmap and the baseline models.

Bin edges that depend on the data (the ``FareBand`` quartiles) are learned
from the training set and reused for test.

Run as a script to write ``data/processed/{train,test}_features.csv``::

    python -m src.features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import PROCESSED_DIR
from src.cleaning import extract_title, fix_dtypes, load_processed

TITLE_ORDER: list[str] = ["Mr", "Mrs", "Miss", "Master", "Rare"]

AGE_BINS: list[float] = [0, 13, 18, 35, 60, np.inf]
AGE_LABELS: list[str] = ["Child (0-12)", "Teen (13-17)", "Young Adult (18-34)",
                         "Adult (35-59)", "Senior (60+)"]

FARE_LABELS: list[str] = ["Low", "Mid", "High", "Very High"]

FAMILY_BINS: list[float] = [0, 1, 4, np.inf]
FAMILY_LABELS: list[str] = ["Alone", "Small (2-4)", "Large (5+)"]

CATEGORICAL_FEATURES: dict[str, list[str]] = {
    "Title": TITLE_ORDER,
    "AgeGroup": AGE_LABELS,
    "FareBand": FARE_LABELS,
    "FamilyGroup": FAMILY_LABELS,
}


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``Title`` (Mr, Mrs, Miss, Master, Rare) taken from ``Name``.

    Title combines sex, rough age and social status in one field: "Master"
    means a boy, and "Rare" covers clergy, military, doctors and nobility.
    """
    df["Title"] = extract_title(df["Name"])
    return df


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``FamilySize``, ``IsAlone`` and ``FamilyGroup``.

    * ``FamilySize`` = siblings/spouses + parents/children + the passenger.
    * ``IsAlone`` = 1 when travelling with no family.
    * ``FamilyGroup`` puts size into Alone / Small (2-4) / Large (5+), because
      the survival effect is not linear: small families did best, while
      solo travellers and large families did worse.
    """
    df["FamilySize"] = (df["SibSp"] + df["Parch"] + 1).astype("int8")
    df["IsAlone"] = (df["FamilySize"] == 1).astype("int8")
    df["FamilyGroup"] = pd.cut(df["FamilySize"], bins=FAMILY_BINS, labels=FAMILY_LABELS)
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``AgeGroup`` using life-stage bins.

    The bins are ``[0, 13)``, ``[13, 18)``, ``[18, 35)``, ``[35, 60)`` and
    ``[60, ∞)``. These follow the "women and children first" question
    directly, unlike equal-width or quantile bins.
    """
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return df


def fare_band_edges(train: pd.DataFrame) -> np.ndarray:
    """Learn quartile edges for ``FareBand`` from the training fares.

    The outer edges are opened to ±∞ so any test fare falls into a band.
    """
    _, edges = pd.qcut(train["Fare"], q=4, retbins=True)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def add_fare_band(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    """Add ``FareBand`` (Low / Mid / High / Very High) from the given quartile edges.

    Quartiles give four equally sized groups, which suits a fare distribution
    with a long right tail; equal-width bins would put nearly everyone in the
    lowest band.
    """
    df["FareBand"] = pd.cut(df["Fare"], bins=edges, labels=FARE_LABELS)
    return df


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add all engineered features to cleaned train and test data.

    Args:
        train: Cleaned training data (from :func:`src.cleaning.clean`).
        test: Cleaned test data.

    Returns:
        ``(train_features, test_features)``. The inputs are not modified.
    """
    edges = fare_band_edges(train)
    out = []
    for df in (train.copy(), test.copy()):
        add_title(df)
        add_family_features(df)
        add_age_group(df)
        add_fare_band(df, edges)
        out.append(set_feature_dtypes(df))
    return out[0], out[1]


def set_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Restore cleaning dtypes and make the engineered labels ordered categories."""
    df = fix_dtypes(df)
    for col, levels in CATEGORICAL_FEATURES.items():
        if col in df:
            df[col] = pd.Categorical(df[col], categories=levels, ordered=(col != "Title"))
    for col in ("FamilySize", "IsAlone"):
        if col in df:
            df[col] = df[col].astype("int8")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the featured data into an all-numeric matrix.

    * ``Sex`` becomes ``IsFemale`` (0/1).
    * ``Pclass`` stays an integer 1–3, since its order is meaningful.
    * ``Embarked`` and ``Title`` are one-hot encoded with every level kept,
      so the correlation heatmap shows each level separately.
    * ``Deck`` is left out because ``HasCabin`` holds most of its signal and
      77% of decks are unknown.

    Identifiers and free text (``PassengerId``, ``Name``, ``Ticket``) are
    dropped. ``Survived`` is kept if present.

    Returns:
        A float DataFrame ready for ``.corr()`` or scikit-learn.
    """
    encoded = pd.DataFrame(index=df.index)
    if "Survived" in df:
        encoded["Survived"] = df["Survived"]
    encoded["Pclass"] = df["Pclass"].astype(int)
    encoded["IsFemale"] = (df["Sex"] == "female").astype(int)
    for col in ("Age", "SibSp", "Parch", "Fare", "LogFare", "FamilySize",
                "IsAlone", "HasCabin", "AgeImputed"):
        encoded[col] = df[col]
    encoded = encoded.join(pd.get_dummies(df["Embarked"], prefix="Embarked"))
    encoded = encoded.join(pd.get_dummies(df["Title"], prefix="Title"))
    return encoded.astype(float)


def load_features(name: str = "train", processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load ``<name>_features.csv`` from ``data/processed/`` with dtypes restored."""
    return set_feature_dtypes(pd.read_csv(processed_dir / f"{name}_features.csv"))


def save(train: pd.DataFrame, test: pd.DataFrame, processed_dir: Path = PROCESSED_DIR) -> None:
    """Write the featured train and test data to ``processed_dir``."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_dir / "train_features.csv", index=False)
    test.to_csv(processed_dir / "test_features.csv", index=False)


if __name__ == "__main__":
    train_feat, test_feat = build_features(load_processed("train"), load_processed("test"))
    save(train_feat, test_feat)

    print(f"train_features: {train_feat.shape} | test_features: {test_feat.shape}")
    print(f"FareBand edges (train quartiles): {np.round(fare_band_edges(load_processed('train')), 2)}")
    for col in ("Title", "AgeGroup", "FareBand", "FamilyGroup", "IsAlone", "Deck"):
        summary = train_feat.groupby(col, observed=True)["Survived"].agg(["count", "mean"])
        summary["mean"] = (summary["mean"] * 100).round(1)
        print(f"\n-- {col} (count, survival %):")
        print(summary.rename(columns={"mean": "survival_%"}).to_string())
    print(f"\nSaved to {PROCESSED_DIR}")
