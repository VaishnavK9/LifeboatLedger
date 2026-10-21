"""Clean the Titanic data: missing values, dtypes, duplicates and outliers.

Every step records what it did and why in a :class:`CleaningLog`, which is
saved next to the cleaned data and feeds the README's decisions table.

All imputation statistics are learned from the **training set only** and then
applied to both train and test, so no information leaks from test into train.

Run as a script to write ``data/processed/``::

    python -m src.cleaning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import PROCESSED_DIR
from src.data_loader import load_test, load_train

TITLE_MAP: dict[str, str] = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Mlle": "Miss",  # French "Mademoiselle"
    "Ms": "Miss",
    "Mme": "Mrs",  # French "Madame"
}
"""Common titles and their French/variant spellings. Anything else becomes ``"Rare"``."""

UNKNOWN_DECK = "U"
"""Deck label for passengers with no cabin recorded (and the single deck-T cabin)."""

CATEGORY_LEVELS: dict[str, list] = {
    "Pclass": [1, 2, 3],
    "Sex": ["male", "female"],
    "Embarked": ["S", "C", "Q"],
    "Deck": ["A", "B", "C", "D", "E", "F", "G", UNKNOWN_DECK],
}


@dataclass
class CleaningLog:
    """Ordered record of cleaning decisions."""

    entries: list[dict[str, object]] = field(default_factory=list)

    def add(self, step: str, column: str, action: str, rationale: str, affected: int) -> None:
        """Record a cleaning step.

        Args:
            step: Short name of the step.
            column: Column(s) the step touched.
            action: What was done.
            rationale: Why it was done.
            affected: Number of values changed across train + test.
        """
        self.entries.append(
            {"step": step, "column": column, "action": action,
             "rationale": rationale, "values_affected": affected}
        )

    def to_frame(self) -> pd.DataFrame:
        """Return the log as a DataFrame."""
        return pd.DataFrame(self.entries)


def extract_title(names: pd.Series) -> pd.Series:
    """Get the honorific from ``"Surname, Title. Given names"`` and group it.

    Variant spellings (Mlle, Ms, Mme) are mapped to their English equivalents,
    and the dozen titles with under 10 passengers (Dr, Rev, Col, Countess, …)
    are grouped into ``"Rare"`` so every group is large enough to be useful.

    Args:
        names: The ``Name`` column.

    Returns:
        One of ``Mr``, ``Mrs``, ``Miss``, ``Master`` or ``Rare`` per passenger.
    """
    raw = names.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    return raw.map(TITLE_MAP).fillna("Rare")


def check_duplicates(train: pd.DataFrame, test: pd.DataFrame, log: CleaningLog) -> None:
    """Make sure there are no duplicate rows or PassengerIds.

    Raises:
        ValueError: If duplicates are found, since they would need a manual decision.
    """
    combined = pd.concat([train, test], ignore_index=True)
    dup_rows = int(train.duplicated().sum() + test.duplicated().sum())
    dup_ids = int(combined["PassengerId"].duplicated().sum())
    if dup_rows or dup_ids:
        raise ValueError(f"Found {dup_rows} duplicate rows and {dup_ids} duplicate IDs")
    log.add("Duplicates", "all", "Checked rows and PassengerId across train + test; none found",
            "Duplicates would double-count passengers and bias survival rates", 0)


def impute_embarked(train: pd.DataFrame, test: pd.DataFrame, log: CleaningLog) -> None:
    """Fill the two missing ``Embarked`` values with the training mode (``"S"``).

    Both passengers (IDs 62 and 830) shared ticket 113572 and cabin B28.
    72% of passengers boarded at Southampton, and historical records confirm
    these two did too, so the mode is the right fill here.
    """
    mode = train["Embarked"].mode().iloc[0]
    n = int(train["Embarked"].isna().sum() + test["Embarked"].isna().sum())
    for df in (train, test):
        df["Embarked"] = df["Embarked"].fillna(mode)
    log.add("Missing values", "Embarked", f"Filled {n} missing with mode '{mode}'",
            "Only 2 rows; 72% boarded at S and records show these two did too", n)


def impute_fare(train: pd.DataFrame, test: pd.DataFrame, log: CleaningLog) -> None:
    """Fill missing ``Fare`` with the training median for the passenger's class.

    Fare depends heavily on class (1st-class median is ~£60 against ~£8 in
    3rd), so the class median is much closer than the overall mean (£32).
    """
    medians = train.groupby("Pclass")["Fare"].median()
    n = 0
    for df in (train, test):
        missing = df["Fare"].isna()
        n += int(missing.sum())
        df.loc[missing, "Fare"] = df.loc[missing, "Pclass"].map(medians)
    log.add("Missing values", "Fare", f"Filled {n} missing with Pclass median "
            f"(3rd class = £{medians[3]:.2f})",
            "Fare is driven by class; the overall mean (£32) is skewed by luxury tickets", n)


def impute_age(train: pd.DataFrame, test: pd.DataFrame, log: CleaningLog) -> None:
    """Fill missing ``Age`` with the median for the passenger's Title × Pclass group.

    Title carries age information ("Master" means a boy, typically under 13;
    "Miss" skews younger than "Mrs"), and class adds to it (1st-class
    passengers were older on average). The group median keeps those
    differences, whereas a single global median would flatten 20% of
    passengers to age 28. A Title-only median, then the overall median, are
    fallbacks for any group missing from train. An ``AgeImputed`` flag keeps
    track of which ages were estimated.
    """
    for df in (train, test):
        df["_Title"] = extract_title(df["Name"])
    group_median = train.groupby(["_Title", "Pclass"])["Age"].median()
    title_median = train.groupby("_Title")["Age"].median()
    overall = train["Age"].median()

    n = 0
    for df in (train, test):
        missing = df["Age"].isna()
        n += int(missing.sum())
        df["AgeImputed"] = missing.astype(int)
        keys = pd.MultiIndex.from_frame(df.loc[missing, ["_Title", "Pclass"]])
        fill = pd.Series(group_median.reindex(keys).to_numpy(), index=df.index[missing])
        fill = fill.fillna(df.loc[missing, "_Title"].map(title_median)).fillna(overall)
        df.loc[missing, "Age"] = fill
        df.drop(columns="_Title", inplace=True)
    log.add("Missing values", "Age", f"Filled {n} missing with Title × Pclass median; "
            "added AgeImputed flag",
            "Title and class predict age (e.g. Master = boys); a global median would "
            "flatten 20% of rows to 28", n)


def process_cabin(train: pd.DataFrame, test: pd.DataFrame, log: CleaningLog) -> None:
    """Replace the 77%-missing ``Cabin`` with ``HasCabin`` and ``Deck``, then drop it.

    The cabin number itself is too sparse to impute, but *whether* a cabin was
    recorded is informative (mostly 1st class), and its first letter gives the
    deck. The one deck-"T" cabin (a single passenger) is grouped into
    :data:`UNKNOWN_DECK` because a category with one member can't support
    any conclusion.
    """
    n_missing = 0
    for df in (train, test):
        n_missing += int(df["Cabin"].isna().sum())
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        deck = df["Cabin"].str[0]
        df["Deck"] = deck.where(deck.isin(CATEGORY_LEVELS["Deck"]), UNKNOWN_DECK)
        df.drop(columns="Cabin", inplace=True)
    log.add("Missing values", "Cabin", "Created HasCabin flag + Deck letter "
            f"('{UNKNOWN_DECK}' = unknown, incl. single deck-T); dropped raw Cabin",
            "77% missing: too sparse to impute, but presence and deck letter carry signal",
            n_missing)


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Return the Tukey fences ``(Q1 − k·IQR, Q3 + k·IQR)``."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def outlier_report(df: pd.DataFrame, columns: tuple[str, ...] = ("Age", "Fare")) -> pd.DataFrame:
    """Count IQR outliers per column.

    Returns:
        One row per column with the fences, outlier count and percentage, and
        the min and max values.
    """
    rows = []
    for col in columns:
        lo, hi = iqr_bounds(df[col])
        mask = (df[col] < lo) | (df[col] > hi)
        rows.append({"column": col, "lower_fence": round(lo, 2), "upper_fence": round(hi, 2),
                     "n_outliers": int(mask.sum()), "pct": round(mask.mean() * 100, 1),
                     "min": df[col].min(), "max": df[col].max()})
    return pd.DataFrame(rows).set_index("column")


def handle_outliers(train: pd.DataFrame, test: pd.DataFrame, log: CleaningLog,
                    report: pd.DataFrame) -> None:
    """Find Age/Fare outliers with the IQR rule and **keep** them.

    * **Fare:** about 13% of passengers fall above the upper fence (~£66), and
      they are real 1st-class tickets, not data errors. Capping them would
      erase the very class/wealth signal the EDA is about. Instead a
      ``LogFare`` column (``log1p``) tames the right skew for plots and models.
      The 15 zero fares (believed to be crew or complimentary tickets) are
      flagged with ``FareZero`` rather than changed.
    * **Age:** the outliers are a handful of passengers aged 65–80, which is
      perfectly plausible and so kept.

    Args:
        report: :func:`outlier_report` of the *raw* training data, computed
            before imputation so filled-in ages don't shift the fences.
    """
    for df in (train, test):
        df["LogFare"] = np.log1p(df["Fare"])
        df["FareZero"] = (df["Fare"] == 0).astype(int)
    fare = report.loc["Fare"]
    age = report.loc["Age"]
    log.add("Outliers", "Fare", f"Kept {int(fare.n_outliers)} IQR outliers (> £{fare.upper_fence}); "
            "added LogFare and FareZero flag",
            "Real luxury tickets: capping would erase the class signal; log1p handles skew",
            0)
    log.add("Outliers", "Age", f"Kept {int(age.n_outliers)} IQR outliers (> {age.upper_fence} yrs)",
            "Ages 65–80 are plausible, not data errors", 0)


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to their natural types.

    ``Pclass`` becomes an *ordered* category (1st > 2nd > 3rd), ``Sex``,
    ``Embarked`` and ``Deck`` become categories, ``Age`` stays float (infants
    have fractional ages such as 0.42), and the flags become ``int8``.
    """
    df = df.copy()
    for col, levels in CATEGORY_LEVELS.items():
        if col in df:
            df[col] = pd.Categorical(df[col], categories=levels, ordered=(col == "Pclass"))
    for col in ("Survived", "SibSp", "Parch", "HasCabin", "AgeImputed", "FareZero"):
        if col in df:
            df[col] = df[col].astype("int8")
    return df


def clean(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, CleaningLog]:
    """Run the full cleaning pipeline.

    Args:
        train: Raw training data.
        test: Raw test data.

    Returns:
        ``(train_clean, test_clean, log)``. The inputs are not modified.
    """
    train, test = train.copy(), test.copy()
    log = CleaningLog()
    raw_outliers = outlier_report(train)
    check_duplicates(train, test, log)
    impute_embarked(train, test, log)
    impute_fare(train, test, log)
    impute_age(train, test, log)
    process_cabin(train, test, log)
    handle_outliers(train, test, log, raw_outliers)
    train, test = fix_dtypes(train), fix_dtypes(test)
    log.add("Dtypes", "Pclass, Sex, Embarked, Deck, flags",
            "Pclass → ordered category; Sex/Embarked/Deck → category; flags → int8",
            "Correct types make category order explicit and prevent arithmetic on labels", 0)

    remaining = int(train.isna().sum().sum() + test.isna().sum().sum())
    if remaining:
        raise ValueError(f"{remaining} missing values remain after cleaning")
    return train, test, log


def load_processed(name: str = "train", processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load a cleaned CSV from ``data/processed/`` with dtypes restored.

    Args:
        name: ``"train"`` or ``"test"`` (reads ``<name>_clean.csv``).
        processed_dir: Folder holding the processed files.
    """
    return fix_dtypes(pd.read_csv(processed_dir / f"{name}_clean.csv"))


def save(train: pd.DataFrame, test: pd.DataFrame, log: CleaningLog,
         processed_dir: Path = PROCESSED_DIR) -> None:
    """Write the cleaned data and the cleaning log to ``processed_dir``."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_dir / "train_clean.csv", index=False)
    test.to_csv(processed_dir / "test_clean.csv", index=False)
    log.to_frame().to_csv(processed_dir / "cleaning_log.csv", index=False)


if __name__ == "__main__":
    raw_train, raw_test = load_train(), load_test()
    print("IQR outlier report (raw train):")
    print(outlier_report(raw_train).to_string(), end="\n\n")

    train_clean, test_clean, cleaning_log = clean(raw_train, raw_test)
    save(train_clean, test_clean, cleaning_log)

    pd.set_option("display.max_colwidth", 70, "display.width", 200)
    print("Cleaning log:")
    print(cleaning_log.to_frame()[["column", "action", "values_affected"]].to_string(), end="\n\n")
    print(f"train_clean: {train_clean.shape}, nulls={int(train_clean.isna().sum().sum())}")
    print(f"test_clean:  {test_clean.shape}, nulls={int(test_clean.isna().sum().sum())}")
    print(train_clean.dtypes.to_string())
    print(f"\nSaved to {PROCESSED_DIR}")
