"""Baseline Logistic Regression and Random Forest on the engineered features.

This is a quick check that the EDA findings carry predictive weight, not a
tuned model. Each model is scored with stratified 5-fold cross-validation on
the training set in two settings:

* **raw**: only the original columns (Pclass, Sex, Age, SibSp, Parch, Fare).
* **engineered**: raw columns plus Title, FamilySize, IsAlone, HasCabin and
  Embarked.

The "all women survive" rule (Kaggle's ``gender_submission.csv``) is the
reference score to beat.

Run as a script::

    python -m src.model_baseline
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import REPORTS_DIR
from src.eda import BAR_COLOR, DIED_COLOR, SURVIVED_COLOR, TEXT, _value_grid, save_fig, set_style
from src.features import encode_features, load_features

RANDOM_STATE = 42

RAW_FEATURES: list[str] = ["Pclass", "IsFemale", "Age", "SibSp", "Parch", "LogFare"]
ENGINEERED_FEATURES: list[str] = RAW_FEATURES + [
    "FamilySize", "IsAlone", "HasCabin",
    "Embarked_C", "Embarked_Q",  # Embarked_S is the reference level
    "Title_Mr", "Title_Mrs", "Title_Miss", "Title_Master",  # Title_Rare is the reference
]


def make_models() -> dict[str, object]:
    """The two baseline classifiers.

    Logistic Regression is wrapped with a ``StandardScaler`` so its
    coefficients are comparable across features. The Random Forest is kept
    shallow (``max_depth=6``) to limit overfitting on 891 rows.
    """
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=RANDOM_STATE),
    }


def gender_rule_accuracy(df: pd.DataFrame) -> float:
    """Accuracy of predicting "survived" for every woman and "died" for every man."""
    return float(((df["Sex"] == "female").astype(int) == df["Survived"]).mean())


def evaluate(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Cross-validate both models on the raw and engineered feature sets.

    Returns:
        One row per (model, feature set) with mean and std CV accuracy.
    """
    encoded = encode_features(df)
    y = encoded["Survived"].astype(int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for set_name, cols in (("raw", RAW_FEATURES), ("engineered", ENGINEERED_FEATURES)):
        for model_name, model in make_models().items():
            scores = cross_val_score(model, encoded[cols], y, cv=cv, scoring="accuracy")
            rows.append({"model": model_name, "features": set_name, "n_features": len(cols),
                         "cv_accuracy": round(scores.mean(), 4),
                         "cv_std": round(scores.std(), 4)})
    return pd.DataFrame(rows)


def feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Fit both models on all training rows with the engineered features.

    Returns:
        A DataFrame indexed by feature with the standardised LR coefficient
        (sign = direction) and the RF impurity importance (share of splits).
    """
    encoded = encode_features(df)
    X, y = encoded[ENGINEERED_FEATURES], encoded["Survived"].astype(int)
    models = make_models()
    lr = models["Logistic Regression"].fit(X, y)
    rf = models["Random Forest"].fit(X, y)
    return pd.DataFrame({
        "lr_coefficient": lr[-1].coef_[0],
        "rf_importance": rf.feature_importances_,
    }, index=ENGINEERED_FEATURES).sort_values("rf_importance", ascending=False)


def plot_feature_importance(importance: pd.DataFrame) -> Figure:
    """Side by side: RF importance (magnitude) and LR coefficients (direction)."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    rf = importance["rf_importance"].sort_values()
    bars = ax1.barh(rf.index, rf.values, height=0.65, color=BAR_COLOR)
    ax1.bar_label(bars, labels=[f"{v:.1%}" for v in rf.values], padding=3,
                  fontsize=8.5, color=TEXT)
    ax1.set_xlim(0, rf.max() * 1.2)
    ax1.set_xlabel("Share of impurity reduction")
    ax1.set_title("Random Forest: feature importance")
    _value_grid(ax1, "x")

    lr = importance["lr_coefficient"].reindex(rf.index)
    colors = [SURVIVED_COLOR if v > 0 else DIED_COLOR for v in lr.values]
    ax2.barh(lr.index, lr.values, height=0.65, color=colors)
    ax2.axvline(0, color=TEXT, linewidth=0.8)
    ax2.set_xlabel("Standardised coefficient (blue: raises survival odds, orange: lowers them)")
    ax2.set_title("Logistic Regression: coefficients")
    _value_grid(ax2, "x")

    fig.suptitle("Sex, Title, class and fare drive the predictions, matching the EDA",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout()
    save_fig(fig, "10_feature_importance")
    return fig


def run() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Evaluate, compute importance, plot, and save results to ``reports/``.

    Returns:
        ``(cv_results, importance, gender_rule_accuracy)``.
    """
    df = load_features("train")
    results = evaluate(df)
    importance = feature_importance(df)
    plot_feature_importance(importance)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "model_baseline_results.csv", index=False)
    importance.round(4).to_csv(REPORTS_DIR / "feature_importance.csv")
    return results, importance, gender_rule_accuracy(df)


if __name__ == "__main__":
    cv_results, imp, rule_acc = run()
    print(f"Reference: 'all women survive' rule accuracy on train = {rule_acc:.2%}\n")
    print("Stratified 5-fold CV accuracy:")
    print(cv_results.to_string(index=False), end="\n\n")
    print("Feature importance (engineered set):")
    print(imp.round(3).to_string())
    best = cv_results.loc[cv_results["cv_accuracy"].idxmax()]
    print(f"\nBest: {best['model']} on {best['features']} features = "
          f"{best['cv_accuracy']:.2%} (±{best['cv_std']:.2%})")
    plt.close("all")
