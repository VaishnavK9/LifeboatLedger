"""Exploratory data analysis: charts saved to reports/figures/ and statistical tests.

Every plotting function returns its ``matplotlib`` Figure (so the notebook
shows it inline) and also saves it as a PNG in ``reports/figures/``.

Colour scheme (checked for colour-blind safety):

* **Survived** is blue and **Died** is orange.
* Single-series survival-rate bars are blue, with a dashed gray line marking
  the overall survival rate (38.4%).
* Heatmaps use a single blue scale (magnitude) or a blue↔red scale with a
  gray midpoint (correlation).

Run as a script to regenerate every figure and print the test results::

    python -m src.eda
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats

from src import FIGURES_DIR, REPORTS_DIR
from src.features import encode_features, load_features

# ── Style ────────────────────────────────────────────────────────────────────

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8985"
GRID = "#e6e5e1"

SURVIVED_COLOR = "#2a78d6"
DIED_COLOR = "#eb6834"
OUTCOME_PALETTE: dict[str, str] = {"Survived": SURVIVED_COLOR, "Died": DIED_COLOR}
BAR_COLOR = SURVIVED_COLOR

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("ledger_blues", BLUE_RAMP)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "ledger_diverging",
    ["#a82828", "#e34948", "#f3b4b3", "#f0efec", "#9ec5f4", "#2a78d6", "#104281"],
)

LABELS: dict[str, str] = {
    "Pclass": "Passenger class",
    "Sex": "Sex",
    "Embarked": "Port of embarkation",
    "Title": "Title",
    "AgeGroup": "Age group",
    "FamilySize": "Family size (incl. self)",
    "FamilyGroup": "Family group",
    "IsAlone": "Travelling alone",
    "Deck": "Deck",
    "FareBand": "Fare band (quartiles)",
    "HasCabin": "Cabin recorded",
}
TICK_LABELS: dict[str, dict] = {
    "Pclass": {1: "1st", 2: "2nd", 3: "3rd"},
    "Embarked": {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"},
    "IsAlone": {0: "With family", 1: "Alone"},
    "HasCabin": {0: "No", 1: "Yes"},
    "Deck": {"U": "Unknown"},
    "Sex": {"male": "Male", "female": "Female"},
}


def set_style() -> None:
    """Apply the project's matplotlib/seaborn style."""
    sns.set_theme(style="white", rc={
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "text.color": TEXT,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.family": "sans-serif",
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def save_fig(fig: Figure, name: str, figures_dir: Path = FIGURES_DIR) -> Path:
    """Save ``fig`` as ``<figures_dir>/<name>.png`` and return the path."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path)
    return path


def _outcome(df: pd.DataFrame) -> pd.Series:
    """``Survived`` as readable labels, with Survived first."""
    return pd.Categorical(df["Survived"].map({1: "Survived", 0: "Died"}),
                          categories=["Survived", "Died"])


def _tick_labels(col: str, values) -> list[str]:
    mapping = TICK_LABELS.get(col, {})
    return [str(mapping.get(v, v)) for v in values]


def _value_grid(ax: plt.Axes, axis: str = "y") -> None:
    """Draw a quiet value grid behind the marks."""
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _subtitle(ax: plt.Axes, text: str) -> None:
    """Small secondary-ink line under the axes title."""
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=9.5,
            color=TEXT_SECONDARY, va="bottom")


# ── Univariate ───────────────────────────────────────────────────────────────

def plot_survival_overview(df: pd.DataFrame) -> Figure:
    """How many passengers survived? Count and share of each outcome."""
    counts = pd.Series(_outcome(df)).value_counts().reindex(["Survived", "Died"])
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bars = ax.barh(counts.index, counts.values, height=0.55,
                   color=[OUTCOME_PALETTE[k] for k in counts.index])
    total = counts.sum()
    ax.bar_label(bars, labels=[f"{v} ({v / total:.1%})" for v in counts.values],
                 padding=6, color=TEXT, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_xlabel("Passengers")
    _value_grid(ax, "x")
    ax.set_title("Only 38% of passengers in the training set survived", pad=22)
    _subtitle(ax, f"n = {total} passengers (train.csv)")
    save_fig(fig, "01_survival_overview")
    return fig


def plot_categorical_distributions(df: pd.DataFrame,
                                   cols: tuple[str, ...] = ("Pclass", "Sex", "Embarked")) -> Figure:
    """Who was on board? Passenger counts for class, sex and port."""
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.8))
    for ax, col in zip(axes, cols):
        counts = df[col].value_counts(sort=False)
        bars = ax.bar(_tick_labels(col, counts.index), counts.values, width=0.6, color=BAR_COLOR)
        ax.bar_label(bars, labels=[f"{v}\n{v / counts.sum():.0%}" for v in counts.values],
                     padding=3, fontsize=9, color=TEXT)
        ax.set_ylim(0, counts.max() * 1.25)
        ax.set_title(LABELS[col])
        ax.set_ylabel("Passengers" if ax is axes[0] else "")
        _value_grid(ax)
    fig.suptitle("Most passengers were 3rd-class men who boarded at Southampton",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout()
    save_fig(fig, "02_categorical_distributions")
    return fig


def plot_age_distribution(df: pd.DataFrame) -> Figure:
    """What was the age profile? Histogram with KDE, and densities by outcome.

    Only **recorded** ages are used (``AgeImputed == 0``). The 177 imputed
    ages all sit on a few group medians and would create false spikes.
    """
    observed = df[df["AgeImputed"] == 0].assign(Outcome=lambda d: _outcome(d))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(observed, x="Age", bins=range(0, 85, 5), kde=True, color=BAR_COLOR,
                 edgecolor=SURFACE, linewidth=1.5, alpha=0.85, ax=ax1,
                 line_kws={"linewidth": 2})
    ax1.axvline(observed["Age"].median(), color=TEXT_SECONDARY, linestyle="--", linewidth=1.2)
    ax1.text(observed["Age"].median() + 1, ax1.get_ylim()[1] * 0.92,
             f"median {observed['Age'].median():.0f}", color=TEXT_SECONDARY, fontsize=9)
    ax1.set(xlabel="Age (years)", ylabel="Passengers")
    ax1.set_title("Ages, histogram + KDE")
    _value_grid(ax1)

    sns.kdeplot(observed, x="Age", hue="Outcome", palette=OUTCOME_PALETTE, common_norm=False,
                fill=True, alpha=0.15, linewidth=2, clip=(0, 80), ax=ax2)
    ax2.set(xlabel="Age (years)", ylabel="Density")
    ax2.set_title("Children were more likely to survive")
    ax2.get_legend().set_title(None)
    _value_grid(ax2)
    fig.suptitle(f"Age distribution: n = {len(observed)} recorded ages (imputed excluded)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout()
    save_fig(fig, "03_age_distribution")
    return fig


def plot_fare_distribution(df: pd.DataFrame) -> Figure:
    """How were fares distributed? Raw (heavily skewed) against log scale."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df, x="Fare", bins=50, color=BAR_COLOR, edgecolor=SURFACE,
                 linewidth=1, alpha=0.85, ax=ax1)
    ax1.set(xlabel="Fare (£)", ylabel="Passengers")
    ax1.set_title(f"Raw fare is extremely right-skewed (skew = {df['Fare'].skew():.1f})")
    _value_grid(ax1)

    plot_df = df.assign(Outcome=_outcome(df))
    sns.histplot(plot_df, x="LogFare", hue="Outcome", palette=OUTCOME_PALETTE, bins=30,
                 multiple="stack", edgecolor=SURFACE, linewidth=1, alpha=0.9, ax=ax2)
    ticks = [0, 5, 10, 25, 50, 100, 250, 500]
    ax2.set_xticks(np.log1p(ticks), [f"£{t}" for t in ticks])
    ax2.set(xlabel="Fare (log scale)", ylabel="Passengers")
    ax2.set_title("On a log scale, higher fares mean more survivors")
    ax2.get_legend().set_title(None)
    _value_grid(ax2)
    fig.suptitle("Fare distribution", x=0.01, ha="left", fontsize=14,
                 fontweight="bold", color=TEXT)
    fig.tight_layout()
    save_fig(fig, "04_fare_distribution")
    return fig


# ── Bivariate ────────────────────────────────────────────────────────────────

def survival_rates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Survival rate, survivors and passenger count per level of ``col``."""
    grouped = df.groupby(col, observed=True)["Survived"]
    return pd.DataFrame({"rate": grouped.mean(), "survivors": grouped.sum(),
                         "n": grouped.size()})


def survival_bar(df: pd.DataFrame, col: str, ax: plt.Axes | None = None,
                 title: str | None = None) -> plt.Axes:
    """Bar chart of the survival rate per level of ``col``, with % labels.

    Each bar is labelled with its survival rate (``n`` sits under the tick
    label, so it stays readable on narrow or 0% bars), and a dashed line
    marks the overall rate, so above- and below-average groups stand out.
    """
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    rates = survival_rates(df, col)
    ticks = [f"{label.replace(' (', chr(10) + '(')}\nn={n}"
             for label, n in zip(_tick_labels(col, rates.index), rates["n"])]
    bars = ax.bar(ticks, rates["rate"], width=0.62, color=BAR_COLOR)
    ax.bar_label(bars, labels=[f"{r:.0%}" for r in rates["rate"]], padding=3,
                 fontsize=9.5, color=TEXT, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8.5 if len(rates) <= 7 else 7.5)
    overall = df["Survived"].mean()
    ax.axhline(overall, color=TEXT_MUTED, linestyle="--", linewidth=1.1)
    ax.text(1.0, overall, f" overall {overall:.0%}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set(xlabel=LABELS.get(col, col), ylabel="Survival rate")
    ax.set_title(title or f"Survival by {LABELS.get(col, col).lower()}")
    _value_grid(ax)
    return ax


BIVARIATE_COLS: list[str] = ["Sex", "Pclass", "AgeGroup", "Embarked",
                             "Title", "FamilySize", "IsAlone", "Deck"]


def plot_survival_by(df: pd.DataFrame, col: str, title: str | None = None) -> Figure:
    """Single survival-rate chart for ``col``, saved as ``05_survival_by_<col>``."""
    width = 7.5 if df[col].nunique() > 5 else 6
    fig, ax = plt.subplots(figsize=(width, 4))
    survival_bar(df, col, ax=ax, title=title)
    save_fig(fig, f"05_survival_by_{col.lower()}")
    return fig


def plot_bivariate_grid(df: pd.DataFrame, cols: list[str] | None = None) -> Figure:
    """Overview grid: survival rate against every key feature at once."""
    cols = cols or BIVARIATE_COLS
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, col in zip(axes.flat, cols):
        survival_bar(df, col, ax=ax)
        ax.set_ylabel("Survival rate" if ax in axes[:, 0] else "")
    fig.suptitle("Survival rate by feature (dashed line = overall 38%)", x=0.01, ha="left",
                 fontsize=15, fontweight="bold", color=TEXT)
    fig.tight_layout(h_pad=2.5, w_pad=3)
    save_fig(fig, "05_survival_by_feature_grid")
    return fig


# ── Multivariate ─────────────────────────────────────────────────────────────

def plot_sex_class_heatmap(df: pd.DataFrame) -> Figure:
    """How do sex and class combine? Survival rate for each Sex × Pclass cell."""
    rates = df.pivot_table(index="Sex", columns="Pclass", values="Survived",
                           aggfunc="mean", observed=True).reindex(["female", "male"])
    counts = df.pivot_table(index="Sex", columns="Pclass", values="Survived",
                            aggfunc="size", observed=True).reindex(["female", "male"])
    annot = rates.map(lambda r: f"{r:.0%}") + counts.map(lambda n: f"\nn={n}")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.heatmap(rates, annot=annot, fmt="", cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1,
                linewidths=2, linecolor=SURFACE, annot_kws={"fontsize": 12},
                cbar_kws={"format": PercentFormatter(1.0, decimals=0), "label": "Survival rate"},
                ax=ax)
    ax.set_xticklabels(_tick_labels("Pclass", rates.columns))
    ax.set_yticklabels(_tick_labels("Sex", rates.index), rotation=0)
    ax.set(xlabel="Passenger class", ylabel="")
    ax.set_title(f"1st-class women: {rates.loc['female', 1]:.0%} survived. "
                 f"3rd-class men: {rates.loc['male', 3]:.0%}", pad=12)
    save_fig(fig, "06_sex_class_heatmap")
    return fig


def plot_age_fare_scatter(df: pd.DataFrame) -> Figure:
    """Do age and fare together separate survivors? Scatter on a log fare axis.

    Only recorded ages are plotted. The 15 zero fares sit on the bottom edge
    of the symlog axis.
    """
    observed = df[df["AgeImputed"] == 0].assign(Outcome=lambda d: _outcome(d))
    fig, ax = plt.subplots(figsize=(9, 5))
    for outcome in ["Died", "Survived"]:
        part = observed[observed["Outcome"] == outcome]
        ax.scatter(part["Age"], part["Fare"], s=26, color=OUTCOME_PALETTE[outcome],
                   alpha=0.75, edgecolor=SURFACE, linewidth=0.8,
                   label=f"{outcome} (n={len(part)})")
    ax.set_yscale("symlog", linthresh=10)
    ticks = [0, 5, 10, 25, 50, 100, 250, 500]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"£{v:.0f}"))
    ax.set(xlabel="Age (years)", ylabel="Fare (log scale)")
    ax.legend(loc="upper right")
    _value_grid(ax, "both")
    ax.set_title("Survivors cluster at higher fares; young children survive at every fare", pad=22)
    _subtitle(ax, f"n = {len(observed)} passengers with a recorded age")
    save_fig(fig, "07_age_fare_scatter")
    return fig


def plot_facet_sex_class(df: pd.DataFrame) -> sns.FacetGrid:
    """Age distribution by outcome in a FacetGrid: rows = Sex, columns = Pclass."""
    observed = df[df["AgeImputed"] == 0].assign(
        Outcome=lambda d: _outcome(d),
        SexLabel=lambda d: d["Sex"].map(TICK_LABELS["Sex"]),
        ClassLabel=lambda d: d["Pclass"].map(TICK_LABELS["Pclass"]),
    )
    # hue goes to histplot (not FacetGrid) so the outcomes stack instead of overlapping
    grid = sns.FacetGrid(observed, row="SexLabel", col="ClassLabel",
                         row_order=["Female", "Male"], col_order=["1st", "2nd", "3rd"],
                         height=2.8, aspect=1.35, margin_titles=True, despine=True)
    grid.map_dataframe(sns.histplot, x="Age", hue="Outcome", hue_order=["Survived", "Died"],
                       palette=OUTCOME_PALETTE, bins=range(0, 85, 5), multiple="stack",
                       edgecolor=SURFACE, linewidth=1, alpha=0.9, legend=False)
    grid.set_titles(row_template="{row_name}", col_template="{col_name} class")
    grid.set_axis_labels("Age (years)", "Passengers")
    grid.add_legend(title="", legend_data={
        k: plt.Rectangle((0, 0), 1, 1, color=c) for k, c in OUTCOME_PALETTE.items()})
    for ax in grid.axes.flat:
        _value_grid(ax)
    grid.figure.suptitle("Age × outcome, split by sex and class", x=0.01, ha="left",
                         fontsize=14, fontweight="bold", color=TEXT)
    grid.figure.subplots_adjust(top=0.88)
    save_fig(grid.figure, "08_facet_sex_class")
    return grid


def plot_correlation_heatmap(df: pd.DataFrame) -> Figure:
    """Pearson correlation of the numeric and encoded features (lower triangle)."""
    corr = encode_features(df).drop(columns=["Fare"]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"fontsize": 7.5}, square=True,
                linewidths=1, linecolor=SURFACE,
                cbar_kws={"shrink": 0.7, "label": "Pearson r"}, ax=ax)
    target = corr["Survived"].drop("Survived")
    top_pos, top_neg = target.idxmax(), target.idxmin()
    ax.set_title(f"Correlation heatmap: {top_pos} ({target[top_pos]:+.2f}) and {top_neg} "
                 f"({target[top_neg]:+.2f}) track Survived most closely", pad=12)
    save_fig(fig, "09_correlation_heatmap")
    return fig


# ── Statistical tests ────────────────────────────────────────────────────────

CHI_SQUARE_COLS: list[str] = ["Sex", "Pclass", "Embarked", "Title", "AgeGroup",
                              "FareBand", "FamilyGroup", "IsAlone", "HasCabin", "Deck"]


def describe_p(p: float, alpha: float = 0.05) -> str:
    """Describe a p-value in plain English."""
    if p < 0.001:
        return "very strong evidence of a real difference"
    if p < 0.01:
        return "strong evidence of a real difference"
    if p < alpha:
        return "moderate evidence of a real difference"
    return "no significant evidence of a difference"


def _strength(value: float, cutoffs: tuple[float, float, float]) -> str:
    small, medium, large = cutoffs
    if value < small:
        return "negligible"
    if value < medium:
        return "small"
    if value < large:
        return "medium"
    return "large"


def chi_square_tests(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Chi-square test of independence between each categorical column and ``Survived``.

    Also reports **Cramér's V** (0 = no association, 1 = perfect) as the
    effect size, because with 891 rows even weak links can be "significant".
    ``min_expected`` flags cells below 5, where the chi-square approximation
    is less reliable (for example the small deck G).
    """
    rows = []
    for col in cols or CHI_SQUARE_COLS:
        table = pd.crosstab(df[col], df["Survived"])
        chi2, p, dof, expected = stats.chi2_contingency(table)
        k = min(table.shape) - 1
        v = float(np.sqrt(chi2 / (table.to_numpy().sum() * k)))
        strength = _strength(v, (0.1, 0.3, 0.5))
        min_expected = float(expected.min())
        caveat = (" Caution: some expected counts are below 5, so treat this p-value as "
                  "approximate." if min_expected < 5 else "")
        rows.append({
            "feature": col, "chi2": round(chi2, 2), "dof": dof, "p_value": p,
            "cramers_v": round(v, 3), "min_expected": round(min_expected, 1),
            "plain_english": f"{describe_p(p).capitalize()} in survival across {col} "
                             f"({strength} effect, V = {v:.2f}).{caveat}",
        })
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def numeric_tests(df: pd.DataFrame, cols: tuple[str, ...] = ("Age", "Fare")) -> pd.DataFrame:
    """Compare survivors and non-survivors on numeric columns.

    * **Welch's t-test** compares means without assuming equal variances.
    * **Mann-Whitney U** compares distributions using ranks, so it is robust
      to Fare's heavy skew and outliers. Treat it as the primary test for Fare.
    * **Rank-biserial r** is the Mann-Whitney effect size (positive means
      survivors tend to have higher values).

    For Age, only recorded values are used, so imputed medians don't dilute
    the difference.
    """
    rows = []
    for col in cols:
        data = df[df["AgeImputed"] == 0] if col == "Age" else df
        surv = data.loc[data["Survived"] == 1, col]
        died = data.loc[data["Survived"] == 0, col]
        t_stat, t_p = stats.ttest_ind(surv, died, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(surv, died, alternative="two-sided")
        r = 2 * u_stat / (len(surv) * len(died)) - 1
        if surv.median() == died.median():
            direction = "the same median"
        else:
            direction = "higher" if surv.median() > died.median() else "lower"
        rows.append({
            "feature": col, "n": len(data),
            "mean_survived": round(surv.mean(), 2), "mean_died": round(died.mean(), 2),
            "median_survived": round(surv.median(), 2), "median_died": round(died.median(), 2),
            "welch_t": round(t_stat, 2), "t_p_value": t_p,
            "mann_whitney_u": round(u_stat, 1), "mw_p_value": u_p,
            "rank_biserial_r": round(r, 3),
            "plain_english": (
                f"Survivors' median {col} was {surv.median():.1f} against {died.median():.1f} "
                f"for those who died ({direction}). Mann-Whitney p = {u_p:.2g}: "
                f"{describe_p(u_p)} ({_strength(abs(r), (0.1, 0.3, 0.5))} effect, r = {r:.2f})."
            ),
        })
    return pd.DataFrame(rows)


def save_stats(chi: pd.DataFrame, num: pd.DataFrame, reports_dir: Path = REPORTS_DIR) -> None:
    """Write the test results to ``reports/`` as CSV."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    chi.to_csv(reports_dir / "chi_square_tests.csv", index=False)
    num.to_csv(reports_dir / "numeric_tests.csv", index=False)


# ── Orchestration ────────────────────────────────────────────────────────────

def run_all(df: pd.DataFrame | None = None, close: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate every figure and run every test.

    Args:
        df: Featured training data. Loaded from ``data/processed/`` if omitted.
        close: Close each figure after saving (set ``False`` in a notebook).

    Returns:
        ``(chi_square_results, numeric_test_results)``.
    """
    set_style()
    df = load_features("train") if df is None else df
    figures = [
        plot_survival_overview(df),
        plot_categorical_distributions(df),
        plot_age_distribution(df),
        plot_fare_distribution(df),
        *[plot_survival_by(df, col) for col in BIVARIATE_COLS],
        plot_bivariate_grid(df),
        plot_sex_class_heatmap(df),
        plot_age_fare_scatter(df),
        plot_facet_sex_class(df).figure,
        plot_correlation_heatmap(df),
    ]
    if close:
        for fig in figures:
            plt.close(fig)
    chi, num = chi_square_tests(df), numeric_tests(df)
    save_stats(chi, num)
    return chi, num


if __name__ == "__main__":
    chi_results, num_results = run_all()
    print(f"Saved {len(list(FIGURES_DIR.glob('*.png')))} figures to {FIGURES_DIR}\n")
    pd.set_option("display.width", 220, "display.max_colwidth", 120)
    print("Chi-square tests (categorical vs Survived):")
    print(chi_results[["feature", "chi2", "dof", "p_value", "cramers_v", "min_expected"]]
          .to_string(), end="\n\n")
    for line in chi_results["plain_english"]:
        print(" •", line)
    print("\nAge & Fare: Welch t-test and Mann-Whitney U:")
    print(num_results.drop(columns="plain_english").to_string(), end="\n\n")
    for line in num_results["plain_english"]:
        print(" •", line)
