"""Lifeboat Ledger: interactive Streamlit dashboard for the Titanic EDA.

Reuses the project pipeline in ``src/`` so every number matches the notebook.

Run locally::

    streamlit run app.py
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import PROCESSED_DIR
from src.cleaning import clean
from src.data_loader import load_test, load_train
from src.eda import (
    BLUE_RAMP, DIED_COLOR, LABELS, SURVIVED_COLOR, TEXT_MUTED, TICK_LABELS,
    chi_square_tests, numeric_tests, survival_rates,
)
from src.features import (
    TITLE_ORDER, build_features, encode_features, load_features, set_feature_dtypes,
)
from src.model_baseline import (
    ENGINEERED_FEATURES, evaluate, feature_importance, gender_rule_accuracy, make_models,
)

st.set_page_config(page_title="Lifeboat Ledger", page_icon="🛟", layout="wide")

OUTCOME_SCALE = alt.Scale(domain=["Survived", "Died"], range=[SURVIVED_COLOR, DIED_COLOR])
EXPLORE_FEATURES: list[str] = ["Sex", "Pclass", "AgeGroup", "Title", "Embarked",
                               "FamilySize", "FamilyGroup", "IsAlone", "FareBand", "Deck",
                               "HasCabin"]


# ── Data & model (cached) ────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    """Featured training data; rebuilt from the raw CSVs if the processed file is missing."""
    if (PROCESSED_DIR / "train_features.csv").exists():
        return load_features("train")
    train_clean, test_clean, _ = clean(load_train(), load_test())
    return build_features(train_clean, test_clean)[0]


@st.cache_data
def load_cleaning_log() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "cleaning_log.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def run_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return chi_square_tests(df), numeric_tests(df)


@st.cache_data
def run_model_eval(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    return evaluate(df), feature_importance(df), gender_rule_accuracy(df)


@st.cache_resource
def fit_model(df: pd.DataFrame):
    """Random Forest fitted on all training rows with the engineered features."""
    encoded = encode_features(df)
    return make_models()["Random Forest"].fit(encoded[ENGINEERED_FEATURES],
                                              encoded["Survived"].astype(int))


# ── Helpers ──────────────────────────────────────────────────────────────────

def label(col: str, value) -> str:
    """Readable label for a category value (e.g. Pclass 1 → '1st')."""
    return str(TICK_LABELS.get(col, {}).get(value, value))


def with_outcome(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(Outcome=df["Survived"].map({1: "Survived", 0: "Died"}))


def survival_bar(df: pd.DataFrame, col: str, overall: float) -> alt.LayerChart:
    """Survival rate per level of ``col`` with % labels, tooltips and an overall-rate rule."""
    rates = survival_rates(df, col).reset_index()
    rates["level"] = [label(col, v) for v in rates[col]]
    order = rates["level"].tolist()
    base = alt.Chart(rates).encode(
        x=alt.X("level:N", sort=order, title=LABELS.get(col, col),
                axis=alt.Axis(labelAngle=0, labelLimit=120)),
        y=alt.Y("rate:Q", title="Survival rate", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("level:N", title=LABELS.get(col, col)),
                 alt.Tooltip("rate:Q", title="Survival rate", format=".1%"),
                 alt.Tooltip("survivors:Q", title="Survivors"),
                 alt.Tooltip("n:Q", title="Passengers")],
    )
    bars = base.mark_bar(color=SURVIVED_COLOR, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    # Labels sit inside the bar in white (readable in light and dark themes); very short
    # bars get a mid-gray label above them instead.
    inside = base.transform_filter(alt.datum.rate >= 0.08).mark_text(
        dy=11, fontWeight="bold", color="white").encode(text=alt.Text("rate:Q", format=".0%"))
    above = base.transform_filter(alt.datum.rate < 0.08).mark_text(
        dy=-9, fontWeight="bold", color=TEXT_MUTED).encode(text=alt.Text("rate:Q", format=".0%"))
    text = inside + above
    rule = alt.Chart(pd.DataFrame({"y": [overall]})).mark_rule(
        strokeDash=[4, 4], color=TEXT_MUTED).encode(
        y="y:Q", tooltip=[alt.Tooltip("y:Q", title="Overall survival", format=".1%")])
    return (bars + text + rule).properties(height=340)


def outcome_counts(df: pd.DataFrame, col: str) -> alt.Chart:
    """Stacked passenger counts per level of ``col``, split by outcome."""
    counts = with_outcome(df).groupby([col, "Outcome"], observed=True).size().reset_index(name="n")
    counts["level"] = [label(col, v) for v in counts[col]]
    order = [label(col, v) for v in df[col].drop_duplicates().sort_values()]
    return alt.Chart(counts).mark_bar(stroke="white", strokeWidth=1).encode(
        x=alt.X("level:N", sort=order, title=LABELS.get(col, col),
                axis=alt.Axis(labelAngle=0, labelLimit=120)),
        y=alt.Y("n:Q", title="Passengers"),
        color=alt.Color("Outcome:N", scale=OUTCOME_SCALE, legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("Outcome:N", sort="descending"),
        tooltip=[alt.Tooltip("level:N", title=LABELS.get(col, col)), "Outcome:N",
                 alt.Tooltip("n:Q", title="Passengers")],
    ).properties(height=340)


def heatmap(df: pd.DataFrame, col: str) -> alt.LayerChart:
    """Survival-rate heatmap: Sex (rows) × ``col`` (columns)."""
    grid = df.groupby(["Sex", col], observed=True)["Survived"].agg(rate="mean", n="size").reset_index()
    grid["sex"] = [label("Sex", v) for v in grid["Sex"]]
    grid["level"] = [label(col, v) for v in grid[col]]
    grid["text"] = [f"{r:.0%}  (n={n})" for r, n in zip(grid["rate"], grid["n"])]
    order = [label(col, v) for v in df[col].drop_duplicates().sort_values()]
    base = alt.Chart(grid).encode(
        x=alt.X("level:N", sort=order, title=LABELS.get(col, col), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("sex:N", sort=["Female", "Male"], title=None),
    )
    rect = base.mark_rect(stroke="white", strokeWidth=3, cornerRadius=4).encode(
        color=alt.Color("rate:Q", scale=alt.Scale(domain=[0, 1], range=BLUE_RAMP),
                        legend=alt.Legend(format="%", title="Survival rate")),
        tooltip=[alt.Tooltip("sex:N", title="Sex"), alt.Tooltip("level:N", title=LABELS.get(col, col)),
                 alt.Tooltip("rate:Q", title="Survival rate", format=".1%"),
                 alt.Tooltip("n:Q", title="Passengers")],
    )
    text = base.mark_text(fontSize=14, fontWeight="bold").encode(
        text="text:N",
        color=alt.condition(alt.datum.rate > 0.55, alt.value("white"), alt.value("#0b0b0b")),
    )
    return (rect + text).properties(height=260)


def age_histogram(df: pd.DataFrame) -> alt.Chart:
    observed = with_outcome(df[df["AgeImputed"] == 0])
    return alt.Chart(observed).mark_bar(stroke="white", strokeWidth=1).encode(
        x=alt.X("Age:Q", bin=alt.Bin(step=5, extent=[0, 80]), title="Age (years)"),
        y=alt.Y("count():Q", title="Passengers"),
        color=alt.Color("Outcome:N", scale=OUTCOME_SCALE, legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("Outcome:N", sort="descending"),
        tooltip=[alt.Tooltip("Age:Q", bin=alt.Bin(step=5, extent=[0, 80]), title="Age band"),
                 "Outcome:N", alt.Tooltip("count():Q", title="Passengers")],
    ).properties(height=320)


def age_fare_scatter(df: pd.DataFrame) -> alt.Chart:
    observed = with_outcome(df[df["AgeImputed"] == 0])
    observed = observed.assign(Class=[label("Pclass", v) for v in observed["Pclass"]],
                               SexLabel=[label("Sex", v) for v in observed["Sex"]],
                               FareLabel=[f"£{v:.2f}" for v in observed["Fare"]])
    return alt.Chart(observed).mark_circle(size=55, opacity=0.75, stroke="white",
                                           strokeWidth=0.8).encode(
        x=alt.X("Age:Q", title="Age (years)"),
        y=alt.Y("Fare:Q", title="Fare (£, symlog scale)",
                scale=alt.Scale(type="symlog", constant=10),
                axis=alt.Axis(values=[0, 5, 10, 25, 50, 100, 250, 500])),
        color=alt.Color("Outcome:N", scale=OUTCOME_SCALE, legend=alt.Legend(orient="top", title=None)),
        tooltip=["Name:N", alt.Tooltip("SexLabel:N", title="Sex"), "Class:N", "Age:Q",
                 alt.Tooltip("FareLabel:N", title="Fare"), "Outcome:N"],
    ).properties(height=380).interactive()


# ── Load ─────────────────────────────────────────────────────────────────────

data = load_data()
OVERALL = float(data["Survived"].mean())

# ── Sidebar filters ──────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛟 Lifeboat Ledger")
    st.caption("Titanic data cleaning & EDA · Prodigy InfoTech Data Science Internship – Task 02")
    st.divider()
    st.subheader("Filter passengers")
    sexes = st.multiselect("Sex", ["female", "male"], default=["female", "male"],
                           format_func=lambda v: label("Sex", v))
    classes = st.multiselect("Passenger class", [1, 2, 3], default=[1, 2, 3],
                             format_func=lambda v: label("Pclass", v))
    ports = st.multiselect("Port of embarkation", ["S", "C", "Q"], default=["S", "C", "Q"],
                           format_func=lambda v: label("Embarked", v))
    age_range = st.slider("Age range", 0, 80, (0, 80))
    include_imputed = st.checkbox("Include imputed ages", value=True,
                                  help="177 ages were estimated from Title × Class medians during cleaning.")
    st.divider()
    st.markdown("Built by **Vaishnav Kumbhar** · [GitHub @VaishnavK9](https://github.com/VaishnavK9)")

mask = (data["Sex"].isin(sexes) & data["Pclass"].isin(classes) & data["Embarked"].isin(ports)
        & data["Age"].between(*age_range))
if not include_imputed:
    mask &= data["AgeImputed"] == 0
df = data[mask]

# ── Header ───────────────────────────────────────────────────────────────────

st.title("🛟 Lifeboat Ledger")
st.markdown("#### Who survived the Titanic, and why?")
if len(df) < len(data):
    st.info(f"Filters active: showing **{len(df)}** of {len(data)} passengers. "
            "Statistics and the model always use the full training set.")

tab_overview, tab_explore, tab_combine, tab_stats, tab_predict, tab_data = st.tabs(
    ["📊 Overview", "🔍 Explore", "🔥 Sex × Feature", "🧪 Statistics", "🤖 Predict", "🧹 Data & Cleaning"])

# ── Overview ─────────────────────────────────────────────────────────────────

with tab_overview:
    if df.empty:
        st.warning("No passengers match the current filters.")
    else:
        women, men = df[df["Sex"] == "female"], df[df["Sex"] == "male"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Passengers", f"{len(df):,}")
        c2.metric("Survival rate", f"{df['Survived'].mean():.1%}",
                  delta=(f"{(df['Survived'].mean() - OVERALL) * 100:+.1f} pts vs all"
                         if len(df) < len(data) else None))
        c3.metric("Women survived", f"{women['Survived'].mean():.0%}" if len(women) else "–")
        c4.metric("Men survived", f"{men['Survived'].mean():.0%}" if len(men) else "–")

        left, right = st.columns(2)
        with left:
            st.markdown("**Survival by sex**")
            st.altair_chart(survival_bar(df, "Sex", OVERALL), width="stretch")
        with right:
            st.markdown("**Survival by passenger class**")
            st.altair_chart(survival_bar(df, "Pclass", OVERALL), width="stretch")
        st.caption(f"Dashed line = overall survival rate across all 891 passengers ({OVERALL:.1%}). "
                   "Hover any bar for exact counts.")

    st.subheader("🔑 Key findings")
    st.markdown("""
1. **Only 38.4% survived.** 342 of 891 passengers in the training set.
2. **Sex was the strongest factor.** 74% of women survived against 19% of men.
3. **Class created a steep gradient.** 63% in 1st, 47% in 2nd and 24% in 3rd class.
4. **1st-class women survived at ~97%**, while 3rd-class men survived at just 14%.
5. **"Children first" was real but unequal.** Every boy in 1st and 2nd class survived, compared with 39% in 3rd.
6. **Title is the best single feature.** Mrs 79%, Miss 70%, Master 57%, Mr 16%.
7. **Money bought survival.** Survivors' median fare was £26 against £10.50 for those who died.
8. **Family size is U-shaped.** Small families survived at 58%, solo travellers at 30%, and large families at 16%.
9. **Cherbourg's 55% is a class illusion.** Half of its passengers travelled 1st class.
10. **Average age hides the age effect.** Both groups had a median age of 28, yet children were far more likely to survive.
""")

# ── Explore ──────────────────────────────────────────────────────────────────

with tab_explore:
    if df.empty:
        st.warning("No passengers match the current filters.")
    else:
        feature = st.selectbox("Feature", EXPLORE_FEATURES, format_func=lambda c: LABELS.get(c, c))
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Survival rate by {LABELS.get(feature, feature).lower()}**")
            st.altair_chart(survival_bar(df, feature, OVERALL), width="stretch")
        with right:
            st.markdown(f"**Passengers by {LABELS.get(feature, feature).lower()} and outcome**")
            st.altair_chart(outcome_counts(df, feature), width="stretch")

        st.divider()
        left, right = st.columns(2)
        with left:
            st.markdown("**Age distribution by outcome** (recorded ages only)")
            st.altair_chart(age_histogram(df), width="stretch")
        with right:
            st.markdown("**Age vs fare** (scroll to zoom, drag to pan, hover for names)")
            st.altair_chart(age_fare_scatter(df), width="stretch")

# ── Sex × feature ────────────────────────────────────────────────────────────

with tab_combine:
    if df.empty:
        st.warning("No passengers match the current filters.")
    else:
        combo = st.radio("Columns", ["Pclass", "AgeGroup", "FareBand", "FamilyGroup", "Embarked"],
                         horizontal=True, format_func=lambda c: LABELS.get(c, c))
        st.markdown(f"**Survival rate: sex × {LABELS.get(combo, combo).lower()}**")
        st.altair_chart(heatmap(df, combo), width="stretch")
        st.caption("Sex and class together tell most of the story: a 3rd-class woman (50%) still "
                   "had better odds than a 1st-class man (37%).")

# ── Statistics ───────────────────────────────────────────────────────────────

with tab_stats:
    chi, num = run_stats(data)
    st.markdown("""
**Is each pattern real, or just chance?**

- **Chi-square test:** checks whether survival depends on a categorical feature.
- **Cramér's V:** measures how *strong* that link is (0 = none, 1 = perfect).
- **Mann-Whitney U:** compares Age and Fare between survivors and non-survivors using ranks, so it is robust to skew.
""")
    st.markdown("##### Categorical features vs survival")
    st.dataframe(
        chi[["feature", "chi2", "dof", "p_value", "cramers_v", "min_expected"]],
        hide_index=True, width="stretch",
        column_config={
            "p_value": st.column_config.NumberColumn("p-value", format="%.2e"),
            "cramers_v": st.column_config.ProgressColumn("Cramér's V", min_value=0, max_value=1,
                                                         format="%.2f"),
            "min_expected": st.column_config.NumberColumn(
                "Min expected count", help="Below 5 means the chi-square p-value is approximate."),
        },
    )
    for line in chi["plain_english"]:
        st.markdown(f"- {line}")
    st.markdown("##### Age & Fare: survivors vs non-survivors")
    st.dataframe(
        num[["feature", "n", "median_survived", "median_died", "t_p_value", "mw_p_value",
             "rank_biserial_r"]],
        hide_index=True, width="stretch",
        column_config={"t_p_value": st.column_config.NumberColumn("Welch t p-value", format="%.2e"),
                       "mw_p_value": st.column_config.NumberColumn("Mann-Whitney p-value", format="%.2e")},
    )
    for line in num["plain_english"]:
        st.markdown(f"- {line}")
    st.info("**Why is age not significant overall, yet age *groups* are?** Age matters only at the "
            "extremes: children survived more and seniors less. Comparing averages hides that.")

# ── Predict ──────────────────────────────────────────────────────────────────

TITLES_BY_SEX = {"male": ["Mr", "Master", "Rare"], "female": ["Mrs", "Miss", "Rare"]}

with tab_predict:
    st.markdown("**Would you have survived?** Describe a passenger, and a Random Forest trained "
                "on the engineered features estimates their chance of survival.")
    model = fit_model(data)
    with st.form("passenger"):
        c1, c2, c3 = st.columns(3)
        with c1:
            p_sex = st.radio("Sex", ["female", "male"], horizontal=True,
                             format_func=lambda v: label("Sex", v))
            p_class = st.radio("Passenger class", [1, 2, 3], index=2, horizontal=True,
                               format_func=lambda v: label("Pclass", v))
            p_age = st.slider("Age", 0, 80, 28)
        with c2:
            p_sibsp = st.number_input("Siblings / spouses aboard", 0, 8, 0)
            p_parch = st.number_input("Parents / children aboard", 0, 6, 0)
            p_fare = st.number_input("Fare paid (£)", 0.0, 520.0,
                                     float({1: 60.29, 2: 14.25, 3: 8.05}[p_class]), step=1.0,
                                     help="Defaults to the median fare for the chosen class.")
        with c3:
            p_port = st.selectbox("Port of embarkation", ["S", "C", "Q"],
                                  format_func=lambda v: label("Embarked", v))
            p_title = st.selectbox("Title", ["Auto"] + TITLE_ORDER,
                                   help="Auto picks Master for boys under 13, Mr for men, "
                                        "Miss for unaccompanied women under 30, otherwise Mrs.")
            p_cabin = st.checkbox("Cabin recorded", value=(p_class == 1))
        submitted = st.form_submit_button("Estimate survival", type="primary")

    if p_title == "Auto":
        if p_sex == "male":
            p_title = "Master" if p_age < 13 else "Mr"
        else:
            p_title = "Miss" if (p_age < 30 and p_sibsp == 0) else "Mrs"

    passenger = set_feature_dtypes(pd.DataFrame([{
        "Pclass": p_class, "Sex": p_sex, "Age": float(p_age), "SibSp": p_sibsp, "Parch": p_parch,
        "Fare": p_fare, "LogFare": np.log1p(p_fare), "Embarked": p_port, "Title": p_title,
        "HasCabin": int(p_cabin), "AgeImputed": 0, "FamilySize": p_sibsp + p_parch + 1,
        "IsAlone": int(p_sibsp + p_parch == 0),
    }]))
    prob = float(model.predict_proba(encode_features(passenger)[ENGINEERED_FEATURES])[0, 1])
    similar = data[(data["Sex"] == p_sex) & (data["Pclass"] == p_class)]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Estimated survival chance", f"{prob:.0%}")
        st.progress(prob)
        verdict = "✅ Likely survived" if prob >= 0.5 else "❌ Likely did not survive"
        st.markdown(f"### {verdict}")
    with c2:
        st.markdown(f"**For context:** {len(similar)} real {label('Pclass', p_class)}-class "
                    f"{'women' if p_sex == 'female' else 'men'} were in the training data, and "
                    f"**{similar['Survived'].mean():.0%}** of them survived.")
        if p_title != "Rare" and p_title not in TITLES_BY_SEX[p_sex]:
            st.warning(f"'{p_title}' is unusual for a {label('Sex', p_sex).lower()} passenger; "
                       "the estimate may be unreliable.")
        st.caption("This is a quick baseline (≈83% cross-validated accuracy), not a tuned model.")

    st.divider()
    results, importance, rule_acc = run_model_eval(data)
    left, right = st.columns(2)
    with left:
        st.markdown("##### Cross-validated accuracy (5-fold)")
        table = results.pivot(index="model", columns="features", values="cv_accuracy")[["raw", "engineered"]]
        table.loc["'All women survive' rule"] = [rule_acc, np.nan]
        st.dataframe(table.style.format("{:.1%}", na_rep="–"), width="stretch")
        st.caption("Engineered features lift Logistic Regression by ~2.4 points.")
    with right:
        st.markdown("##### Random Forest feature importance")
        imp = importance.reset_index(names="feature")
        st.altair_chart(
            alt.Chart(imp).mark_bar(color=SURVIVED_COLOR, cornerRadiusTopRight=4,
                                    cornerRadiusBottomRight=4).encode(
                x=alt.X("rf_importance:Q", title="Importance", axis=alt.Axis(format="%")),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature:N", alt.Tooltip("rf_importance:Q", format=".1%", title="Importance")],
            ).properties(height=380),
            width="stretch",
        )

# ── Data & cleaning ──────────────────────────────────────────────────────────

with tab_data:
    log = load_cleaning_log()
    if log is not None:
        st.markdown("##### Cleaning decisions")
        st.dataframe(log[["column", "action", "rationale", "values_affected"]], hide_index=True,
                     width="stretch")
    st.markdown(f"##### Cleaned & featured data ({len(df)} rows after filters)")
    st.dataframe(df, hide_index=True, width="stretch", height=380)
    st.download_button("⬇️ Download filtered data (CSV)", df.to_csv(index=False).encode(),
                       file_name="lifeboat_ledger_filtered.csv", mime="text/csv")
