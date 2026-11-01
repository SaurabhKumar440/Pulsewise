"""
app.py - Early Diabetes Risk Screening for adult women and men (Streamlit)

Run:  streamlit run app.py        (after `python train.py` has created ./models)

Two models are served:
  * screening - no blood test needed (sex, age, BMI, hypertension, heart disease, smoking)
  * full      - adds HbA1c and blood glucose; used when the person has lab results

Each uses the exact preprocessing code and parameters learned in training
(imputation_params.joblib -> ClinicalPreprocessor -> scaler.joblib -> best_model.joblib)
and flags "High Risk" at its own recall-tuned threshold (model_metadata.json).

EDUCATIONAL TOOL ONLY - not a diagnostic device.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from preprocessing import (
    BMI_PLACEHOLDER,
    FEATURE_GROUPS,
    FEATURE_SETS,
    ClinicalPreprocessor,
)

MODELS_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="Early Diabetes Risk Screening", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def positive_class_shap(values) -> np.ndarray:
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    """Load model, scaler, imputation parameters, metadata and SHAP explainer per feature set."""
    needed = ["best_model.joblib", "scaler.joblib", "imputation_params.joblib", "model_metadata.json"]
    bundles = {}
    for fs in FEATURE_SETS:
        d = MODELS_DIR / fs
        if not all((d / f).exists() for f in needed):
            return None
        model = joblib.load(d / "best_model.joblib")
        explainer = None
        if not hasattr(model, "coef_"):  # tree model -> SHAP explainer (built once)
            import shap

            explainer = shap.TreeExplainer(model)
        bundles[fs] = {
            "model": model,
            "scaler": joblib.load(d / "scaler.joblib"),
            "prep": ClinicalPreprocessor.from_params_dict(joblib.load(d / "imputation_params.joblib")),
            "meta": json.loads((d / "model_metadata.json").read_text()),
            "explainer": explainer,
        }
    return bundles


bundles = load_artifacts()
if bundles is None:
    st.error("Model files not found in `./models`. Run `python train.py` first, then reload this page.")
    st.stop()


# --------------------------------------------------------------------------- #
# Clinical reference ranges (shown under each input)
# --------------------------------------------------------------------------- #
INF = float("inf")
RANGES = {
    "age": [(35, 0, "Lower age-related risk (< 35)"), (45, 1, "Rising risk (35-44)"),
            (INF, 2, "Higher risk (45+)")],
    "bmi": [(18.5, 1, "Underweight (< 18.5)"), (25.0, 0, "Healthy (18.5-24.9)"),
            (30.0, 1, "Overweight (25-29.9)"), (INF, 2, "Obese (30+)")],
    "HbA1c_level": [(5.7, 0, "Normal (< 5.7%)"), (6.5, 1, "Prediabetes range (5.7-6.4%)"),
                    (INF, 2, "Diabetes range (6.5%+)")],
    "blood_glucose_level": [(100, 0, "Normal if fasting (< 100)"), (140, 1, "Raised (100-139)"),
                            (200, 1, "High (140-199)"), (INF, 2, "Very high (200+)")],
}
LEVEL_ICON = {0: "🟢", 1: "🟡", 2: "🔴"}


def classify(feature: str, value: float) -> tuple[int, str]:
    for upper, level, label in RANGES[feature]:
        if value < upper:
            return level, label
    return 2, RANGES[feature][-1][2]


# --------------------------------------------------------------------------- #
# Prediction + explanation
# --------------------------------------------------------------------------- #
def predict(bundle: dict, patient: dict):
    """Raw patient dict -> (probability, cleaned features, scaled features)."""
    raw = pd.DataFrame([patient])
    cleaned = bundle["prep"].transform(raw)
    scaled = pd.DataFrame(np.asarray(bundle["scaler"].transform(cleaned)), columns=cleaned.columns)
    return float(bundle["model"].predict_proba(scaled)[0, 1]), cleaned, scaled


def explain(bundle: dict, scaled: pd.DataFrame) -> tuple[pd.Series, str]:
    """Per-factor contribution relative to an average patient, grouped clinically."""
    model = bundle["model"]
    if hasattr(model, "coef_"):
        contrib = pd.Series(model.coef_[0] * scaled.iloc[0].to_numpy(), index=scaled.columns)
        unit = "log-odds"
    else:
        contrib = pd.Series(positive_class_shap(bundle["explainer"].shap_values(scaled))[0],
                            index=scaled.columns)
        # tree ensembles from scikit-learn explain probability; XGBoost explains log-odds
        unit = "log-odds" if type(model).__name__.startswith("XGB") else "probability"
    return contrib.groupby(lambda f: FEATURE_GROUPS.get(f, f)).sum(), unit


# --------------------------------------------------------------------------- #
# Visual components
# --------------------------------------------------------------------------- #
def risk_badge(high: bool) -> None:
    if high:
        style = "background:#fde8e8;color:#7a1414;border:2px solid #e34948;"
        text, icon = "High Risk of Diabetes", "⚠️"
    else:
        style = "background:#e3f6ec;color:#0b4f33;border:2px solid #1baf7a;"
        text, icon = "Low Risk of Diabetes", "✅"
    st.markdown(
        f"<div style='{style}border-radius:12px;padding:18px 22px;font-size:1.6rem;font-weight:700;"
        f"text-align:center;'>{icon} {text}</div>",
        unsafe_allow_html=True,
    )


def probability_gauge(prob: float, high: bool, threshold: float) -> go.Figure:
    top = max(25.0, min(100.0, float(np.ceil(max(prob * 100, threshold * 100) * 1.6 / 5) * 5)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 44}},
        title={"text": "Estimated probability of diabetes"},
        gauge={
            "axis": {"range": [0, top], "ticksuffix": "%"},
            "bar": {"color": "#c53030" if high else "#1a7f5a", "thickness": 0.28},
            "steps": [
                {"range": [0, threshold * 100], "color": "rgba(27,175,122,0.30)"},
                {"range": [threshold * 100, top], "color": "rgba(227,73,72,0.30)"},
            ],
            "threshold": {"line": {"color": "#111111", "width": 4}, "thickness": 0.85,
                          "value": threshold * 100},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def drivers_chart(contrib: pd.Series, unit: str) -> go.Figure:
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6).iloc[::-1]
    colors = ["#e34948" if v > 0 else "#2a78d6" for v in top]
    fmt = "{:+.2f}" if unit == "log-odds" else "{:+.1%}"
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h", marker_color=colors,
        text=[fmt.format(v) for v in top.values], textposition="outside",
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    pad = max(0.05 if unit != "log-odds" else 0.3, float(np.abs(top.values).max()) * 0.3)
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=40), showlegend=False,
        xaxis=dict(title=f"← lowers risk | raises risk →  ({unit}, vs. an average patient)",
                   zeroline=True, range=[min(top.values.min(), 0) - pad, max(top.values.max(), 0) + pad]),
        yaxis=dict(title=None),
    )
    return fig


# --------------------------------------------------------------------------- #
# Clinical advice
# --------------------------------------------------------------------------- #
def personalised_tips(p: dict, has_labs: bool, bmi_unknown: bool) -> list[str]:
    tips = []
    if has_labs:
        a1c, glu = p["HbA1c_level"], p["blood_glucose_level"]
        if a1c >= 6.5:
            tips.append("**HbA1c:** 6.5% or higher is in the diabetes range. A repeat or confirmatory test "
                        "ordered by a clinician is the next step.")
        elif a1c >= 5.7:
            tips.append("**HbA1c:** 5.7-6.4% is the prediabetes range. This stage is often reversible with "
                        "lifestyle change - ask about a prevention programme.")
        if glu >= 200:
            tips.append("**Blood glucose:** 200 mg/dL or higher is in the diabetes range, especially with symptoms "
                        "such as thirst or frequent urination. Seek prompt medical review.")
        elif glu >= 126:
            tips.append("**Blood glucose:** if this reading was taken while fasting, 126 mg/dL or higher is in the "
                        "diabetes range; 100-125 mg/dL is prediabetes. Confirm with a repeat test.")
    if not bmi_unknown and p["bmi"] >= 25:
        tips.append(f"**Weight:** BMI {p['bmi']:.1f} is above the healthy range. Losing 5-7% of body weight "
                    "substantially lowers the risk of developing type 2 diabetes.")
    if p["hypertension"] == 1:
        tips.append("**Blood pressure:** hypertension and diabetes often occur together - keep blood pressure controlled "
                    "and ask for a glucose test at your next review.")
    if p["heart_disease"] == 1:
        tips.append("**Heart disease:** diabetes raises cardiovascular risk further, so a glucose / HbA1c check is "
                    "worthwhile if you haven't had one recently.")
    if p["smoking_history"] == "current":
        tips.append("**Smoking:** smoking raises diabetes and heart risk. Quitting is one of the most effective changes you can make.")
    if p["age"] >= 35:
        tips.append("**Age:** adults 35 and over should be screened for diabetes at least every 3 years "
                    "(earlier and more often with overweight, high blood pressure or family history).")
    if p["gender"] == "Female":
        tips.append("**For women:** tell your doctor if you have had gestational diabetes or polycystic ovary syndrome - "
                    "both raise long-term risk.")
    if not has_labs:
        tips.append("**No blood test yet:** this estimate uses only lifestyle and history information. "
                    "A blood test (HbA1c or fasting glucose) is the only way to know.")
    if bmi_unknown:
        tips.append("**Data quality:** BMI was not provided, so a typical value was assumed and the estimate is less reliable.")
    return tips


def clinical_advice(high: bool, tips: list[str], has_labs: bool) -> None:
    st.markdown("#### Recommended next steps")
    if high:
        st.markdown(
            "1. **Arrange a blood test soon** (HbA1c, fasting plasma glucose, or an oral glucose tolerance test). "
            "Diagnostic cut-offs: HbA1c ≥ 6.5%, fasting glucose ≥ 126 mg/dL, or 2-hour glucose ≥ 200 mg/dL.\n"
            "2. **Start protective habits now:** about 150 minutes of moderate activity per week, fewer sugary drinks "
            "and refined carbohydrates, more fibre, vegetables and whole grains.\n"
            "3. **Know the warning signs** - excessive thirst, frequent urination, unexplained weight loss, blurred vision, "
            "constant tiredness, slow-healing sores - and seek care promptly if they appear."
        )
    else:
        st.markdown(
            "1. **Keep it up:** about 150 minutes of moderate activity per week and a balanced, high-fibre diet.\n"
            "2. **Stay on a screening schedule:** repeat glucose screening at least every 3 years, sooner if weight, "
            "blood pressure or family history change.\n"
            "3. **Know the warning signs** - excessive thirst, frequent urination, unexplained weight loss, blurred vision - "
            "and see a clinician if they appear, whatever this result says."
        )
    if tips:
        st.markdown("#### Based on your inputs")
        for t in tips:
            st.markdown(f"- {t}")


# --------------------------------------------------------------------------- #
# Sidebar - patient inputs
# --------------------------------------------------------------------------- #
st.sidebar.header("Patient health parameters")
st.sidebar.caption("🟢 typical · 🟡 caution · 🔴 high-risk range (shown under each input)")

patient: dict = {}
patient["gender"] = st.sidebar.radio("Sex", ["Female", "Male"], horizontal=True, key="gender",
                                     help="The model was trained on both adult women and men.")

patient["age"] = float(st.sidebar.slider("Age (years)", 18, 80, 45, 1, key="age",
                                         help="Trained on adults aged 18-80."))
st.sidebar.caption(f"{LEVEL_ICON[classify('age', patient['age'])[0]]} {classify('age', patient['age'])[1]}")

bmi_unknown = st.sidebar.checkbox("BMI: not known", key="na_bmi",
                                  help="Tick if unavailable; a typical value will be assumed.")
if bmi_unknown:
    patient["bmi"] = BMI_PLACEHOLDER  # the preprocessor treats this value as missing -> median
else:
    patient["bmi"] = float(st.sidebar.slider(
        "BMI (kg/m²)", 12.0, 60.0, 28.0, 0.1, key="bmi",
        help="Body-mass index. Healthy 18.5-24.9; overweight 25-29.9; obese 30 and above."))
    lvl, label = classify("bmi", patient["bmi"])
    st.sidebar.caption(f"{LEVEL_ICON[lvl]} {label}")

patient["hypertension"] = int(st.sidebar.checkbox("Diagnosed hypertension (high blood pressure)", key="hyp"))
patient["heart_disease"] = int(st.sidebar.checkbox("Diagnosed heart disease", key="heart"))
patient["smoking_history"] = st.sidebar.selectbox(
    "Smoking history", ["never", "former", "current", "unknown"], key="smoke",
    format_func=lambda s: {"never": "Never smoked", "former": "Former smoker",
                           "current": "Current smoker", "unknown": "Not known"}[s])

st.sidebar.divider()
has_labs = st.sidebar.toggle("I have recent blood test results", key="has_labs",
                             help="Adds HbA1c and blood glucose, which makes the estimate far more accurate.")
if has_labs:
    patient["HbA1c_level"] = float(st.sidebar.slider(
        "HbA1c (%)", 3.5, 9.0, 5.7, 0.1, key="a1c",
        help="Average blood sugar over ~3 months. Normal < 5.7%; prediabetes 5.7-6.4%; diabetes 6.5% and above."))
    lvl, label = classify("HbA1c_level", patient["HbA1c_level"])
    st.sidebar.caption(f"{LEVEL_ICON[lvl]} {label}")
    patient["blood_glucose_level"] = float(st.sidebar.slider(
        "Blood glucose (mg/dL)", 80, 300, 110, 1, key="glucose",
        help="Fasting: normal < 100, prediabetes 100-125, diabetes 126 and above. Random reading: 200 and above suggests diabetes."))
    lvl, label = classify("blood_glucose_level", patient["blood_glucose_level"])
    st.sidebar.caption(f"{LEVEL_ICON[lvl]} {label}")
else:
    patient["HbA1c_level"], patient["blood_glucose_level"] = np.nan, np.nan  # ignored by the screening model

# --------------------------------------------------------------------------- #
# Main panel
# --------------------------------------------------------------------------- #
fs = "full" if has_labs else "screening"
bundle = bundles[fs]
meta, threshold = bundle["meta"], float(bundle["meta"]["decision_threshold"])

st.title("🩺 Early Diabetes Risk Screening")
st.warning(
    "**Educational screening aid - not a diagnosis.** Trained on a public dataset of adult women and men "
    "whose origin is not documented, so it may not generalise to your population. "
    "Only a clinician's laboratory tests can diagnose diabetes."
)
st.caption(f"Model in use: **{meta['model_name']}** · "
           + ("with HbA1c + blood glucose" if has_labs else "screening model (no blood test)"))

prob, cleaned, scaled = predict(bundle, patient)
high_risk = prob >= threshold

col_badge, col_gauge = st.columns(2, gap="large")
with col_badge:
    risk_badge(high_risk)
    st.metric("Estimated probability", f"{prob:.1%}")
    tuned = meta["test_metrics_tuned"]
    st.caption(
        f"Screening threshold: **{threshold:.1%}**. At or above it you are flagged *High Risk*. The threshold is "
        f"deliberately low - tuned to catch about {meta['recall_target']:.0%} of people who have diabetes - so "
        f"about **{1 - tuned['precision']:.0%} of flagged people turn out not to have diabetes**. "
        "A High Risk flag means **'get a blood test'**, not 'you have diabetes'."
    )
with col_gauge:
    st.plotly_chart(probability_gauge(prob, high_risk, threshold), width="stretch")

st.divider()
col_advice, col_why = st.columns(2, gap="large")
with col_advice:
    clinical_advice(high_risk, personalised_tips(patient, has_labs, bmi_unknown), has_labs)
with col_why:
    st.markdown("#### Why this result? Main contributing factors")
    contributions, unit = explain(bundle, scaled)
    st.plotly_chart(drivers_chart(contributions, unit), width="stretch")
    st.caption("How much each factor raises (red) or lowers (blue) this person's risk compared with an average "
               "patient. Derived features are grouped with the measurement they come from.")

with st.expander("Values used by the model (after cleaning)"):
    st.dataframe(cleaned.iloc[0].round(3).rename("value").to_frame(), width="stretch")
    st.caption("BMI left blank (or exactly 27.32, the dataset's placeholder value) is replaced by the training median.")

with st.expander("About the model"):
    by_sex = meta["test_metrics_by_sex_tuned"]
    lo, hi = meta["test_recall_ci95"]
    st.markdown(
        f"- **Held-out test ROC-AUC:** {meta['test_roc_auc']:.3f} "
        f"({meta['n_test']:,} test patients, {meta['n_train']:,} training patients)\n"
        f"- **At the {threshold:.1%} threshold:** recall {tuned['recall']:.0%} (95% CI {lo:.0%}-{hi:.0%}), "
        f"precision {tuned['precision']:.0%}, specificity {tuned['specificity']:.0%}\n"
        f"- **Women:** recall {by_sex['Female']['recall']:.0%}, ROC-AUC {by_sex['Female']['roc_auc']:.3f} · "
        f"**Men:** recall {by_sex['Male']['recall']:.0%}, ROC-AUC {by_sex['Male']['roc_auc']:.3f}\n"
        f"- **Trained:** {meta['trained_at_utc']}"
    )
    if not has_labs:
        st.markdown(
            "The no-blood-test model is a genuine *early* screen but a modest one: it ranks patients "
            "well above chance (ROC-AUC ≈ 0.80), yet most flagged people are false alarms because age, weight "
            "and blood pressure alone cannot separate diabetic from non-diabetic people very sharply."
        )
    else:
        st.markdown(
            "With HbA1c and glucose the model is very accurate, but that is because these two tests are how "
            "diabetes is *defined* - it is closer to confirming a diagnosis than to predicting one early."
        )
