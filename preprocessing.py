"""
preprocessing.py
================
Shared, leak-free clinical preprocessing for the both-sexes Diabetes Prediction project.

The same code is used by ``train.py`` (inside every cross-validation fold) and by
``app.py`` (at inference time), so training and serving can never drift apart.

Two feature sets are supported
------------------------------
* ``"full"``      - demographics + history + BMI + HbA1c + blood glucose.
* ``"screening"`` - the same WITHOUT the two lab values. In this dataset the label is
  almost fully determined by HbA1c / glucose, so the "full" model is closer to
  diagnosis than to early prediction. The "screening" model answers the genuinely
  early question: "who should be tested?" using only information a person has
  before any blood test.

Steps performed by :class:`ClinicalPreprocessor`
------------------------------------------------
1. Encode sex (Male = 1) and smoking history (never / former / current / unknown).
2. Treat the BMI placeholder value (27.32 - see note below) as missing data.
3. Median imputation using medians learned from the TRAINING data only.
4. IQR clipping with fences learned from the TRAINING data only (observed values).
5. Medically motivated feature engineering (see ``add_engineered_features``).

Note on the BMI placeholder: 24% of adult rows have BMI exactly 27.32, the median of
the original data. It is an upstream imputation marker, not a measurement, so it is
treated as missing here. No "was-missing" indicator is added on purpose: those rows
have a different diabetes rate (7.9% vs 11.4%) for reasons unrelated to biology, and an
indicator would teach the model a data-collection artefact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------- #
# Dataset constants
# --------------------------------------------------------------------------- #
TARGET = "diabetes"
ID_COLUMNS = ["EmployeeName", "Patient_id"]  # dropped if present (not clinical data)

RAW_BASE = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi"]
RAW_LABS = ["HbA1c_level", "blood_glucose_level"]
FEATURE_SETS = ("full", "screening")
RAW_COLUMNS = {"full": RAW_BASE + RAW_LABS, "screening": RAW_BASE}
ALL_RAW_COLUMNS = RAW_BASE + RAW_LABS

SEX_VALUES = ["Female", "Male"]  # 'Other' (18 rows, all non-diabetic) is excluded
MIN_AGE = 18  # type-2 diabetes prediction is meaningless for infants / children

# Raw smoking labels -> 4 clinically coherent groups
SMOKING_MAP = {
    "never": "never",
    "former": "former",
    "not current": "former",
    "ever": "former",
    "current": "current",
    "No Info": "unknown",
}
SMOKING_CATEGORIES = ["never", "former", "current", "unknown"]

BMI_PLACEHOLDER = 27.32  # upstream median-imputation marker -> treated as missing

# Tukey's "far-out" fence (3 x IQR): trims only extreme / implausible values and keeps
# the clinically meaningful tails (e.g. severe obesity, very high HbA1c).
IQR_FACTOR = 3.0
CLIP_COLUMNS = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]

# Bin edges (np.digitize: a value equal to an edge falls in the UPPER bin)
AGE_BIN_EDGES = [35, 45, 55, 65]  # <35, 35-44, 45-54, 55-64, 65+   (screening starts at 35)
BMI_BIN_EDGES = [18.5, 25.0, 30.0]  # underweight, healthy, overweight, obese (WHO)
HBA1C_BIN_EDGES = [5.7, 6.5]  # normal, prediabetes, diabetes range (ADA)
GLUCOSE_BIN_EDGES = [100.0, 140.0, 200.0]  # <100, 100-139, 140-199, >=200 mg/dL

# Features that are summarised under one clinical heading in explanations.
FEATURE_GROUPS = {
    "age": "Age", "Age_Group": "Age", "Age_BMI_Risk": "Age",
    "bmi": "BMI", "BMI_Category": "BMI",
    "is_male": "Sex",
    "hypertension": "Hypertension", "heart_disease": "Heart disease",
    "Cardiometabolic_Burden": "Hypertension / heart disease",
    "smoking_never": "Smoking", "smoking_former": "Smoking",
    "smoking_current": "Smoking", "smoking_unknown": "Smoking",
    "HbA1c_level": "HbA1c", "HbA1c_Category": "HbA1c", "Glucose_eAG_Gap": "HbA1c",
    "blood_glucose_level": "Blood glucose", "Glucose_Category": "Blood glucose",
}


def model_features(feature_set: str) -> list[str]:
    """Ordered list of columns the model sees for a feature set."""
    feats = ["is_male", "age", "hypertension", "heart_disease",
             *[f"smoking_{c}" for c in SMOKING_CATEGORIES], "bmi",
             "Age_Group", "BMI_Category", "Age_BMI_Risk", "Cardiometabolic_Burden"]
    if feature_set == "full":
        feats += ["HbA1c_level", "blood_glucose_level",
                  "HbA1c_Category", "Glucose_Category", "Glucose_eAG_Gap"]
    return feats


# --------------------------------------------------------------------------- #
# Loading and dataset-level cleaning
# --------------------------------------------------------------------------- #
def load_data(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load the CSV and apply dataset-level cleaning. Returns (frame, cleaning_log).

    * drops any name / patient-id columns (not clinical features)
    * keeps Male / Female rows only (the 18 'Other' rows are all non-diabetic and far
      too few to learn from)
    * keeps adults (age >= 18)
    * removes exact duplicate rows BEFORE the train/test split, so a patient cannot
      appear in both sets
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Place diabetes_prediction_dataset.csv there."
        )
    df = pd.read_csv(path).drop(columns=ID_COLUMNS, errors="ignore")
    missing = [c for c in ALL_RAW_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    log = {"rows_raw": len(df)}
    df = df[df["gender"].isin(SEX_VALUES)]
    log["after_removing_other_gender"] = len(df)
    df = df[df["age"] >= MIN_AGE]
    log["after_keeping_adults"] = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log["after_removing_duplicates"] = len(df)
    return df, log


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def add_engineered_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Add clinically motivated features to an already cleaned frame.

    * ``Age_Group``              ordinal age bucket (<35, 35-44, 45-54, 55-64, 65+)
    * ``BMI_Category``           WHO class: 0 underweight, 1 healthy, 2 overweight, 3 obese
    * ``Age_BMI_Risk``           age x BMI / 1000 - risk compounds when both are high
    * ``Cardiometabolic_Burden`` hypertension + heart disease (0-2)
    * ``HbA1c_Category``         0 normal (<5.7), 1 prediabetes (5.7-6.4), 2 diabetes range (>=6.5)
    * ``Glucose_Category``       0: <100, 1: 100-139, 2: 140-199, 3: >=200 mg/dL
    * ``Glucose_eAG_Gap``        blood glucose minus the glucose predicted from HbA1c
      (eAG = 28.7 x HbA1c - 46.7, ADAG formula). Large gaps flag a mismatch between
      long-term and current glycaemia.
    """
    out = df.copy()
    out["Age_Group"] = np.digitize(out["age"].to_numpy(), AGE_BIN_EDGES).astype(float)
    out["BMI_Category"] = np.digitize(out["bmi"].to_numpy(), BMI_BIN_EDGES).astype(float)
    out["Age_BMI_Risk"] = out["age"] * out["bmi"] / 1000.0
    out["Cardiometabolic_Burden"] = out["hypertension"] + out["heart_disease"]
    if feature_set == "full":
        out["HbA1c_Category"] = np.digitize(out["HbA1c_level"].to_numpy(), HBA1C_BIN_EDGES).astype(float)
        out["Glucose_Category"] = np.digitize(out["blood_glucose_level"].to_numpy(), GLUCOSE_BIN_EDGES).astype(float)
        out["Glucose_eAG_Gap"] = out["blood_glucose_level"] - (28.7 * out["HbA1c_level"] - 46.7)
    return out


class ClinicalPreprocessor(BaseEstimator, TransformerMixin):
    """Encoding, placeholder->NaN, median imputation, IQR clipping, feature engineering.

    Everything that depends on data (medians, clip fences) is learned in ``fit`` so
    that, inside cross-validation or a train/test split, nothing leaks from the
    validation / test rows.
    """

    def __init__(self, feature_set: str = "full", iqr_factor: float = IQR_FACTOR):
        self.feature_set = feature_set
        self.iqr_factor = iqr_factor

    # -- helpers ------------------------------------------------------------ #
    def _prepare(self, X) -> pd.DataFrame:
        """Select the raw columns for this feature set and encode them numerically."""
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        cols = RAW_COLUMNS[self.feature_set]
        raw = pd.DataFrame(X).loc[:, cols].copy()

        sex = raw["gender"].map({"Female": 0.0, "Male": 1.0})
        if sex.isna().any():
            bad = sorted(raw.loc[sex.isna(), "gender"].astype(str).unique())
            raise ValueError(f"Unsupported gender value(s): {bad}. Use 'Female' or 'Male'.")

        df = pd.DataFrame({"is_male": sex})
        for c in ["age", "hypertension", "heart_disease", "bmi"]:
            df[c] = pd.to_numeric(raw[c], errors="coerce").astype(float)
        df.loc[df["bmi"].round(2) == BMI_PLACEHOLDER, "bmi"] = np.nan  # upstream placeholder
        for c in RAW_LABS:
            if c in raw.columns:
                df[c] = pd.to_numeric(raw[c], errors="coerce").astype(float)
        # Unrecognised smoking labels are treated as "unknown" rather than failing.
        df["smoking_history"] = raw["smoking_history"].map(SMOKING_MAP).fillna("unknown")
        return df

    # -- sklearn API -------------------------------------------------------- #
    def fit(self, X, y=None):
        df = self._prepare(X)
        numeric = [c for c in df.columns if c != "smoking_history"]
        self.medians_ = df[numeric].median()
        clip_cols = [c for c in CLIP_COLUMNS if c in df.columns]
        q1, q3 = df[clip_cols].quantile(0.25), df[clip_cols].quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.iqr_factor * iqr
        self.upper_ = q3 + self.iqr_factor * iqr
        self.model_features_ = model_features(self.feature_set)
        return self

    def transform(self, X) -> pd.DataFrame:
        df = self._prepare(X)
        smoking = df.pop("smoking_history")
        df = df.fillna(self.medians_)
        for c in self.lower_.index:
            df[c] = df[c].clip(lower=self.lower_[c], upper=self.upper_[c])
        for cat in SMOKING_CATEGORIES:
            df[f"smoking_{cat}"] = (smoking == cat).astype(float)
        return add_engineered_features(df, self.feature_set)[self.model_features_]

    def get_feature_names_out(self, input_features=None):
        return np.array(model_features(self.feature_set), dtype=object)

    def count_clipped(self, X) -> pd.Series:
        """Number of values per clipped column outside the learned IQR fences."""
        df = self._prepare(X).drop(columns="smoking_history").fillna(self.medians_)
        return pd.Series({c: int(((df[c] < self.lower_[c]) | (df[c] > self.upper_[c])).sum())
                          for c in self.lower_.index})

    # -- (de)serialisation of the learned parameters ------------------------- #
    def get_params_dict(self) -> dict:
        """Plain-python dict of everything learned in ``fit`` (safe to joblib/JSON)."""
        return {
            "feature_set": self.feature_set,
            "iqr_factor": float(self.iqr_factor),
            "raw_columns": list(RAW_COLUMNS[self.feature_set]),
            "model_features": list(self.model_features_),
            "medians": {k: float(v) for k, v in self.medians_.items()},
            "clip_lower": {k: float(v) for k, v in self.lower_.items()},
            "clip_upper": {k: float(v) for k, v in self.upper_.items()},
            "bmi_placeholder": BMI_PLACEHOLDER,
            "smoking_map": dict(SMOKING_MAP),
        }

    @classmethod
    def from_params_dict(cls, params: dict) -> "ClinicalPreprocessor":
        """Rebuild a fitted preprocessor from ``get_params_dict`` output."""
        obj = cls(feature_set=params["feature_set"], iqr_factor=params["iqr_factor"])
        obj.medians_ = pd.Series(params["medians"])
        obj.lower_ = pd.Series(params["clip_lower"])
        obj.upper_ = pd.Series(params["clip_upper"])
        obj.model_features_ = list(params["model_features"])
        return obj
