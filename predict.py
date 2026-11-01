import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import BMI_PLACEHOLDER, FEATURE_GROUPS, FEATURE_SETS, ClinicalPreprocessor

ROOT = Path(__file__).parent


def positive_class_shap(values):
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


def load_bundle(feature_set):
    model_dir = ROOT / "models" / feature_set
    model = joblib.load(model_dir / "best_model.joblib")
    metadata = json.loads((model_dir / "model_metadata.json").read_text())
    return {
        "model": model,
        "prep": ClinicalPreprocessor.from_params_dict(
            joblib.load(model_dir / "imputation_params.joblib")
        ),
        "scaler": joblib.load(model_dir / "scaler.joblib"),
        "meta": metadata,
    }


def main():
    request = json.load(sys.stdin)
    feature_set = request["featureSet"]
    bundle = load_bundle(feature_set)
    patient = {
        "gender": request["gender"],
        "age": float(request["age"]),
        "hypertension": int(bool(request["hypertension"])),
        "heart_disease": int(bool(request["heartDisease"])),
        "smoking_history": request["smokingHistory"],
        "bmi": BMI_PLACEHOLDER if request.get("bmiUnknown") else float(request["bmi"]),
    }
    if feature_set == "full":
        patient["HbA1c_level"] = float(request["hba1c"])
        patient["blood_glucose_level"] = float(request["bloodGlucose"])

    cleaned = bundle["prep"].transform(pd.DataFrame([patient]))
    scaled = pd.DataFrame(
        np.asarray(bundle["scaler"].transform(cleaned)), columns=cleaned.columns
    )
    probability = float(bundle["model"].predict_proba(scaled)[0, 1])
    model = bundle["model"]
    if hasattr(model, "coef_"):
        contributions = pd.Series(
            model.coef_[0] * scaled.iloc[0].to_numpy(), index=scaled.columns
        )
        unit = "log-odds"
    else:
        import shap
        contributions = pd.Series(
            positive_class_shap(shap.TreeExplainer(model).shap_values(scaled))[0],
            index=scaled.columns,
        )
        unit = "log-odds" if type(model).__name__.startswith("XGB") else "probability"
    grouped = contributions.groupby(lambda feature: FEATURE_GROUPS.get(feature, feature))
    drivers = [
        {"name": name, "value": round(float(value), 6)}
        for name, value in grouped.sum().sort_values(key=lambda values: values.abs(), ascending=False).head(6).items()
    ]
    threshold = float(bundle["meta"]["decision_threshold"])
    print(json.dumps({
        "probability": probability,
        "threshold": threshold,
        "highRisk": probability >= threshold,
        "modelName": bundle["meta"]["model_name"],
        "featureSet": feature_set,
        "drivers": drivers,
        "driverUnit": unit,
        "cleaned": {key: round(float(value), 4) for key, value in cleaned.iloc[0].items()},
        "metrics": {
            "rocAuc": bundle["meta"]["test_roc_auc"],
            "recall": bundle["meta"]["test_metrics_tuned"]["recall"],
            "precision": bundle["meta"]["test_metrics_tuned"]["precision"],
            "specificity": bundle["meta"]["test_metrics_tuned"]["specificity"],
        },
    }))


if __name__ == "__main__":
    main()
