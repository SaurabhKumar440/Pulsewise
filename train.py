#!/usr/bin/env python3
"""
train.py - Early Diabetes Prediction for BOTH sexes (adult men and women)

Run:  python train.py --data data/diabetes_prediction_dataset.csv

For each feature set ("full" = with HbA1c / glucose, "screening" = without labs):
1. Stratified 80/20 split (random_state=42; identical rows for both feature sets)
2. 5-fold stratified CV benchmark of Logistic Regression / Random Forest / XGBoost.
   Every fold re-fits the WHOLE pipeline (encode -> impute -> clip -> engineer ->
   StandardScaler -> model), so nothing leaks from validation folds.
3. Best model chosen by mean CV ROC-AUC
4. Decision threshold tuned on out-of-fold training predictions so Recall >= 85%
   (the test set is never used to pick the threshold)
5. Held-out test evaluation at the default (0.50) and the tuned threshold, plus a
   sensitivity breakdown for women and men
6. SHAP explanations
7. Artifacts saved with joblib / JSON under models/<feature_set>/
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend: figures are written to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier  # swap for lightgbm.LGBMClassifier if you prefer

from preprocessing import (
    ALL_RAW_COLUMNS,
    FEATURE_GROUPS,
    FEATURE_SETS,
    TARGET,
    ClinicalPreprocessor,
    load_data,
    model_features,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5
DEFAULT_THRESHOLD = 0.50
CLASS_NAMES = ["Non-diabetic", "Diabetic"]
SHAP_SAMPLE = 2000  # rows used for SHAP (keeps tree explainers fast)

MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",  # blue
    "Random Forest": "#eb6834",  # orange
    "XGBoost": "#1baf7a",  # aqua
}
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#0d366b"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "font.size": 10, "lines.linewidth": 2,
    }
)

SET_TITLES = {"full": "with HbA1c + glucose", "screening": "screening (no lab values)"}


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# Models and pipeline
# --------------------------------------------------------------------------- #
def build_models() -> dict:
    """The three benchmark models (regularised: the label is noisy)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            # No class_weight: it would inflate probabilities, and the app shows them to
            # the user. The recall target is met through threshold tuning instead.
            n_estimators=200, max_depth=8, min_samples_leaf=30, max_features="sqrt",
            n_jobs=-1, random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            eval_metric="logloss", n_jobs=-1, random_state=SEED,
        ),
    }


def make_pipeline(model, feature_set: str) -> Pipeline:
    """Preprocess -> scale -> classify. Fitting this on a training fold learns the
    medians, IQR fences and scaler statistics from that fold only."""
    return Pipeline(
        [
            ("prep", ClinicalPreprocessor(feature_set=feature_set)),
            ("scaler", StandardScaler().set_output(transform="pandas")),
            ("clf", model),
        ]
    )


# --------------------------------------------------------------------------- #
# Metrics and threshold helpers
# --------------------------------------------------------------------------- #
def metrics_at_threshold(y_true, proba, thr: float) -> dict:
    pred = (np.asarray(proba) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_threshold(y_true, proba, min_recall: float) -> float:
    """Highest-precision threshold whose recall is >= min_recall (ties -> higher
    threshold). If no threshold reaches the target, the best-recall one is returned."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    eligible = np.where(recall >= min_recall)[0]
    if eligible.size == 0:
        return float(thresholds[np.argmax(recall)])
    best_precision = precision[eligible].max()
    ties = eligible[np.isclose(precision[eligible], best_precision)]
    return float(thresholds[ties[-1]])


def bootstrap_ci(y_true, proba, thr, n_boot=1000, seed=SEED):
    """95% bootstrap CI for recall and precision at a threshold."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    recalls, precisions = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, pred = y_true[idx], proba[idx] >= thr
        tp, fn = np.sum(pred & (yt == 1)), np.sum(~pred & (yt == 1))
        fp = np.sum(pred & (yt == 0))
        if tp + fn == 0:
            continue
        recalls.append(tp / (tp + fn))
        precisions.append(tp / (tp + fp) if tp + fp else np.nan)
    return np.nanpercentile(recalls, [2.5, 97.5]), np.nanpercentile(precisions, [2.5, 97.5])


def print_threshold_report(title, y_true, proba, thr) -> dict:
    m = metrics_at_threshold(y_true, proba, thr)
    pred = (np.asarray(proba) >= thr).astype(int)
    print(f"\n--- {title} (threshold = {thr:.3f}) ---")
    print("Confusion matrix  [rows = actual, cols = predicted]")
    print(pd.DataFrame(
        [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
        index=[f"Actual {c}" for c in CLASS_NAMES],
        columns=[f"Pred {c}" for c in CLASS_NAMES],
    ).to_string())
    print(classification_report(y_true, pred, target_names=CLASS_NAMES, digits=3))
    return m


# --------------------------------------------------------------------------- #
# Cross-validated benchmark (manual loop: one fit per fold gives BOTH the fold
# scores and the out-of-fold probabilities used for threshold tuning)
# --------------------------------------------------------------------------- #
def cv_benchmark(models: dict, X_train: pd.DataFrame, y_train: np.ndarray, feature_set: str):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows, oof_probas = [], {}
    for name, model in models.items():
        oof = np.zeros(len(y_train))
        folds = []
        for tr, va in cv.split(X_train, y_train):
            pipe = make_pipeline(clone(model), feature_set).fit(X_train.iloc[tr], y_train[tr])
            p = pipe.predict_proba(X_train.iloc[va])[:, 1]
            oof[va] = p
            m = metrics_at_threshold(y_train[va], p, DEFAULT_THRESHOLD)
            folds.append({
                "roc_auc": roc_auc_score(y_train[va], p),
                "pr_auc": average_precision_score(y_train[va], p),
                "recall": m["recall"], "precision": m["precision"],
                "f1": m["f1"], "accuracy": m["accuracy"],
            })
        fold_df = pd.DataFrame(folds)
        row = {"model": name}
        for key in fold_df.columns:
            row[f"{key}_mean"], row[f"{key}_std"] = fold_df[key].mean(), fold_df[key].std()
        rows.append(row)
        oof_probas[name] = oof
        print(f"  {name:<20} ROC-AUC {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}")
    return pd.DataFrame(rows), oof_probas


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_model_comparison(cv_df: pd.DataFrame, path: Path, title_suffix: str) -> None:
    metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("recall", "Recall @ 0.50"),
               ("precision", "Precision @ 0.50")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width, x = 0.2, np.arange(len(metrics))
    for i, name in enumerate(cv_df["model"]):
        row = cv_df.iloc[i]
        ax.bar(x + (i - 1) * (width + 0.02), [row[f"{k}_mean"] for k, _ in metrics], width,
               yerr=[row[f"{k}_std"] for k, _ in metrics], label=name, color=MODEL_COLORS[name],
               error_kw={"ecolor": INK_2, "lw": 1, "capsize": 3})
    ax.set_xticks(x, [lbl for _, lbl in metrics])
    ax.set(ylim=(0, 1.0), ylabel="5-fold CV score (mean ± SD)",
           title=f"Model benchmark - {title_suffix}")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc_pr(y_test, test_probas: dict, best: str, thr_opt: float, path: Path,
                recall_target: float, title_suffix: str) -> None:
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for name, proba in test_probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax_roc.plot(fpr, tpr, color=MODEL_COLORS[name],
                    label=f"{name} (AUC {roc_auc_score(y_test, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax_pr.plot(rec, prec, color=MODEL_COLORS[name],
                   label=f"{name} (AP {average_precision_score(y_test, proba):.3f})")
    ax_roc.plot([0, 1], [0, 1], ls="--", color=INK_2, lw=1, label="Chance")
    ax_roc.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (recall)",
               title=f"ROC curve - test set, {title_suffix}")
    ax_roc.legend(loc="lower right")
    for thr, label, marker in [(DEFAULT_THRESHOLD, "Default 0.50", "o"),
                               (thr_opt, f"Tuned {thr_opt:.2f}", "D")]:
        m = metrics_at_threshold(y_test, test_probas[best], thr)
        ax_pr.plot(m["recall"], m["precision"], marker, ms=9, mfc=INK, mec=SURFACE, mew=2,
                   label=f"{best}: {label}", zorder=5)
    ax_pr.axvline(recall_target, color=INK_2, ls=":", lw=1.2)
    ax_pr.text(recall_target + 0.01, 0.05, f"recall target {recall_target:.0%}", color=INK_2, fontsize=9)
    ax_pr.axhline(np.mean(y_test), color=INK_2, ls="--", lw=1)
    ax_pr.set(xlabel="Recall (sensitivity)", ylabel="Precision", ylim=(0, 1.02),
              title=f"Precision-Recall - test set, {title_suffix}")
    ax_pr.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_threshold_tuning(y_true, oof_proba, thr_opt, recall_target, path: Path, title_suffix) -> None:
    grid = np.linspace(0.02, 0.95, 187)
    rows = [metrics_at_threshold(y_true, oof_proba, t) for t in grid]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(grid, [r["recall"] for r in rows], color="#2a78d6", label="Recall")
    ax.plot(grid, [r["precision"] for r in rows], color="#eb6834", label="Precision")
    ax.plot(grid, [r["f1"] for r in rows], color="#1baf7a", label="F1")
    ax.axhline(recall_target, color=INK_2, ls=":", lw=1.2)
    ax.axvline(DEFAULT_THRESHOLD, color=INK_2, ls="--", lw=1)
    ax.axvline(thr_opt, color=INK, lw=1.5)
    ax.text(DEFAULT_THRESHOLD + 0.01, 0.04, "default 0.50", color=INK_2, fontsize=9)
    ax.text(thr_opt + 0.01, 0.12, f"tuned {thr_opt:.2f}", color=INK, fontsize=9)
    ax.set(xlabel="Decision threshold", ylabel="Score (out-of-fold, training set)", ylim=(0, 1.02),
           title=f"Threshold tuning - {title_suffix}")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusions(y_test, proba, thr_default, thr_opt, path: Path, title_suffix) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, thr, title in [(axes[0], thr_default, f"Default threshold {thr_default:.2f}"),
                           (axes[1], thr_opt, f"Tuned threshold {thr_opt:.2f}")]:
        cm = confusion_matrix(y_test, (np.asarray(proba) >= thr).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap=SEQ_CMAP, vmin=0, vmax=cm.max())
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=17, fontweight="bold",
                    color="white" if v > cm.max() * 0.5 else INK)
        ax.set_xticks([0, 1], CLASS_NAMES)
        ax.set_yticks([0, 1], CLASS_NAMES)
        ax.set(xlabel="Predicted", ylabel="Actual", title=title)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"Test-set confusion matrices - {title_suffix}", x=0.02, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Interpretability
# --------------------------------------------------------------------------- #
def positive_class_shap(values) -> np.ndarray:
    """Normalise the different SHAP output layouts to (n_samples, n_features)."""
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


def shap_analysis(pipe: Pipeline, X_train, X_test, out_dir: Path, feature_set: str,
                  title_suffix: str) -> pd.DataFrame:
    import shap

    prep, scaler, clf = (pipe.named_steps[k] for k in ("prep", "scaler", "clf"))
    feats = model_features(feature_set)
    Xtr = scaler.transform(prep.transform(X_train.sample(min(SHAP_SAMPLE, len(X_train)), random_state=SEED)))
    Xte = scaler.transform(prep.transform(X_test.sample(min(SHAP_SAMPLE, len(X_test)), random_state=SEED)))

    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, Xtr)
    else:
        explainer = shap.TreeExplainer(clf)
    sv = positive_class_shap(explainer.shap_values(Xte))

    # Group derived features under the measurement they come from (clinical reading)
    sv_df = pd.DataFrame(sv, columns=feats)
    grouped = sv_df.T.groupby(lambda f: FEATURE_GROUPS.get(f, f)).sum().T
    importance = (grouped.abs().mean().rename("mean_abs_shap").rename_axis("factor")
                  .sort_values(ascending=False).reset_index())

    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(top["factor"], top["mean_abs_shap"], color="#2a78d6", height=0.6)
    # scikit-learn forests explain probability; logistic regression / XGBoost explain log-odds
    unit = "probability" if isinstance(clf, RandomForestClassifier) else "log-odds"
    ax.set(xlabel=f"Mean |SHAP value| (impact on {unit} of diabetes)",
           title=f"What drives high-risk predictions - {title_suffix}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_dir / "shap_importance_bar.png", dpi=160)
    plt.close(fig)

    plt.figure(figsize=(9, 6.5))
    shap.summary_plot(sv, Xte, feature_names=feats, show=False, max_display=14)
    plt.title(f"SHAP summary - {title_suffix} (red = high feature value)", loc="left")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=160)
    plt.close()

    if hasattr(clf, "feature_importances_"):
        fi = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
        print("\nBuilt-in tree feature importances (top 10):")
        print(fi.head(10).round(4).to_string())
    elif hasattr(clf, "coef_"):
        coefs = pd.Series(clf.coef_[0], index=feats).sort_values(key=np.abs, ascending=False)
        print("\nLogistic-regression coefficients on standardised features (top 10):")
        print(coefs.head(10).round(4).to_string())
    return importance


# --------------------------------------------------------------------------- #
# One full run for one feature set
# --------------------------------------------------------------------------- #
def run_feature_set(feature_set, X_train, X_test, y_train, y_test, sex_test, args, clean_log):
    suffix = SET_TITLES[feature_set]
    models_dir = Path(args.models_dir) / feature_set
    out_dir = Path(args.outputs_dir) / feature_set
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    section(f"[{feature_set.upper()}] {suffix} - preprocessing parameters (training set only)")
    probe = ClinicalPreprocessor(feature_set=feature_set).fit(X_train)
    params = pd.DataFrame({"median": probe.medians_})
    params["clip_low"], params["clip_high"] = probe.lower_, probe.upper_
    params["n_clipped"] = probe.count_clipped(X_train)
    print(params.round(2).to_string())
    print(f"Model features ({len(model_features(feature_set))}): {', '.join(model_features(feature_set))}")

    section(f"[{feature_set.upper()}] 5-fold stratified CV benchmark (training set)")
    cv_df, oof_probas = cv_benchmark(build_models(), X_train, y_train, feature_set)
    display = cv_df[["model"]].copy()
    for key in ["roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"]:
        display[key] = cv_df.apply(lambda r, k=key: f"{r[f'{k}_mean']:.3f} ± {r[f'{k}_std']:.3f}", axis=1)
    print("\n" + display.to_string(index=False))
    cv_df.round(4).to_csv(out_dir / "cv_results.csv", index=False)
    best_name = cv_df.loc[cv_df["roc_auc_mean"].idxmax(), "model"]
    print(f"\nBest model by mean CV ROC-AUC: {best_name}")
    plot_model_comparison(cv_df, out_dir / "model_comparison.png", suffix)

    section(f"[{feature_set.upper()}] Clinical threshold tuning (target recall >= {args.recall_target:.0%})")
    oof_best = oof_probas[best_name]
    thr_opt = select_threshold(y_train, oof_best, args.recall_target)
    for label, thr in [("0.50", DEFAULT_THRESHOLD), (f"{thr_opt:.3f}", thr_opt)]:
        m = metrics_at_threshold(y_train, oof_best, thr)
        print(f"Out-of-fold @ {label:<6}: recall {m['recall']:.3f}, precision {m['precision']:.3f}, "
              f"specificity {m['specificity']:.3f}")
    sweep = pd.DataFrame([metrics_at_threshold(y_train, oof_best, t)
                          for t in np.round(np.arange(0.05, 0.71, 0.05), 2)])
    sweep.round(3).to_csv(out_dir / "threshold_sweep.csv", index=False)
    print("\nThreshold sweep (out-of-fold, training set):")
    print(sweep[["threshold", "recall", "specificity", "precision", "f1"]].round(3).to_string(index=False))
    plot_threshold_tuning(y_train, oof_best, thr_opt, args.recall_target,
                          out_dir / "threshold_tuning.png", suffix)

    section(f"[{feature_set.upper()}] Held-out test set evaluation")
    fitted, test_probas = {}, {}
    for name, model in build_models().items():
        fitted[name] = make_pipeline(model, feature_set).fit(X_train, y_train)
        test_probas[name] = fitted[name].predict_proba(X_test)[:, 1]
    print("Test ROC-AUC / PR-AUC (reference only - selection used CV):")
    for name, p in test_probas.items():
        print(f"  {name:<20} ROC-AUC {roc_auc_score(y_test, p):.3f}   PR-AUC {average_precision_score(y_test, p):.3f}")

    best_pipe, best_proba = fitted[best_name], test_probas[best_name]
    print(f"\nSelected model: {best_name}")
    m_default = print_threshold_report("DEFAULT decision threshold", y_test, best_proba, DEFAULT_THRESHOLD)
    m_tuned = print_threshold_report("OPTIMAL (recall-targeted) decision threshold", y_test, best_proba, thr_opt)
    r_ci, p_ci = bootstrap_ci(y_test, best_proba, thr_opt)
    print(f"95% bootstrap CI at tuned threshold -> recall {r_ci[0]:.3f}-{r_ci[1]:.3f}, "
          f"precision {p_ci[0]:.3f}-{p_ci[1]:.3f}")

    # Does the model serve women and men equally well? (same threshold for everyone)
    by_sex = {}
    print("\nPerformance by sex at the tuned threshold:")
    for sex in ["Female", "Male"]:
        mask = (sex_test == sex).to_numpy()
        m = metrics_at_threshold(y_test[mask], best_proba[mask], thr_opt)
        m["roc_auc"] = float(roc_auc_score(y_test[mask], best_proba[mask]))
        m["n"], m["n_diabetic"] = int(mask.sum()), int(y_test[mask].sum())
        by_sex[sex] = m
        print(f"  {sex:<6} n={m['n']:>5} (diabetic {m['n_diabetic']:>4})  ROC-AUC {m['roc_auc']:.3f}  "
              f"recall {m['recall']:.3f}  precision {m['precision']:.3f}  specificity {m['specificity']:.3f}")

    plot_roc_pr(y_test, test_probas, best_name, thr_opt, out_dir / "roc_pr_curves.png",
                args.recall_target, suffix)
    plot_confusions(y_test, best_proba, DEFAULT_THRESHOLD, thr_opt,
                    out_dir / "confusion_matrices.png", suffix)

    section(f"[{feature_set.upper()}] Interpretability (SHAP)")
    importance = shap_analysis(best_pipe, X_train, X_test, out_dir, feature_set, suffix)
    importance.round(4).to_csv(out_dir / "shap_importance.csv", index=False)
    print("\nTop clinical factors by mean |SHAP| (derived features grouped with their base measurement):")
    print(importance.head(8).round(4).to_string(index=False))

    prep = best_pipe.named_steps["prep"]
    joblib.dump(best_pipe.named_steps["clf"], models_dir / "best_model.joblib")
    joblib.dump(best_pipe.named_steps["scaler"], models_dir / "scaler.joblib")
    joblib.dump(prep.get_params_dict(), models_dir / "imputation_params.joblib")
    joblib.dump(best_pipe, models_dir / "full_pipeline.joblib")  # convenience for batch scoring
    best_cv = cv_df[cv_df["model"] == best_name].iloc[0]
    metadata = {
        "feature_set": feature_set,
        "feature_set_description": suffix,
        "model_name": best_name,
        "decision_threshold": thr_opt,
        "default_threshold": DEFAULT_THRESHOLD,
        "recall_target": args.recall_target,
        "raw_columns": prep.get_params_dict()["raw_columns"],
        "model_features": model_features(feature_set),
        "cv_roc_auc_mean": float(best_cv["roc_auc_mean"]),
        "test_roc_auc": float(roc_auc_score(y_test, best_proba)),
        "test_pr_auc": float(average_precision_score(y_test, best_proba)),
        "test_metrics_default": m_default,
        "test_metrics_tuned": m_tuned,
        "test_recall_ci95": [float(r_ci[0]), float(r_ci[1])],
        "test_precision_ci95": [float(p_ci[0]), float(p_ci[1])],
        "test_metrics_by_sex_tuned": by_sex,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "positive_rate": float(np.mean(np.r_[y_train, y_test])),
        "cleaning_log": clean_log,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "random_state": SEED,
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved artifacts to {models_dir.resolve()}: "
          + ", ".join(sorted(f.name for f in models_dir.iterdir())))
    return metadata


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the both-sexes early diabetes prediction models.")
    parser.add_argument("--data", default="data/diabetes_prediction_dataset.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=FEATURE_SETS)
    parser.add_argument("--recall-target", type=float, default=0.85)
    args = parser.parse_args()

    section("1. DATA LOADING, CLEANING AND AUDIT")
    df, clean_log = load_data(args.data)
    print("Cleaning log (rows remaining after each step):")
    for k, v in clean_log.items():
        print(f"  {k:<30} {v:>8,}")
    print(f"\nRows: {len(df):,}   diabetic: {df[TARGET].sum():,} ({df[TARGET].mean():.1%})")
    by_sex = df.groupby("gender")[TARGET].agg(rows="count", diabetic="sum", rate="mean")
    print("\nBy sex:\n" + by_sex.round(3).to_string())
    bmi_placeholder = (df["bmi"].round(2) == 27.32).mean()
    print(f"\nBMI == 27.32 (upstream placeholder, treated as missing): {bmi_placeholder:.1%} of rows")

    section("2. STRATIFIED 80/20 TRAIN-TEST SPLIT")
    X, y = df[ALL_RAW_COLUMNS], df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_train):,} rows ({y_train.mean():.1%} diabetic) | "
          f"Test: {len(X_test):,} rows ({y_test.mean():.1%} diabetic)")
    print(f"Test set sex mix: {X_test['gender'].value_counts().to_dict()}")

    results = {}
    for fs in args.feature_sets:
        results[fs] = run_feature_set(fs, X_train, X_test, y_train, y_test,
                                      X_test["gender"], args, clean_log)

    section("SUMMARY")
    rows = []
    for fs, md in results.items():
        t = md["test_metrics_tuned"]
        rows.append({
            "feature set": fs, "model": md["model_name"], "threshold": round(md["decision_threshold"], 3),
            "test ROC-AUC": round(md["test_roc_auc"], 3), "recall": round(t["recall"], 3),
            "precision": round(t["precision"], 3), "specificity": round(t["specificity"], 3),
            "recall F / M": f"{md['test_metrics_by_sex_tuned']['Female']['recall']:.2f} / "
                            f"{md['test_metrics_by_sex_tuned']['Male']['recall']:.2f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nFigures: {Path(args.outputs_dir).resolve()}   Models: {Path(args.models_dir).resolve()}")
    print("Launch the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
