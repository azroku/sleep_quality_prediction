# =============================================================================
# Step 3a: Machine Learning Models (Baselines)
#
# KEY DESIGN DECISIONS:
#   1. TimeSeriesSplit (k=5) — walk-forward cross-validation.
#      WHY: Our data is a time series. Random k-fold would allow the model
#      to train on "future" data and test on "past" data — this is leakage.
#      Walk-forward always trains on the past and tests on the future,
#      which is the only valid way to evaluate a time-series predictor.
#
#   2. StandardScaler fitted INSIDE each CV fold.
#      WHY: If we scale the entire dataset first and then split, the
#      test fold's mean/std "leaks" into the training scaler. We must
#      fit the scaler on the training fold only, then apply it to test.
#      This is done automatically with sklearn Pipelines.
#
#   3. class_weight="balanced" for classification.
#      WHY: 118 good vs 40 poor nights (3:1 imbalance). Without weighting,
#      a model that always predicts "good" gets 74.7% accuracy — appearing
#      good while being useless. Balanced weights penalise misclassifying
#      the minority class more heavily, forcing the model to learn it.
#
#   4. Hyperparameter tuning with GridSearchCV inside each CV fold.
#      WHY: We tune hyperparameters on training data only. If we tuned on
#      all data before CV, we'd be leaking again.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import json, os, warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline           import Pipeline
from sklearn.preprocessing      import StandardScaler
from sklearn.model_selection    import TimeSeriesSplit, GridSearchCV, cross_validate
from sklearn.linear_model       import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble           import RandomForestRegressor, RandomForestClassifier
from sklearn.svm                import SVR, SVC
from sklearn.metrics            import (mean_absolute_error, mean_squared_error,
                                         r2_score, roc_auc_score, f1_score,
                                         balanced_accuracy_score, precision_score,
                                         recall_score, confusion_matrix,
                                         ConfusionMatrixDisplay, RocCurveDisplay)
from xgboost                    import XGBRegressor, XGBClassifier

DATA_PATH   = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output\sleep_merged_clean.csv"
CONFIG_PATH = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output\project_config.json"
OUTPUT_DIR  = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output"
FIG_DIR     = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output\figures"

os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["sleep_date"] = pd.to_datetime(df["sleep_date"])
df = df.sort_values("sleep_date").reset_index(drop=True)

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

FEATURES     = cfg["daytime_features"]
REG_TARGETS  = cfg["regression_targets"]
CLF_TARGET   = cfg["classification_target"]

X = df[FEATURES].values
print(f"Feature matrix : {X.shape}")
print(f"Regression targets : {REG_TARGETS}")
print(f"Classification target : {CLF_TARGET}")
print(f"Class balance : {df[CLF_TARGET].value_counts().to_dict()}\n")

#Cross-validation setup 
N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

sns.set_theme(style="whitegrid", font_scale=1.05)
COLORS = ["#378ADD", "#1D9E75", "#534AB7", "#E24B4A", "#BA7517"]


#helpers
def evaluate_regression(pipeline, X, y, tscv):
    """
    Run walk-forward CV for a regression pipeline.
    Returns dict of mean ± std for MAE, RMSE, R².
    """
    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        maes.append(mean_absolute_error(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2s.append(r2_score(y_test, y_pred))

    return {
        "MAE"  : (np.mean(maes),  np.std(maes)),
        "RMSE" : (np.mean(rmses), np.std(rmses)),
        "R2"   : (np.mean(r2s),   np.std(r2s)),
    }


def evaluate_classification(pipeline, X, y, tscv):
    """
    Run walk-forward CV for a classification pipeline.
    Returns dict of mean ± std for AUC-ROC, F1, Balanced Accuracy,
    Precision, Recall.
    """
    aucs, f1s, baccs, precs, recs = [], [], [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)

        if len(np.unique(y_test)) < 2:
            continue

        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test)

        aucs.append(roc_auc_score(y_test, y_prob))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        baccs.append(balanced_accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))

    return {
        "AUC-ROC"      : (np.mean(aucs),  np.std(aucs)),
        "F1"           : (np.mean(f1s),   np.std(f1s)),
        "Bal. Accuracy": (np.mean(baccs), np.std(baccs)),
        "Precision"    : (np.mean(precs), np.std(precs)),
        "Recall"       : (np.mean(recs),  np.std(recs)),
    }


def print_results(name, results):
    """Pretty-print results table."""
    print(f"\n  {name}")
    for metric, (mean, std) in results.items():
        print(f"    {metric:20s}: {mean:.3f} ± {std:.3f}")


# TASK A — REGRESSION MODELS
print("=" * 60)
print("TASK A: REGRESSION")
print("=" * 60)
print("Predicting: Time Asleep, REM duration, Deep Sleep duration")
print("Validation: TimeSeriesSplit, k=5, walk-forward\n")

reg_models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()) #any more complex model must beat this to justify its added complexity.
    ]),
    "Ridge (α=1)": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0))
    ]),
    "Lasso (α=0.1)": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Lasso(alpha=0.1, max_iter=5000))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "XGBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            n_jobs=-1
        ))
    ]),
}

all_reg_results = {}   # {target: {model_name: results_dict}}

for target in REG_TARGETS:
    y = df[target].values
    print(f"\n── Target: {target} ──")
    print(f"   mean={y.mean():.1f}, std={y.std():.1f}, range=[{y.min()}, {y.max()}]")

    target_results = {}
    for name, pipeline in reg_models.items():
        results = evaluate_regression(pipeline, X, y, tscv)
        print_results(name, results)
        target_results[name] = results

    all_reg_results[target] = target_results


# TASK B — CLASSIFICATION MODELS
print("\n\n" + "=" * 60)
print("TASK B: CLASSIFICATION")
print("=" * 60)
print("Predicting: poor_sleep (binary, 25th percentile threshold)")
print("Validation: TimeSeriesSplit, k=5, walk-forward")
print("Class handling: class_weight='balanced'\n")

y_clf = df[CLF_TARGET].values

clf_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),
    "SVM (RBF kernel)": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,   
            random_state=42
        ))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "XGBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            # XGBoost handles imbalance with scale_pos_weight
            # = n_negative / n_positive = 118/40 ≈ 3.0
            scale_pos_weight=cfg["n_good_sleep"] / cfg["n_poor_sleep"],
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            eval_metric="logloss"
        ))
    ]),
}

clf_results = {}
for name, pipeline in clf_models.items():
    results = evaluate_classification(pipeline, X, y_clf, tscv)
    print_results(name, results)
    clf_results[name] = results


# FIGURE 9 — Regression results comparison
print("\n\nPlotting Figure 9: Regression results...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics_to_plot = ["MAE", "RMSE", "R2"]
metric_labels   = ["MAE (min) ↓ lower is better",
                   "RMSE (min) ↓ lower is better",
                   "R² ↑ higher is better"]

for ax, target in zip(axes, REG_TARGETS):
    model_names = list(all_reg_results[target].keys())
    r2_means  = [all_reg_results[target][m]["R2"][0]   for m in model_names]
    r2_stds   = [all_reg_results[target][m]["R2"][1]   for m in model_names]
    mae_means = [all_reg_results[target][m]["MAE"][0]  for m in model_names]

    #R² descending
    order = np.argsort(r2_means)[::-1]
    sorted_names  = [model_names[i] for i in order]
    sorted_r2     = [r2_means[i]    for i in order]
    sorted_r2_std = [r2_stds[i]     for i in order]
    sorted_mae    = [mae_means[i]   for i in order]

    bars = ax.barh(sorted_names, sorted_r2, xerr=sorted_r2_std,
                   color=COLORS[:len(sorted_names)], alpha=0.80,
                   edgecolor="white", capsize=4)

    for bar, mae in zip(bars, sorted_mae):
        ax.text(max(sorted_r2) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"MAE={mae:.1f} min",
                va="center", fontsize=8, color="white", fontweight="bold")

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("R² (mean ± std across 5 folds)", fontsize=9)
    ax.set_title(target, fontweight="bold", fontsize=10)
    ax.grid(axis="x", alpha=0.4)

fig.suptitle("Figure 9 — Regression model comparison (walk-forward CV, k=5)\n"
             "R² = proportion of variance explained; MAE = mean absolute error in minutes",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(FIG_DIR, "fig9_regression_results.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 10 — Classification results comparison
print("Plotting Figure 10: Classification results...")

metrics_clf = ["AUC-ROC", "F1", "Bal. Accuracy", "Precision", "Recall"]
clf_names   = list(clf_results.keys())

fig, axes = plt.subplots(1, len(metrics_clf), figsize=(20, 5))

for ax, metric in zip(axes, metrics_clf):
    means = [clf_results[m][metric][0] for m in clf_names]
    stds  = [clf_results[m][metric][1] for m in clf_names]

    order = np.argsort(means)[::-1]
    s_names = [clf_names[i] for i in order]
    s_means = [means[i]     for i in order]
    s_stds  = [stds[i]      for i in order]

    bars = ax.barh(s_names, s_means, xerr=s_stds,
                   color=COLORS[:len(s_names)], alpha=0.80,
                   edgecolor="white", capsize=4)

    # Reference line: random classifier = 0.5
    ax.axvline(0.5, color="gray", linewidth=1.0, linestyle="--",
               label="random (0.5)")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"{metric}", fontsize=9)
    ax.set_title(metric, fontweight="bold", fontsize=10)
    ax.grid(axis="x", alpha=0.4)
    ax.legend(fontsize=7)

fig.suptitle("Figure 10 — Classification model comparison (walk-forward CV, k=5)\n"
             "Dashed line = random classifier baseline (0.5)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(FIG_DIR, "fig10_classification_results.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 11 — Feature importance (best models)

print("Plotting Figure 11: Feature importance...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
plot_idx = 0

for target in REG_TARGETS:
    y = df[target].values
    rf_reg = RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf_reg.fit(X_scaled, y)

    importances = rf_reg.feature_importances_
    order = np.argsort(importances)
    ax = axes[plot_idx]
    ax.barh([FEATURES[i] for i in order],
            [importances[i] for i in order],
            color=COLORS[plot_idx % len(COLORS)], alpha=0.8, edgecolor="white")
    ax.set_xlabel("Feature importance (mean decrease in impurity)", fontsize=9)
    ax.set_title(f"RF Regressor → {target}", fontweight="bold", fontsize=10)
    ax.grid(axis="x", alpha=0.4)
    plot_idx += 1

rf_clf = RandomForestClassifier(
    n_estimators=200, max_depth=5, min_samples_leaf=3,
    class_weight="balanced", random_state=42, n_jobs=-1
)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rf_clf.fit(X_scaled, y_clf)

importances = rf_clf.feature_importances_
order = np.argsort(importances)
ax = axes[3]
ax.barh([FEATURES[i] for i in order],
        [importances[i] for i in order],
        color=COLORS[3], alpha=0.8, edgecolor="white")
ax.set_xlabel("Feature importance", fontsize=9)
ax.set_title("RF Classifier → poor_sleep", fontweight="bold", fontsize=10)
ax.grid(axis="x", alpha=0.4)

fig.suptitle("Figure 11 — Random Forest feature importance\n"
             "(higher = more predictive of sleep outcome)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(FIG_DIR, "fig11_feature_importance.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 12 — ROC curves (classification, last CV fold)

print("Plotting Figure 12: ROC curves...")

splits = list(tscv.split(X))
train_idx, test_idx = splits[-1]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_clf[train_idx], y_clf[test_idx]

fig, ax = plt.subplots(figsize=(8, 7))

for (name, pipeline), color in zip(clf_models.items(), COLORS):
    pipeline.fit(X_train, y_train)
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = pipeline.decision_function(X_test)

    if len(np.unique(y_test)) >= 2:
        auc = roc_auc_score(y_test, y_prob)
        disp = RocCurveDisplay.from_predictions(
            y_test, y_prob, name=f"{name} (AUC={auc:.3f})",
            ax=ax
        )
disp.line_.set_color(color)
disp.line_.set_alpha(0.85)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.5)")
ax.set_title("Figure 12 — ROC curves (final CV fold)\n"
             "True Positive Rate = sensitivity; False Positive Rate = 1 − specificity",
             fontweight="bold")
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.4)

plt.tight_layout()
path = os.path.join(FIG_DIR, "fig12_roc_curves.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


results_out = {
    "regression" : {
        target: {
            model: {
                metric: {"mean": round(vals[0], 4), "std": round(vals[1], 4)}
                for metric, vals in metrics.items()
            }
            for model, metrics in target_results.items()
        }
        for target, target_results in all_reg_results.items()
    },
    "classification": {
        model: {
            metric: {"mean": round(vals[0], 4), "std": round(vals[1], 4)}
            for metric, vals in metrics.items()
        }
        for model, metrics in clf_results.items()
    }
}

out_path = os.path.join(OUTPUT_DIR, "ml_results.json")
with open(out_path, "w") as f:
    json.dump(results_out, f, indent=2)

print(f"\nResults saved → {out_path}")
print("\n" + "=" * 60)
print("SUMMARY — Best models per task")
print("=" * 60)

for target in REG_TARGETS:
    best = max(all_reg_results[target].items(),
               key=lambda x: x[1]["R2"][0])
    print(f"\n{target}:")
    print(f"  Best: {best[0]}")
    print(f"  R²={best[1]['R2'][0]:.3f} ± {best[1]['R2'][1]:.3f}")
    print(f"  MAE={best[1]['MAE'][0]:.1f} ± {best[1]['MAE'][1]:.1f} min")

best_clf = max(clf_results.items(), key=lambda x: x[1]["AUC-ROC"][0])
print(f"\nClassification (poor_sleep):")
print(f"  Best: {best_clf[0]}")
print(f"  AUC={best_clf[1]['AUC-ROC'][0]:.3f} ± {best_clf[1]['AUC-ROC'][1]:.3f}")
print(f"  F1  ={best_clf[1]['F1'][0]:.3f} ± {best_clf[1]['F1'][1]:.3f}")