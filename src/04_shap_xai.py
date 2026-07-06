# =============================================================================
# Step 4: Explainable AI (XAI) with SHAP
# =============================================================================
#   SHAP (SHapley Additive exPlanations) comes from cooperative game theory.
#   The core idea: each feature is a "player" in a game, and the prediction
#   is the "payout". SHAP fairly distributes the payout among players by
#   asking: "how much does each feature contribute to this prediction,
#   compared to the average prediction?"
#
#   For each sample and each feature:
#     SHAP value > 0  → this feature INCREASED the prediction above average
#     SHAP value < 0  → this feature DECREASED the prediction below average
#     SHAP value = 0  → this feature had no effect
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import json, os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble    import RandomForestRegressor, RandomForestClassifier
from sklearn.svm         import SVC
from sklearn.preprocessing import StandardScaler
from xgboost             import XGBRegressor, XGBClassifier

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

FEATURES    = cfg["daytime_features"]
REG_TARGETS = cfg["regression_targets"]
CLF_TARGET  = cfg["classification_target"]

FEAT_LABELS = {
    "Avg. Heart Rate(bpm)" : "Avg HR",
    "Min. Heart Rate(bpm)" : "Min HR",
    "Max. Heart Rate(bpm)" : "Max HR",
    "Avg. HRV(ms)"         : "Avg HRV",
    "Min. HRV(ms)"         : "Min HRV",
    "Max. HRV(ms)"         : "Max HRV",
    "hrv_range_ms"         : "HRV Range",
    "Avg. Spo2(%)"         : "Avg SpO₂",
    "Min. Spo2(%)"         : "Min SpO₂",
    "Max. Spo2(%)"         : "Max SpO₂",
    "Steps"                : "Steps",
    "Calories(kcal)"       : "Calories",
    "possible_nonwear"     : "Non-wear flag",
}
SHORT_NAMES = [FEAT_LABELS[f] for f in FEATURES]

TARGET_LABELS = {
    "Time Asleep(min)"              : "Total Sleep (min)",
    "Sleep Stages - REM(min)"       : "REM Sleep (min)",
    "Sleep Stages - Deep Sleep(min)": "Deep Sleep (min)",
}

X     = df[FEATURES].values
X_df  = pd.DataFrame(X, columns=FEATURES)

scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
X_sc_df = pd.DataFrame(X_sc, columns=FEATURES)

sns.set_theme(style="whitegrid", font_scale=1.05)
POOR_COLOR = "#E24B4A"
GOOD_COLOR = "#378ADD"
HRV_COLOR  = "#1D9E75"

print("SHAP XAI Analysis")
print(f"Dataset: {X.shape[0]} nights, {X.shape[1]} features")
print(f"SHAP version: {shap.__version__}\n")


# FIT MODELS ON FULL DATA (for SHAP)

print("Fitting models for SHAP analysis...")

rf_regressors = {}
shap_values_reg = {}

for target in REG_TARGETS:
    y = df[target].values
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_sc, y)
    rf_regressors[target] = rf

    explainer   = shap.TreeExplainer(rf)
    shap_vals   = explainer.shap_values(X_sc)
    shap_values_reg[target] = shap_vals
    print(f"  RF regressor → {target}: SHAP computed, shape={shap_vals.shape}")

y_clf = df[CLF_TARGET].values
rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_sc, y_clf)
explainer_clf   = shap.TreeExplainer(rf_clf)
# For binary RF, shap_values returns [class0_vals, class1_vals]
# We take class 1 (poor sleep) — positive class
shap_vals_clf_raw = explainer_clf.shap_values(X_sc)
if isinstance(shap_vals_clf_raw, list):
    shap_vals_clf = shap_vals_clf_raw[1]
elif shap_vals_clf_raw.ndim == 3:
    shap_vals_clf = shap_vals_clf_raw[:, :, 1]
else:
    shap_vals_clf = shap_vals_clf_raw
print(f"  RF classifier → poor_sleep: SHAP computed, shape={shap_vals_clf.shape}")


# FIGURE 16 — SHAP Beeswarm (Summary Plot)

print("\nPlotting Figure 16: SHAP beeswarm plots...")

fig, axes = plt.subplots(1, len(REG_TARGETS), figsize=(20, 7))

for ax, target in zip(axes, REG_TARGETS):
    plt.sca(ax)

    shap.summary_plot(
        shap_values_reg[target],
        X_sc_df,
        feature_names=SHORT_NAMES,
        plot_type="dot",       # beeswarm
        show=False,
        max_display=13,        
        color_bar=False,       
        plot_size=None,
    )

    ax.set_title(TARGET_LABELS[target], fontweight="bold", fontsize=11)
    ax.set_xlabel("SHAP value\n(impact on predicted sleep duration)", fontsize=9)

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

fig.suptitle(
    "Figure 16 — SHAP beeswarm plots: feature impact on sleep predictions\n"
    "Colour = feature value (red=high, blue=low). "
    "X-axis = effect on prediction (right=increases, left=decreases).",
    fontsize=11, fontweight="bold", y=1.01
)
plt.tight_layout()
path = os.path.join(FIG_DIR, "fig16_shap_beeswarm.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 17 — SHAP Mean |SHAP| bar chart (all targets side by side)

print("Plotting Figure 17: SHAP mean |SHAP| bar chart...")

fig, axes = plt.subplots(1, len(REG_TARGETS), figsize=(18, 6))
colors_bar = ["#378ADD", "#1D9E75", "#534AB7"]

for ax, target, color in zip(axes, REG_TARGETS, colors_bar):
    mean_abs = np.abs(shap_values_reg[target]).mean(axis=0)
    order    = np.argsort(mean_abs)   

    ax.barh(
        [SHORT_NAMES[i] for i in order],
        [mean_abs[i]    for i in order],
        color=color, alpha=0.82, edgecolor="white"
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.set_title(TARGET_LABELS[target], fontweight="bold", fontsize=10)
    ax.grid(axis="x", alpha=0.4)

    top3 = np.argsort(mean_abs)[-3:]
    for idx in top3:
        ax.text(
            mean_abs[idx] * 1.02,
            list(np.argsort(mean_abs)).index(idx),
            f"{mean_abs[idx]:.2f}",
            va="center", fontsize=8, fontweight="bold", color=color
        )

fig.suptitle(
    "Figure 17 — Global SHAP feature importance: mean |SHAP| per target\n"
    "Higher = feature has larger average impact on that sleep outcome",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
path = os.path.join(FIG_DIR, "fig17_shap_importance_bar.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 18 — SHAP Dependence Plots: HRV → sleep outcomes
#   The colour of each dot = value of the most interacting feature

print("Plotting Figure 18: SHAP dependence plots (HRV)...")

hrv_features = ["Avg. HRV(ms)", "Min. HRV(ms)", "hrv_range_ms"]
fig, axes = plt.subplots(len(hrv_features), len(REG_TARGETS),
                          figsize=(18, 14))

for row, hrv_feat in enumerate(hrv_features):
    feat_idx = FEATURES.index(hrv_feat)

    for col, target in enumerate(REG_TARGETS):
        ax = axes[row, col]
        shap_col = shap_values_reg[target][:, feat_idx]
        feat_col = X_sc[:, feat_idx]

        steps_idx  = FEATURES.index("Steps")
        steps_vals = X[:, steps_idx]   

        sc = ax.scatter(
            X[:, feat_idx],  
            shap_col,
            c=steps_vals,
            cmap="RdYlGn",
            alpha=0.6,
            s=25,
            edgecolors="none"
        )

        z = np.polyfit(X[:, feat_idx], shap_col, 1)
        p = np.poly1d(z)
        x_range = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 100)
        ax.plot(x_range, p(x_range), color="#444441",
                linewidth=2, linestyle="--", alpha=0.8)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xlabel(FEAT_LABELS[hrv_feat], fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(f"{FEAT_LABELS[hrv_feat]}\n→ {TARGET_LABELS[target]}",
                     fontsize=9, fontweight="bold")
        ax.grid(alpha=0.3)

        from scipy import stats as sp_stats
        r, p_val = sp_stats.pearsonr(X[:, feat_idx], shap_col)
        p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
        ax.text(0.05, 0.93, f"r={r:.2f}, {p_str}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="gray", alpha=0.8))

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                             norm=plt.Normalize(steps_vals.min(), steps_vals.max()))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Steps (colour)", fontsize=9)

fig.suptitle(
    "Figure 18 — SHAP dependence plots: HRV features vs sleep outcomes\n"
    "Each dot = one night. Colour = step count (green=active, red=sedentary).\n"
    "Trend line: positive slope → higher HRV predicts more sleep",
    fontsize=11, fontweight="bold"
)
path = os.path.join(FIG_DIR, "fig18_shap_dependence_hrv.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 19 — SHAP Waterfall: local explanation of specific nights

print("Plotting Figure 19: SHAP waterfall (local explanations)...")

#using REM as the target for local explanation 
target_local = "Sleep Stages - REM(min)"
shap_local   = shap_values_reg[target_local]
rf_local     = rf_regressors[target_local]
baseline     = rf_local.predict(X_sc).mean()

#finding worst predicted night (lowest predicted REM) and best
preds = rf_local.predict(X_sc)
worst_idx = np.argmin(preds)
best_idx  = np.argmax(preds)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, idx, label, color in [
    (axes[0], worst_idx, "Night with lowest predicted REM", POOR_COLOR),
    (axes[1], best_idx,  "Night with highest predicted REM", GOOD_COLOR)
]:
    night_shap = shap_local[idx]
    order      = np.argsort(np.abs(night_shap))   
    names_ord  = [SHORT_NAMES[i]  for i in order]
    vals_ord   = [night_shap[i]   for i in order]
    bar_colors = [POOR_COLOR if v < 0 else GOOD_COLOR for v in vals_ord]

    ax.barh(names_ord, vals_ord, color=bar_colors, alpha=0.82, edgecolor="white")
    ax.axvline(0, color="gray", linewidth=1.0)
    ax.set_xlabel("SHAP value (contribution to prediction)", fontsize=9)
    ax.set_title(
        f"{label}\n"
        f"Date: {df['sleep_date'].iloc[idx].date()}\n"
        f"Actual REM: {df[target_local].iloc[idx]} min | "
        f"Predicted: {preds[idx]:.0f} min",
        fontsize=9, fontweight="bold", color=color
    )
    ax.grid(axis="x", alpha=0.4)

    ax.text(0.02, 0.02,
            f"Baseline (avg prediction): {baseline:.0f} min",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

red_patch  = mpatches.Patch(color=POOR_COLOR, label="Reduces predicted REM")
blue_patch = mpatches.Patch(color=GOOD_COLOR, label="Increases predicted REM")
fig.legend(handles=[red_patch, blue_patch], loc="lower center",
           ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.03))

fig.suptitle(
    "Figure 19 — Local SHAP explanations (waterfall): REM sleep\n"
    "Why the model predicted extreme outcomes for specific nights",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
path = os.path.join(FIG_DIR, "fig19_shap_waterfall.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 20 — SHAP for Classification (poor_sleep prediction)
# Positive SHAP = pushes toward poor sleep prediction.
# Negative SHAP = pushes toward good sleep prediction.

print("Plotting Figure 20: SHAP classification analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

plt.sca(axes[0])
shap.summary_plot(
    shap_vals_clf,
    X_sc_df,
    feature_names=SHORT_NAMES,
    plot_type="dot",
    show=False,
    max_display=13,
    color_bar=True,
    plot_size=None,
)
axes[0].set_title("SHAP beeswarm: poor sleep classifier\n"
                   "Positive SHAP → predicts poor sleep",
                   fontweight="bold", fontsize=10)
axes[0].set_xlabel("SHAP value (impact on poor sleep probability)", fontsize=9)
axes[0].axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

ax = axes[1]
mean_abs_clf = np.abs(shap_vals_clf).mean(axis=0)
order        = np.argsort(mean_abs_clf)

ax.barh(
    [SHORT_NAMES[i] for i in order],
    [mean_abs_clf[i] for i in order],
    color=POOR_COLOR, alpha=0.80, edgecolor="white"
)
ax.set_xlabel("Mean |SHAP value|", fontsize=9)
ax.set_title("Global feature importance\nfor poor sleep classification",
             fontweight="bold", fontsize=10)
ax.grid(axis="x", alpha=0.4)

top5 = np.argsort(mean_abs_clf)[-5:]
for idx in top5:
    pos = list(np.argsort(mean_abs_clf)).index(idx)
    ax.text(mean_abs_clf[idx] * 1.02, pos,
            f"{mean_abs_clf[idx]:.4f}",
            va="center", fontsize=8, fontweight="bold", color=POOR_COLOR)

fig.suptitle(
    "Figure 20 — SHAP analysis: poor sleep night classification\n"
    "Which daytime features most influence the model's poor-sleep prediction?",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
path = os.path.join(FIG_DIR, "fig20_shap_classification.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


print("\n" + "=" * 60)
print("KEY SHAP FINDINGS")
print("=" * 60)

for target in REG_TARGETS:
    mean_abs = np.abs(shap_values_reg[target]).mean(axis=0)
    ranked   = sorted(zip(SHORT_NAMES, mean_abs), key=lambda x: -x[1])
    print(f"\n{TARGET_LABELS[target]}:")
    print("  Top 5 most influential features:")
    for i, (name, val) in enumerate(ranked[:5]):
        print(f"    {i+1}. {name:20s}: mean |SHAP| = {val:.3f}")

    hrv_idx  = FEATURES.index("Avg. HRV(ms)")
    hrv_shap = shap_values_reg[target][:, hrv_idx]
    hrv_vals = X[:, hrv_idx]
    from scipy import stats as sp_stats
    r, p = sp_stats.pearsonr(hrv_vals, hrv_shap)
    direction = "POSITIVE" if r > 0 else "NEGATIVE"
    print(f"  HRV direction of effect: r={r:.3f} ({direction})")
    if r > 0:
        print(f"  → Higher daytime HRV → model predicts MORE {TARGET_LABELS[target]}")
    else:
        print(f"  → Higher daytime HRV → model predicts LESS {TARGET_LABELS[target]}")

print(f"\nClassification (poor_sleep):")
mean_abs_clf = np.abs(shap_vals_clf).mean(axis=0)
ranked_clf   = sorted(zip(SHORT_NAMES, mean_abs_clf), key=lambda x: -x[1])
print("  Top 5 most influential features:")
for i, (name, val) in enumerate(ranked_clf[:5]):
    print(f"    {i+1}. {name:20s}: mean |SHAP| = {val:.4f}")

hrv_idx_clf  = FEATURES.index("Avg. HRV(ms)")
hrv_shap_clf = shap_vals_clf[:, hrv_idx_clf]
hrv_vals_clf = X[:, hrv_idx_clf]
r_clf, p_clf = sp_stats.pearsonr(hrv_vals_clf, hrv_shap_clf)
print(f"\n  HRV effect on poor-sleep prediction: r={r_clf:.3f}")
if r_clf < 0:
    print("  → Higher HRV → model LESS likely to predict poor sleep (protective)")
else:
    print("  → Higher HRV → model MORE likely to predict poor sleep")

shap_summary = {}
for target in REG_TARGETS:
    mean_abs = np.abs(shap_values_reg[target]).mean(axis=0)
    shap_summary[target] = {
        name: round(float(val), 4)
        for name, val in zip(SHORT_NAMES, mean_abs)
    }
shap_summary["poor_sleep_classification"] = {
    name: round(float(val), 4)
    for name, val in zip(SHORT_NAMES, np.abs(shap_vals_clf).mean(axis=0))
}

out_path = os.path.join(OUTPUT_DIR, "shap_summary.json")
with open(out_path, "w") as f:
    json.dump(shap_summary, f, indent=2)
print(f"\nSHAP summary saved → {out_path}")
print(f"All figures saved  → {FIG_DIR}")