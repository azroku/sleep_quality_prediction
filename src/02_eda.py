# =============================================================================
# Step 2: Exploratory Data Analysis (EDA)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings("ignore")

DATA_PATH   = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output\sleep_merged_clean.csv"
CONFIG_PATH = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output\project_config.json"
OUTPUT_DIR  = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output\figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["sleep_date"] = pd.to_datetime(df["sleep_date"])

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

FEATURES = cfg["daytime_features"]
REG_TARGETS = cfg["regression_targets"]
CLF_TARGET  = cfg["classification_target"]
THRESHOLD   = cfg["poor_sleep_threshold"]

print(f"Dataset loaded: {df.shape[0]} nights, {len(FEATURES)} features")
print(f"Date range: {df['sleep_date'].min().date()} → {df['sleep_date'].max().date()}")
print(f"Poor sleep nights: {df[CLF_TARGET].sum()} / {len(df)}")

#Global plot style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
POOR_COLOR = "#E24B4A"   # red  — poor sleep nights
GOOD_COLOR = "#378ADD"   # blue — good sleep nights
HRV_COLOR  = "#1D9E75"   # teal — HRV line


# FIGURE 1 — Distribution of daytime features

print("\nPlotting Figure 1: Feature distributions...")
plot_features = [f for f in FEATURES if f != "possible_nonwear"]

n_cols = 4
n_rows = int(np.ceil(len(plot_features) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
axes = axes.flatten()

for i, col in enumerate(plot_features):
    ax = axes[i]
    data = df[col].dropna()

    ax.hist(data, bins=20, color=GOOD_COLOR, alpha=0.6, edgecolor="white",
            density=True, label="observed")

    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color="#E24B4A",
            linewidth=2, label="normal fit")

    skew = data.skew()
    skew_color = "#E24B4A" if abs(skew) > 1 else "#444441"
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_xlabel(f"skewness = {skew:+.2f}", fontsize=9, color=skew_color)
    ax.set_ylabel("density", fontsize=9)
    ax.legend(fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Figure 1 — Distribution of daytime features\n"
             "(red curve = expected normal distribution; red label = skewness > 1)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig1_feature_distributions.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 2 — Distribution of sleep targets
print("Plotting Figure 2: Sleep target distributions...")

extra_sleep = ["Sleep Time Ratio(%)", "waso_min", "sleep_onset_latency_min",
               "rem_proportion", "deep_proportion"]
all_sleep_cols = REG_TARGETS + extra_sleep

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
axes = axes.flatten()

for i, col in enumerate(all_sleep_cols):
    ax = axes[i]
    data = df[col].dropna()

    ax.hist(data, bins=20, color=HRV_COLOR, alpha=0.6,
            edgecolor="white", density=True)

    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color="#E24B4A", linewidth=2)

    skew = data.skew()
    skew_color = "#E24B4A" if abs(skew) > 1 else "#444441"
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_xlabel(f"mean={mu:.1f}, skew={skew:+.2f}", fontsize=9, color=skew_color)
    ax.set_ylabel("density", fontsize=9)

ax = axes[len(all_sleep_cols)]
counts = df[CLF_TARGET].value_counts().sort_index()
bars = ax.bar(["Good sleep\n(0)", "Poor sleep\n(1)"],
               counts.values,
               color=[GOOD_COLOR, POOR_COLOR],
               edgecolor="white", alpha=0.85)
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(count), ha="center", va="bottom", fontweight="bold")
ax.set_title("poor_sleep (classification target)", fontsize=10, fontweight="bold")
ax.set_ylabel("number of nights")
ax.set_xlabel(f"threshold: Sleep Ratio ≤ {THRESHOLD:.1f}%", fontsize=9)

for j in range(len(all_sleep_cols) + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Figure 2 — Distribution of sleep outcome variables",
             fontsize=13, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig2_sleep_distributions.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 3 — Correlation heatmap

print("Plotting Figure 3: Correlation heatmap...")

heatmap_cols = (
    ["Avg. Heart Rate(bpm)", "Avg. HRV(ms)", "Min. HRV(ms)", "Max. HRV(ms)",
     "hrv_range_ms", "Avg. Spo2(%)", "Min. Spo2(%)", "Steps", "Calories(kcal)"]
    + REG_TARGETS
    + ["Sleep Time Ratio(%)", "waso_min", "sleep_onset_latency_min"]
)

corr_matrix = df[heatmap_cols].corr(method="pearson")

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(14, 11))
cmap = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=cmap,
    vmin=-1, vmax=1,
    center=0,
    annot=True,          
    fmt=".2f",          
    square=True,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 8},
    cbar_kws={"shrink": 0.8, "label": "Pearson r"}
)

ax.set_title("Figure 3 — Pearson correlation matrix\n"
             "(features vs sleep outcomes)", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.tick_params(axis="y", rotation=0,  labelsize=9)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig3_correlation_heatmap.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 4 — Time series: HRV and sleep over 6 months

print("Plotting Figure 4: Time series...")

# 7-day rolling mean smooths out day-to-day noise so trends are visible.
df_ts = df.set_index("sleep_date").sort_index()
roll = df_ts.rolling(window=7, min_periods=3)

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

#Average HRV 
ax = axes[0]
ax.plot(df_ts.index, df_ts["Avg. HRV(ms)"], color=HRV_COLOR,
        alpha=0.25, linewidth=0.8, label="daily")
ax.plot(df_ts.index, roll["Avg. HRV(ms)"].mean(), color=HRV_COLOR,
        linewidth=2.2, label="7-day mean")
ax.set_ylabel("Avg. HRV (ms)", fontweight="bold")
ax.set_title("Figure 4 — Time series overview (6 months)", fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.4)

#Total time asleep
ax = axes[1]
ax.plot(df_ts.index, df_ts["Time Asleep(min)"], color=GOOD_COLOR,
        alpha=0.25, linewidth=0.8)
ax.plot(df_ts.index, roll["Time Asleep(min)"].mean(), color=GOOD_COLOR,
        linewidth=2.2)
# Mark poor sleep nights as red dots
poor = df_ts[df_ts[CLF_TARGET] == 1]
ax.scatter(poor.index, poor["Time Asleep(min)"],
           color=POOR_COLOR, zorder=5, s=30, label="poor sleep night", alpha=0.8)
ax.set_ylabel("Time asleep (min)", fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.4)

#REM duration 
ax = axes[2]
ax.plot(df_ts.index, df_ts["Sleep Stages - REM(min)"], color="#534AB7",
        alpha=0.25, linewidth=0.8)
ax.plot(df_ts.index, roll["Sleep Stages - REM(min)"].mean(), color="#534AB7",
        linewidth=2.2)
ax.set_ylabel("REM duration (min)", fontweight="bold")
ax.grid(axis="y", alpha=0.4)

#Steps 
ax = axes[3]
ax.bar(df_ts.index, df_ts["Steps"], color="#BA7517", alpha=0.45, width=1.0)
ax.plot(df_ts.index, roll["Steps"].mean(), color="#BA7517",
        linewidth=2.2)
ax.set_ylabel("Steps", fontweight="bold")
ax.grid(axis="y", alpha=0.4)

axes[3].xaxis.set_major_locator(mdates.MonthLocator())
axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[3].xaxis.set_minor_locator(mdates.WeekdayLocator())
plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig4_time_series.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 5 — Lag correlation: daytime HRV → next-night sleep

print("Plotting Figure 5: Lag correlations...")

df_lag = df.set_index("sleep_date").sort_index().copy()
lags = range(0, 8)   # 0 to 7 days back
hrv_features = ["Avg. HRV(ms)", "Min. HRV(ms)", "hrv_range_ms"]
sleep_targets_lag = ["Time Asleep(min)", "Sleep Stages - REM(min)",
                     "Sleep Stages - Deep Sleep(min)", "Sleep Time Ratio(%)"]

fig, axes = plt.subplots(1, len(sleep_targets_lag),
                          figsize=(18, 5), sharey=False)

colors_lag = [HRV_COLOR, "#534AB7", POOR_COLOR]

for ax, target in zip(axes, sleep_targets_lag):
    for hrv_col, color in zip(hrv_features, colors_lag):
        lag_corrs = []
        for lag in lags:
            # shift(lag): move the HRV values DOWN by `lag` rows,
            # so row i gets the HRV from lag days earlier.
            shifted = df_lag[hrv_col].shift(lag)
            corr = df_lag[target].corr(shifted)
            lag_corrs.append(corr)

        ax.plot(list(lags), lag_corrs, marker="o", color=color,
                linewidth=2, markersize=6, label=hrv_col)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(1, color="orange", linewidth=1.2, linestyle=":",
               label="lag used (t-1)")

    ax.set_title(target, fontsize=10, fontweight="bold")
    ax.set_xlabel("lag (days)", fontsize=9)
    ax.set_ylabel("Pearson r", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)

fig.suptitle("Figure 5 — Lag correlation: HRV from N days ago vs next-night sleep\n"
             "(lag=1 = yesterday's HRV; the lag used in all models)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig5_lag_correlations.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 6 — Scatter plots: key features vs sleep targets

print("Plotting Figure 6: Scatter plots...")

scatter_pairs = [
    ("Avg. HRV(ms)",  "Time Asleep(min)"),
    ("Avg. HRV(ms)",  "Sleep Stages - REM(min)"),
    ("Avg. HRV(ms)",  "Sleep Stages - Deep Sleep(min)"),
    ("Steps",         "Time Asleep(min)"),
    ("Steps",         "Sleep Stages - REM(min)"),
    ("Steps",         "Sleep Stages - Deep Sleep(min)"),
    ("Avg. Heart Rate(bpm)", "Sleep Stages - REM(min)"),
    ("Min. Spo2(%)",  "Sleep Stages - Deep Sleep(min)"),
]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for ax, (x_col, y_col) in zip(axes, scatter_pairs):
    colors_pts = df[CLF_TARGET].map({0: GOOD_COLOR, 1: POOR_COLOR})
    ax.scatter(df[x_col], df[y_col], c=colors_pts,
               alpha=0.55, s=30, edgecolors="none")

    slope, intercept, r, p, _ = stats.linregress(df[x_col].dropna(),
                                                   df[y_col].dropna())
    x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    ax.plot(x_range, slope * x_range + intercept,
            color="#444441", linewidth=1.8, linestyle="--")

    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    ax.set_title(f"{x_col}\nvs {y_col}", fontsize=9, fontweight="bold")
    ax.set_xlabel(x_col, fontsize=8)
    ax.set_ylabel(y_col, fontsize=8)
    ax.text(0.05, 0.93, f"r={r:.2f}, {p_str}",
            transform=ax.transAxes, fontsize=9,
            color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="gray", alpha=0.8))

from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=GOOD_COLOR, label="good sleep"),
                Patch(facecolor=POOR_COLOR, label="poor sleep")]
fig.legend(handles=legend_elems, loc="lower center", ncol=2,
           fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

fig.suptitle("Figure 6 — Scatter plots: daytime features vs sleep outcomes\n"
             "(blue = good sleep night, red = poor sleep night)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig6_scatter_plots.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 7 — Box plots: poor vs good sleep nights

print("Plotting Figure 7: Box plots by sleep class...")

box_features = [
    "Avg. HRV(ms)", "Min. HRV(ms)", "hrv_range_ms",
    "Avg. Heart Rate(bpm)", "Steps", "Calories(kcal)",
    "Avg. Spo2(%)", "Min. Spo2(%)"
]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for ax, col in zip(axes, box_features):
    good_data = df[df[CLF_TARGET] == 0][col]
    poor_data  = df[df[CLF_TARGET] == 1][col]

    bp = ax.boxplot(
        [good_data, poor_data],
        patch_artist=True,
        tick_labels=["Good sleep", "Poor sleep"],
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5)
    )
    bp["boxes"][0].set_facecolor(GOOD_COLOR)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(POOR_COLOR)
    bp["boxes"][1].set_alpha(0.7)

    stat, p = stats.mannwhitneyu(good_data, poor_data, alternative="two-sided")
    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    sig = " *" if p < 0.05 else ""

    ax.set_title(f"{col}{sig}", fontsize=10, fontweight="bold")
    ax.set_ylabel(col, fontsize=8)
    ax.text(0.5, 0.97, f"Mann-Whitney {p_str}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="gray", alpha=0.8))
    ax.grid(axis="y", alpha=0.4)

fig.suptitle("Figure 7 — Feature distributions: good vs poor sleep nights\n"
             "(* = statistically significant difference, p < 0.05)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig7_boxplots_class.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 8 — Sleep stage composition

print("Plotting Figure 8: Sleep stage composition...")

df_stages = df.set_index("sleep_date").sort_index()
stage_cols = [
    "Sleep Stages - Deep Sleep(min)",
    "Sleep Stages - REM(min)",
    "Sleep Stages - Light Sleep(min)",
    "waso_min"
]
stage_labels = ["Deep sleep", "REM", "Light sleep", "Awake (WASO)"]
stage_colors = ["#0F6E56", "#534AB7", "#378ADD", "#E24B4A"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

bottom = np.zeros(len(df_stages))
for col, label, color in zip(stage_cols, stage_labels, stage_colors):
    vals = df_stages[col].values
    ax1.bar(df_stages.index, vals, bottom=bottom, color=color,
            alpha=0.85, label=label, width=1.0)
    bottom += vals

ax1.set_ylabel("Duration (min)", fontweight="bold")
ax1.set_title("Figure 8 — Sleep stage composition per night", fontweight="bold")
ax1.legend(loc="upper right", fontsize=9)
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.grid(axis="y", alpha=0.3)

roll7 = df_stages[["rem_proportion", "deep_proportion"]].rolling(7, min_periods=3).mean()
ax2.plot(df_stages.index, roll7["rem_proportion"] * 100,
         color="#534AB7", linewidth=2, label="REM % (7-day mean)")
ax2.plot(df_stages.index, roll7["deep_proportion"] * 100,
         color="#0F6E56", linewidth=2, label="Deep % (7-day mean)")
ax2.axhline(20, color="#534AB7", linewidth=0.8, linestyle=":",
            alpha=0.6, label="20% REM reference")
ax2.axhline(13, color="#0F6E56", linewidth=0.8, linestyle=":",
            alpha=0.6, label="13% deep reference")
ax2.set_ylabel("Stage proportion (%)", fontweight="bold")
ax2.set_xlabel("Date", fontweight="bold")
ax2.legend(loc="upper right", fontsize=9)
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.grid(alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig8_sleep_stages.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")



print("\n" + "=" * 60)
print("KEY FINDINGS — paste these into your paper")
print("=" * 60)

# Correlations between HRV and sleep
r_hrv_rem, p_hrv_rem = stats.pearsonr(df["Avg. HRV(ms)"],
                                        df["Sleep Stages - REM(min)"])
r_hrv_deep, p_hrv_deep = stats.pearsonr(df["Avg. HRV(ms)"],
                                          df["Sleep Stages - Deep Sleep(min)"])
r_hrv_total, p_hrv_total = stats.pearsonr(df["Avg. HRV(ms)"],
                                            df["Time Asleep(min)"])
r_steps_deep, p_steps_deep = stats.pearsonr(df["Steps"],
                                              df["Sleep Stages - Deep Sleep(min)"])

print(f"\nHRV → REM      : r={r_hrv_rem:.3f}, p={p_hrv_rem:.3f}")
print(f"HRV → Deep     : r={r_hrv_deep:.3f}, p={p_hrv_deep:.3f}")
print(f"HRV → Total    : r={r_hrv_total:.3f}, p={p_hrv_total:.3f}")
print(f"Steps → Deep   : r={r_steps_deep:.3f}, p={p_steps_deep:.3f}")

stat, p_mw = stats.mannwhitneyu(
    df[df[CLF_TARGET]==0]["Avg. HRV(ms)"],
    df[df[CLF_TARGET]==1]["Avg. HRV(ms)"],
    alternative="two-sided"
)
print(f"\nMann-Whitney HRV (good vs poor): U={stat:.0f}, p={p_mw:.3f}")
print(f"  Mean HRV good nights: {df[df[CLF_TARGET]==0]['Avg. HRV(ms)'].mean():.1f} ms")
print(f"  Mean HRV poor nights: {df[df[CLF_TARGET]==1]['Avg. HRV(ms)'].mean():.1f} ms")

print(f"\nDescriptive stats (key variables):")
key_cols = ["Avg. HRV(ms)", "Steps", "Time Asleep(min)",
            "Sleep Stages - REM(min)", "Sleep Stages - Deep Sleep(min)",
            "Sleep Time Ratio(%)"]
print(df[key_cols].describe().round(1).to_string())

print(f"\nAll 8 figures saved to: {OUTPUT_DIR}")