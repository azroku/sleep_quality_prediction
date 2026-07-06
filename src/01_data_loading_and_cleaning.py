# =============================================================================
# Step 1: Data Loading, Cleaning & Preprocessing
# =============================================================================

import pandas as pd
import numpy as np
import os
import json

DATA_DIR   = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\data"
OUTPUT_DIR = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SLEEP_PATH    = os.path.join(DATA_DIR, "Sleep-Didiconn-2025-11-01-2026-05-04.csv")
VITAL_PATH    = os.path.join(DATA_DIR, "Vital-Signs-Didiconn-2025-11-01-2026-05-04.csv")
ACTIVITY_PATH = os.path.join(DATA_DIR, "Activity-Didiconn-2025-11-01-2026-05-04.csv")


# SECTION 1 — LOAD RAW DATA

print("=" * 60)
print("SECTION 1: Loading raw data")
print("=" * 60)

sleep_raw    = pd.read_csv(SLEEP_PATH)
vital_raw    = pd.read_csv(VITAL_PATH)
activity_raw = pd.read_csv(ACTIVITY_PATH)

print(f"Sleep    shape : {sleep_raw.shape}")
print(f"Vital    shape : {vital_raw.shape}")
print(f"Activity shape : {activity_raw.shape}\n")

print("--- Sleep dtypes ---")
print(sleep_raw.dtypes)
print("\n--- Vital dtypes ---")
print(vital_raw.dtypes)
print("\n--- Activity dtypes ---")
print(activity_raw.dtypes)

# SECTION 2 — CLEAN SLEEP DATASET

print("\n" + "=" * 60)
print("SECTION 2: Cleaning sleep dataset")
print("=" * 60)

sleep = sleep_raw.copy()

datetime_cols = ["Start Time", "End Time", "Falling Asleep Time", "Wake-up time"]
for col in datetime_cols:
    sleep[col] = pd.to_datetime(sleep[col])

sleep["sleep_date"] = sleep["End Time"].dt.normalize()

#10 dates have two sleep records — always one long overnight session and one short (95–200 min) afternoon/evening session. The short ones are naps.
#keep the LONGEST session per day (main sleep), flag nap days.

nap_dates = sleep[sleep.duplicated("sleep_date", keep=False)]["sleep_date"].unique()
sleep["had_nap"] = sleep["sleep_date"].isin(nap_dates).astype(int)
print(f"Dates with multiple sleep records (naps): {len(nap_dates)}")

sleep = sleep.loc[
    sleep.groupby("sleep_date")["Time Asleep(min)"].idxmax()
].reset_index(drop=True)

print(f"Sleep rows after keeping longest per day: {len(sleep)}")

sleep["Sleep Time Ratio(%)"] = (
    sleep["Sleep Time Ratio(%)"]
    .str.replace("%", "", regex=False)
    .astype(float)
)

#Sleep Onset Latency (SOL): time from getting into bed to actually falling asleep.
#high SOL (>20 min) is a hallmark of insomnia and stress.
sleep["sleep_onset_latency_min"] = (
    (sleep["Falling Asleep Time"] - sleep["Start Time"])
    .dt.total_seconds() / 60
).round(1)

#Rename Awake column to WASO (Wake After Sleep Onset) — standard clinical term.
sleep.rename(columns={"Sleep Stages - Awake(min)": "waso_min"}, inplace=True)

#REM proportion: REM minutes / total minutes asleep.
sleep["rem_proportion"] = (
    sleep["Sleep Stages - REM(min)"] / sleep["Time Asleep(min)"]
).round(4)

#Deep sleep proportion — same reasoning as REM.
sleep["deep_proportion"] = (
    sleep["Sleep Stages - Deep Sleep(min)"] / sleep["Time Asleep(min)"]
).round(4)


# SECTION 3 — DEFINE PREDICTION TARGETS

print("\n" + "=" * 60)
print("SECTION 3: Defining prediction targets")
print("=" * 60)

#TASK A: Regression targets 
#train regression models to predict each of these as a continuous value.
#Time Asleep (min), REM duration (min), Deep Sleep duration (min) 

REGRESSION_TARGETS = [
    "Time Asleep(min)",
    "Sleep Stages - REM(min)",
    "Sleep Stages - Deep Sleep(min)",
]

print("Regression targets (Task A):")
for t in REGRESSION_TARGETS:
    print(f"  {t}: mean={sleep[t].mean():.1f}, "
          f"std={sleep[t].std():.1f}, "
          f"range=[{sleep[t].min()}, {sleep[t].max()}]")

#TASK B: Binary classification target 

threshold_pct = sleep["Sleep Time Ratio(%)"].quantile(0.25)
sleep["poor_sleep"] = (sleep["Sleep Time Ratio(%)"] <= threshold_pct).astype(int)

n_poor = sleep["poor_sleep"].sum()
n_good = len(sleep) - n_poor

print(f"\nClassification target (Task B):")
print(f"  Data-driven threshold (25th percentile): {threshold_pct:.2f}%")
print(f"  Poor sleep nights  : {n_poor} ({n_poor/len(sleep)*100:.1f}%)")
print(f"  Good sleep nights  : {n_good} ({n_good/len(sleep)*100:.1f}%)")
print(f"  Class ratio        : 1 : {n_good/n_poor:.1f}")
print(f"\n  NOTE — For comparison, the clinical 75% threshold gives only")
print(f"  1 poor night (0.6%) — unusable for training any model.")
print(f"  The 25th percentile threshold is justified for N-of-1 studies.")


# SECTION 4 — CLEAN VITAL SIGNS DATASET

print("\n" + "=" * 60)
print("SECTION 4: Cleaning vital signs dataset")
print("=" * 60)

vital = vital_raw.copy()

vital["Date"] = pd.to_datetime(vital["Date"]).dt.normalize()

spo2_cols = ["Avg. Spo2(%)", "Min. Spo2(%)", "Max. Spo2(%)"]
for col in spo2_cols:
    vital[col] = vital[col].str.replace("%", "", regex=False).astype(float)

print("SpO2 columns converted. Sample:")
print(vital[spo2_cols].head(3))

#Compute HRV range as an additional feature
vital["hrv_range_ms"] = vital["Max. HRV(ms)"] - vital["Min. HRV(ms)"]

print(f"\nHR  range: {vital['Avg. Heart Rate(bpm)'].min()}–"
      f"{vital['Avg. Heart Rate(bpm)'].max()} bpm")
print(f"HRV range: {vital['Avg. HRV(ms)'].min()}–"
      f"{vital['Avg. HRV(ms)'].max()} ms")
print(f"SpO2 range: {vital['Avg. Spo2(%)'].min()}–"
      f"{vital['Avg. Spo2(%)'].max()} %")


# SECTION 5 — CLEAN ACTIVITY DATASET

print("\n" + "=" * 60)
print("SECTION 5: Cleaning activity dataset")
print("=" * 60)

activity = activity_raw.copy()

activity["Date"] = pd.to_datetime(activity["Date"]).dt.normalize()

low_steps_threshold = 500
activity["possible_nonwear"] = (activity["Steps"] < low_steps_threshold).astype(int)

low_step_days = activity[activity["Steps"] < low_steps_threshold]
print(f"Possible non-wear days (< {low_steps_threshold} steps): {len(low_step_days)}")
print(low_step_days[["Date", "Steps", "Calories(kcal)"]])


# SECTION 6 — MERGE WITH t-1 LAG

print("\n" + "=" * 60)
print("SECTION 6: Merging datasets with t-1 lag")
print("=" * 60)

vital_lagged            = vital.copy()
activity_lagged         = activity.copy()
vital_lagged["join_date"]    = vital_lagged["Date"]    + pd.DateOffset(days=1)
activity_lagged["join_date"] = activity_lagged["Date"] + pd.DateOffset(days=1)

df = sleep.merge(
    vital_lagged.drop(columns="Date"),
    left_on="sleep_date",
    right_on="join_date",
    how="inner"
).drop(columns="join_date")

print(f"After merging sleep + vitals : {len(df)} rows")

df = df.merge(
    activity_lagged.drop(columns="Date"),
    left_on="sleep_date",
    right_on="join_date",
    how="inner"
).drop(columns="join_date")

print(f"After merging + activity     : {len(df)} rows")
print(f"Rows lost vs sleep-only      : {len(sleep) - len(df)}")


# SECTION 7 — FINAL COLUMN SELECTION

print("\n" + "=" * 60)
print("SECTION 7: Final column selection")
print("=" * 60)

DAYTIME_FEATURES = [
    # Heart rate — captures cardiovascular load and stress across the day
    "Avg. Heart Rate(bpm)", "Min. Heart Rate(bpm)", "Max. Heart Rate(bpm)",
    # HRV — KEY variable: low HRV → high sympathetic dominance → poor sleep
    "Avg. HRV(ms)", "Min. HRV(ms)", "Max. HRV(ms)", "hrv_range_ms",
    # SpO2 — blood oxygen; drops may indicate respiratory issues
    "Avg. Spo2(%)", "Min. Spo2(%)", "Max. Spo2(%)",
    # Physical activity — exercise affects sleep architecture strongly
    "Steps", "Calories(kcal)",
    # Quality flag
    "possible_nonwear",
]

REGRESSION_TARGETS = [
    "Time Asleep(min)",
    "Sleep Stages - REM(min)",
    "Sleep Stages - Deep Sleep(min)",
]

CLASSIFICATION_TARGET = ["poor_sleep"]

EXTRA_SLEEP_COLS = [
    "Sleep Stages - Light Sleep(min)",
    "waso_min",
    "Sleep Time Ratio(%)",
    "rem_proportion",
    "deep_proportion",
    "sleep_onset_latency_min",
]

METADATA = ["sleep_date", "had_nap"]

ALL_COLS = (METADATA + DAYTIME_FEATURES +
            REGRESSION_TARGETS + CLASSIFICATION_TARGET + EXTRA_SLEEP_COLS)

df_final = df[ALL_COLS].copy()
df_final = df_final.sort_values("sleep_date").reset_index(drop=True)

print(f"Final dataset shape : {df_final.shape}")
print(f"Date range          : {df_final['sleep_date'].min().date()} → "
      f"{df_final['sleep_date'].max().date()}")
print(f"\nFeatures  ({len(DAYTIME_FEATURES)}) : {DAYTIME_FEATURES}")
print(f"\nRegression targets ({len(REGRESSION_TARGETS)}) : {REGRESSION_TARGETS}")
print(f"Classification target : {CLASSIFICATION_TARGET}")


# SECTION 8 — MISSING VALUE CHECK

print("\n" + "=" * 60)
print("SECTION 8: Missing value check")
print("=" * 60)

missing = df_final.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("No missing values in final dataset.")
else:
    print("Missing values found — action required before modelling:")
    print(missing)


# SECTION 9 — DESCRIPTIVE STATISTICS (

print("\n" + "=" * 60)
print("SECTION 9: Descriptive statistics")
print("=" * 60)

analysis_cols = DAYTIME_FEATURES + REGRESSION_TARGETS + ["Sleep Time Ratio(%)"]
desc = df_final[analysis_cols].describe().T
desc = desc[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)
print(desc.to_string())

print(f"\nClass distribution (Task B — poor_sleep):")
print(f"  0 = Good sleep : {(df_final['poor_sleep']==0).sum()} nights")
print(f"  1 = Poor sleep : {(df_final['poor_sleep']==1).sum()} nights")
print(f"  Threshold used : Sleep Time Ratio <= {threshold_pct:.2f}% (25th percentile)")


# SECTION 10 — SAVE OUTPUTS

print("\n" + "=" * 60)
print("SECTION 10: Saving outputs")
print("=" * 60)

out_csv = os.path.join(OUTPUT_DIR, "sleep_merged_clean.csv")
df_final.to_csv(out_csv, index=False)
print(f"Cleaned dataset    → {out_csv}")

config = {
    "daytime_features"      : DAYTIME_FEATURES,
    "regression_targets"    : REGRESSION_TARGETS,
    "classification_target" : CLASSIFICATION_TARGET[0],
    "poor_sleep_threshold"  : threshold_pct,
    "n_samples"             : len(df_final),
    "n_poor_sleep"          : int(df_final["poor_sleep"].sum()),
    "n_good_sleep"          : int((df_final["poor_sleep"] == 0).sum()),
    "date_range_start"      : str(df_final["sleep_date"].min().date()),
    "date_range_end"        : str(df_final["sleep_date"].max().date()),
    "nap_days_removed"      : len(nap_dates),
}

out_cfg = os.path.join(OUTPUT_DIR, "project_config.json")
with open(out_cfg, "w") as f:
    json.dump(config, f, indent=2)
print(f"Project config     → {out_cfg}")