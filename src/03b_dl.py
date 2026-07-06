# =============================================================================
# Step 3b: Deep Learning Models (LSTM + 1D-CNN)
#
# MODELS:
#   LSTM (Long Short-Term Memory):
#     A recurrent neural network designed specifically for sequences.
#     It has memory cells that can learn to "remember" or "forget"
#     information across timesteps. With a 7-day window, the LSTM sees
#     7 consecutive days of physiological data and predicts the 8th night.
#
#   1D-CNN (1-Dimensional Convolutional Neural Network):
#     Applies convolutional filters along the time dimension to detect
#     local temporal patterns (e.g. "HRV dropped for 3 consecutive days").
#     Often faster and sometimes more accurate than LSTM on short sequences.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              roc_auc_score, f1_score, balanced_accuracy_score,
                              precision_score, recall_score)
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH   = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output\sleep_merged_clean.csv"
CONFIG_PATH = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output\project_config.json"
ML_RESULTS  = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output\ml_results.json"
OUTPUT_DIR  = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output"
FIG_DIR     = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output\figures"

os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["sleep_date"] = pd.to_datetime(df["sleep_date"])
df = df.sort_values("sleep_date").reset_index(drop=True)

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

with open(ML_RESULTS) as f:
    ml_results = json.load(f)

FEATURES    = cfg["daytime_features"]
REG_TARGETS = cfg["regression_targets"]
CLF_TARGET  = cfg["classification_target"]

print(f"Dataset: {df.shape[0]} nights, {len(FEATURES)} features")
print(f"TensorFlow: {tf.__version__}")

sns.set_theme(style="whitegrid", font_scale=1.05)
COLORS = ["#378ADD", "#1D9E75", "#534AB7", "#E24B4A", "#BA7517"]


# BUILDING SLIDING WINDOW SEQUENCES

WINDOW_SIZE = 7   # days of history to use as input

def make_sequences(X_arr, y_arr, window=7):
    """
    Convert flat arrays into (samples, timesteps, features) format.

    Returns:
        Xs: shape (n_samples, window, n_features)
        ys: shape (n_samples,)
    """
    Xs, ys = [], []
    for i in range(window, len(X_arr)):
        Xs.append(X_arr[i - window : i])   
        ys.append(y_arr[i])                
    return np.array(Xs), np.array(ys)

X_flat = df[FEATURES].values
print(f"\nSliding window: {WINDOW_SIZE} days → {len(X_flat) - WINDOW_SIZE} sequence samples")


# WALK-FORWARD CV FOR SEQUENCES

N = len(X_flat)
N_SPLITS = 5

# Build fold boundaries at the raw data level
fold_size = N // (N_SPLITS + 1)
folds = []
for k in range(N_SPLITS):
    train_end  = fold_size * (k + 1) + WINDOW_SIZE
    test_start = train_end
    test_end   = min(test_start + fold_size, N)
    if test_end > test_start + WINDOW_SIZE:
        folds.append((train_end, test_start, test_end))

print(f"CV folds (raw indices): {len(folds)}")
for i, (te, ts, te2) in enumerate(folds):
    print(f"  Fold {i+1}: train[0:{te}] → test[{ts}:{te2}]")


# MODEL BUILDERS

def build_lstm_regressor(n_features, units=64, dropout=0.3):
    """
    Architecture:
      Input (7, 13) → LSTM(64) → Dropout → Dense(32) → Dense(1)
    """
    inp = keras.Input(shape=(WINDOW_SIZE, n_features))
    x   = layers.LSTM(units, return_sequences=False)(inp)
    x   = layers.Dropout(dropout)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)    

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",   
        metrics=["mae"]
    )
    return model


def build_cnn_regressor(n_features, filters=64, dropout=0.3):
    """
    Architecture:
      Input (7, 13) → Conv1D(64, k=3) → MaxPool → Conv1D(32, k=2)
                     → GlobalAvgPool → Dropout → Dense(32) → Dense(1)
    """
    inp = keras.Input(shape=(WINDOW_SIZE, n_features))
    x   = layers.Conv1D(filters, kernel_size=3, activation="relu", padding="same")(inp)
    x   = layers.MaxPooling1D(pool_size=2)(x)
    x   = layers.Conv1D(filters // 2, kernel_size=2, activation="relu", padding="same")(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dropout(dropout)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


def build_lstm_classifier(n_features, units=64, dropout=0.3):
    """
    Output: sigmoid activation → probability of poor sleep (0 to 1)
    Loss: binary_crossentropy — standard for binary classification.
    """
    inp = keras.Input(shape=(WINDOW_SIZE, n_features))
    x   = layers.LSTM(units, return_sequences=False)(inp)
    x   = layers.Dropout(dropout)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)  

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_cnn_classifier(n_features, filters=64, dropout=0.3):
    """1D-CNN binary classification model."""
    inp = keras.Input(shape=(WINDOW_SIZE, n_features))
    x   = layers.Conv1D(filters, kernel_size=3, activation="relu", padding="same")(inp)
    x   = layers.MaxPooling1D(pool_size=2)(x)
    x   = layers.Conv1D(filters // 2, kernel_size=2, activation="relu", padding="same")(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dropout(dropout)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# training settings
EPOCHS    = 80
BATCH     = 16   
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=15,
    restore_best_weights=True, verbose=0
)


# TASK A — DL REGRESSION
print("\n" + "=" * 60)
print("TASK A: DL REGRESSION")
print("=" * 60)

dl_reg_results  = {}   # {target: {model_name: {metric: (mean, std)}}}
history_store   = {}   # for learning curve plots

for target in REG_TARGETS:
    y_flat = df[target].values
    print(f"\n── Target: {target} ──")

    dl_reg_results[target] = {}
    history_store[target]  = {}

    for model_name, builder in [("LSTM", build_lstm_regressor),
                                  ("1D-CNN", build_cnn_regressor)]:
        maes, rmses, r2s = [], [], []
        histories = []

        for fold_i, (train_end, test_start, test_end) in enumerate(folds):
            scaler = StandardScaler()
            X_train_raw = X_flat[:train_end]
            X_test_raw  = X_flat[test_start:test_end]

            X_train_sc = scaler.fit_transform(X_train_raw)
            X_test_sc  = scaler.transform(X_test_raw)

            y_train_raw = y_flat[:train_end]
            y_test_raw  = y_flat[test_start:test_end]

            Xs_train, ys_train = make_sequences(X_train_sc, y_train_raw, WINDOW_SIZE)
            Xs_test,  ys_test  = make_sequences(X_test_sc,  y_test_raw,  WINDOW_SIZE)

            if len(Xs_test) == 0:
                continue

            model = builder(len(FEATURES))
            hist = model.fit(
                Xs_train, ys_train,
                epochs=EPOCHS,
                batch_size=BATCH,
                validation_split=0.15,  # 15% of train for early stopping monitor
                callbacks=[early_stop],
                verbose=0
            )
            histories.append(hist.history)

            y_pred = model.predict(Xs_test, verbose=0).flatten()
            maes.append(mean_absolute_error(ys_test, y_pred))
            rmses.append(np.sqrt(mean_squared_error(ys_test, y_pred)))
            r2s.append(r2_score(ys_test, y_pred))

        results = {
            "MAE"  : (np.mean(maes),  np.std(maes)),
            "RMSE" : (np.mean(rmses), np.std(rmses)),
            "R2"   : (np.mean(r2s),   np.std(r2s)),
        }
        dl_reg_results[target][model_name] = results
        history_store[target][model_name]  = histories

        print(f"  {model_name:10s} — MAE={results['MAE'][0]:.1f}±{results['MAE'][1]:.1f}  "
              f"RMSE={results['RMSE'][0]:.1f}±{results['RMSE'][1]:.1f}  "
              f"R²={results['R2'][0]:.3f}±{results['R2'][1]:.3f}")


# TASK B — DL CLASSIFICATION
print("\n" + "=" * 60)
print("TASK B: DL CLASSIFICATION")
print("=" * 60)

y_clf_flat = df[CLF_TARGET].values
# Class weights for imbalanced classification
n_neg = (y_clf_flat == 0).sum()
n_pos = (y_clf_flat == 1).sum()
class_weight_dict = {0: n_neg / (2 * n_neg),
                     1: n_pos / (2 * n_pos) * (n_neg / n_pos)}
class_weight_dict = {0: 1.0, 1: n_neg / n_pos}
print(f"Class weights: 0→1.0, 1→{n_neg/n_pos:.2f}  "
      f"(minority class gets {n_neg/n_pos:.1f}× more weight)")

dl_clf_results = {}

for model_name, builder in [("LSTM", build_lstm_classifier),
                              ("1D-CNN", build_cnn_classifier)]:
    aucs, f1s, baccs, precs, recs = [], [], [], [], []

    for fold_i, (train_end, test_start, test_end) in enumerate(folds):
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_flat[:train_end])
        X_test_sc  = scaler.transform(X_flat[test_start:test_end])

        y_train_raw = y_clf_flat[:train_end]
        y_test_raw  = y_clf_flat[test_start:test_end]

        Xs_train, ys_train = make_sequences(X_train_sc, y_train_raw, WINDOW_SIZE)
        Xs_test,  ys_test  = make_sequences(X_test_sc,  y_test_raw,  WINDOW_SIZE)

        if len(Xs_test) == 0 or len(np.unique(ys_test)) < 2:
            continue

        model = builder(len(FEATURES))
        model.fit(
            Xs_train, ys_train,
            epochs=EPOCHS,
            batch_size=BATCH,
            validation_split=0.15,
            callbacks=[early_stop],
            class_weight=class_weight_dict,
            verbose=0
        )

        y_prob = model.predict(Xs_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        if len(np.unique(ys_test)) >= 2:
            aucs.append(roc_auc_score(ys_test, y_prob))
            f1s.append(f1_score(ys_test, y_pred, zero_division=0))
            baccs.append(balanced_accuracy_score(ys_test, y_pred))
            precs.append(precision_score(ys_test, y_pred, zero_division=0))
            recs.append(recall_score(ys_test, y_pred, zero_division=0))

    results = {
        "AUC-ROC"      : (np.mean(aucs),  np.std(aucs)),
        "F1"           : (np.mean(f1s),   np.std(f1s)),
        "Bal. Accuracy": (np.mean(baccs), np.std(baccs)),
        "Precision"    : (np.mean(precs), np.std(precs)),
        "Recall"       : (np.mean(recs),  np.std(recs)),
    }
    dl_clf_results[model_name] = results
    print(f"\n  {model_name}")
    for metric, (mean, std) in results.items():
        print(f"    {metric:20s}: {mean:.3f} ± {std:.3f}")


# FIGURE 13 — Learning curves (training history)

print("\nPlotting Figure 13: Learning curves...")

fig, axes = plt.subplots(len(REG_TARGETS), 2, figsize=(14, 4 * len(REG_TARGETS)))

for row, target in enumerate(REG_TARGETS):
    for col, model_name in enumerate(["LSTM", "1D-CNN"]):
        ax = axes[row, col]
        hists = history_store[target].get(model_name, [])

        if hists:
            h = hists[-1]
            ax.plot(h["loss"],     color=COLORS[0], linewidth=1.8, label="train loss")
            ax.plot(h["val_loss"], color=COLORS[3], linewidth=1.8,
                    linestyle="--", label="val loss")
            ax.set_title(f"{model_name} → {target}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("MSE Loss", fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.4)

fig.suptitle("Figure 13 — Learning curves (last CV fold)\n"
             "Converging curves = healthy training; val loss rising = overfitting",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(FIG_DIR, "fig13_learning_curves.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 14 — Full model comparison (ML vs DL)
print("Plotting Figure 14: ML vs DL comparison...")

fig, axes = plt.subplots(1, len(REG_TARGETS), figsize=(18, 6))

for ax, target in zip(axes, REG_TARGETS):
    all_models = {}

    # ML models
    for model_name, metrics in ml_results["regression"][target].items():
        all_models[model_name] = metrics["R2"]["mean"]

    # DL models
    for model_name, metrics in dl_reg_results[target].items():
        all_models[f"{model_name} (DL)"] = metrics["R2"][0]

    names = list(all_models.keys())
    r2s   = list(all_models.values())
    order = np.argsort(r2s)[::-1]

    bar_colors = [COLORS[4] if "(DL)" in names[i] else COLORS[0]
                  for i in order]
    ax.barh([names[i] for i in order],
            [r2s[i]   for i in order],
            color=bar_colors, alpha=0.85, edgecolor="white")

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("R²", fontsize=9)
    ax.set_title(target, fontweight="bold", fontsize=9)
    ax.grid(axis="x", alpha=0.4)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=COLORS[0], label="ML models"),
                    Patch(color=COLORS[4], label="DL models (LSTM/CNN)")],
           loc="lower center", ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle("Figure 14 — Complete model comparison: ML vs DL (R², regression)\n"
             "R² > 0 means the model beats the mean-prediction baseline",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(FIG_DIR, "fig14_ml_vs_dl_comparison.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# FIGURE 15 — AUC comparison across all classifiers
print("Plotting Figure 15: Classification AUC comparison...")

all_clf = {}
for name, metrics in ml_results["classification"].items():
    all_clf[name] = metrics["AUC-ROC"]["mean"]
for name, metrics in dl_clf_results.items():
    all_clf[f"{name} (DL)"] = metrics["AUC-ROC"][0]

names = list(all_clf.keys())
aucs  = list(all_clf.values())
order = np.argsort(aucs)[::-1]

bar_colors = [COLORS[4] if "(DL)" in names[i] else COLORS[1]
              for i in order]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh([names[i] for i in order], [aucs[i] for i in order],
        color=bar_colors, alpha=0.85, edgecolor="white")
ax.axvline(0.5, color="gray", linewidth=1.2, linestyle="--", label="random (0.5)")
ax.set_xlabel("AUC-ROC (mean across CV folds)", fontsize=10)
ax.set_title("Figure 15 — AUC-ROC comparison: all classifiers\n"
             "predicting poor sleep night (walk-forward CV, k=5)",
             fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
path = os.path.join(FIG_DIR, "fig15_auc_comparison.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {path}")


# SAVING DL RESULTS
dl_out = {
    "regression": {
        target: {
            model: {
                metric: {"mean": round(vals[0], 4), "std": round(vals[1], 4)}
                for metric, vals in metrics.items()
            }
            for model, metrics in model_dict.items()
        }
        for target, model_dict in dl_reg_results.items()
    },
    "classification": {
        model: {
            metric: {"mean": round(vals[0], 4), "std": round(vals[1], 4)}
            for metric, vals in metrics.items()
        }
        for model, metrics in dl_clf_results.items()
    }
}

out_path = os.path.join(OUTPUT_DIR, "dl_results.json")
with open(out_path, "w") as f:
    json.dump(dl_out, f, indent=2)
print(f"\nDL results saved → {out_path}")


print("\n" + "=" * 60)
print("FINAL SUMMARY — All models")
print("=" * 60)

for target in REG_TARGETS:
    print(f"\n{target}:")
    all_r2 = {}
    for name, m in ml_results["regression"][target].items():
        all_r2[name] = m["R2"]["mean"]
    for name, m in dl_reg_results[target].items():
        all_r2[f"{name} (DL)"] = m["R2"][0]
    best = max(all_r2, key=all_r2.get)
    for name, r2 in sorted(all_r2.items(), key=lambda x: -x[1]):
        marker = " ← best" if name == best else ""
        print(f"  {name:30s}: R²={r2:.3f}{marker}")

print(f"\nClassification (AUC-ROC):")
all_auc = {}
for name, m in ml_results["classification"].items():
    all_auc[name] = m["AUC-ROC"]["mean"]
for name, m in dl_clf_results.items():
    all_auc[f"{name} (DL)"] = m["AUC-ROC"][0]
best = max(all_auc, key=all_auc.get)
for name, auc in sorted(all_auc.items(), key=lambda x: -x[1]):
    marker = " ← best" if name == best else ""
    print(f"  {name:30s}: AUC={auc:.3f}{marker}")