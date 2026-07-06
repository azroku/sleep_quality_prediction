# =============================================================================
# Step 5: Additional Experiments (1–6)
# EXPERIMENT 1 — Rolling / trend features + remodelling
# EXPERIMENT 2 — SOL and WASO as regression targets
# EXPERIMENT 3 — Weekly behavioural pattern analysis
# EXPERIMENT 4 — Hyperparameter tuning (GridSearchCV)
# EXPERIMENT 5 — Multivariate multi-output regression
# EXPERIMENT 6 — Anomaly detection (GMM + Isolation Forest)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model      import Ridge
from sklearn.ensemble          import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.multioutput       import MultiOutputRegressor
from sklearn.metrics           import (mean_absolute_error, mean_squared_error,
                                        r2_score, roc_auc_score, f1_score,
                                        balanced_accuracy_score)
from sklearn.mixture           import GaussianMixture
from xgboost                   import XGBRegressor, XGBClassifier
from scipy                     import stats as sp_stats

DATA_PATH   = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output\sleep_merged_clean.csv"
CONFIG_PATH = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output\project_config.json"
OUTPUT_DIR  = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output"
FIG_DIR     = r"C:\Users\sa1-b\OneDrive\Desktop\azra\mlprcda\output\figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df['sleep_date'] = pd.to_datetime(df['sleep_date'])
df = df.sort_values('sleep_date').reset_index(drop=True)

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

FEATURES    = cfg['daytime_features']
REG_TARGETS = cfg['regression_targets']
CLF_TARGET  = cfg['classification_target']
N_SPLITS    = 5
tscv        = TimeSeriesSplit(n_splits=N_SPLITS)

sns.set_theme(style='whitegrid', font_scale=1.05)
COLORS = ['#378ADD','#1D9E75','#534AB7','#E24B4A','#BA7517']
results_summary = {}   

print('='*60)
print('ADDITIONAL EXPERIMENTS')
print('='*60)


# EXPERIMENT 1 — Rolling / trend features

print('\n' + '='*60)
print('EXPERIMENT 1: Rolling / trend features')
print('='*60)

feat_no_flag = [f for f in FEATURES if f != 'possible_nonwear']

# 3-day rolling mean — "recent average"
roll3 = df[feat_no_flag].rolling(3, min_periods=2).mean()
roll3.columns = [f + '_roll3' for f in feat_no_flag]

# 7-day rolling mean — "weekly baseline"
roll7 = df[feat_no_flag].rolling(7, min_periods=4).mean()
roll7.columns = [f + '_roll7' for f in feat_no_flag]

# 3-day trend (difference) — "direction of change"
trend3 = df[feat_no_flag].diff(3)
trend3.columns = [f + '_trend3' for f in feat_no_flag]

df_enriched = pd.concat([df, roll3, roll7, trend3], axis=1)
df_enriched = df_enriched.dropna().reset_index(drop=True)

FEATURES_ENRICHED = (FEATURES +
                      list(roll3.columns) +
                      list(roll7.columns) +
                      list(trend3.columns))

X_orig = df[FEATURES].values
X_enr  = df_enriched[FEATURES_ENRICHED].values

print(f'Original feature count  : {len(FEATURES)}')
print(f'Enriched feature count  : {len(FEATURES_ENRICHED)}')
print(f'Rows after dropna       : {len(df_enriched)} (lost {len(df)-len(df_enriched)} rows)')

def cv_regression(X, y, model_pipeline, tscv):
    maes, rmses, r2s = [], [], []
    for tr, te in tscv.split(X):
        model_pipeline.fit(X[tr], y[tr])
        p = model_pipeline.predict(X[te])
        maes.append(mean_absolute_error(y[te], p))
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        r2s.append(r2_score(y[te], p))
    return np.mean(maes), np.mean(rmses), np.mean(r2s)

exp1_results = {}
print('\nR² comparison — original vs enriched features (Random Forest):')
print(f'{"Target":40s} {"Original R²":>12} {"Enriched R²":>12} {"Delta":>8}')
print('-'*76)

for target in REG_TARGETS:
    y_orig = df[target].values
    y_enr  = df_enriched[target].values

    pipe = Pipeline([('sc', StandardScaler()),
                     ('m',  RandomForestRegressor(n_estimators=200, max_depth=5,
                                                   min_samples_leaf=5, random_state=42,
                                                   n_jobs=-1))])
    _, _, r2_orig = cv_regression(X_orig, y_orig, pipe, tscv)
    _, _, r2_enr  = cv_regression(X_enr,  y_enr,  pipe, tscv)
    delta = r2_enr - r2_orig
    exp1_results[target] = {'original': round(r2_orig,4), 'enriched': round(r2_enr,4)}
    print(f'  {target:38s} {r2_orig:>12.4f} {r2_enr:>12.4f} {delta:>+8.4f}')

results_summary['exp1_rolling_features'] = exp1_results

df_enriched.to_csv(os.path.join(OUTPUT_DIR, 'sleep_enriched.csv'), index=False)
enr_cfg = cfg.copy()
enr_cfg['enriched_features'] = FEATURES_ENRICHED
with open(os.path.join(OUTPUT_DIR, 'enriched_config.json'), 'w') as f:
    json.dump(enr_cfg, f, indent=2)
print('Enriched dataset saved.')


# Figure: R² original vs enriched
fig, ax = plt.subplots(figsize=(10, 4))
x_pos  = np.arange(len(REG_TARGETS))
width  = 0.35
short_labels = ['Total Sleep', 'REM', 'Deep Sleep']
bars1 = ax.bar(x_pos - width/2,
               [exp1_results[t]['original'] for t in REG_TARGETS],
               width, label='Original features (13)', color=COLORS[0], alpha=0.8)
bars2 = ax.bar(x_pos + width/2,
               [exp1_results[t]['enriched'] for t in REG_TARGETS],
               width, label=f'Enriched features ({len(FEATURES_ENRICHED)})', color=COLORS[1], alpha=0.8)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_xticks(x_pos); ax.set_xticklabels(short_labels)
ax.set_ylabel('R² (walk-forward CV, k=5)')
ax.set_title('Figure 21 — Experiment 1: Original vs rolling/trend features\nRandom Forest regression R²',
             fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.4)
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h + 0.005,
            f'{h:.3f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig21_rolling_features.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Figure 21 saved.')


# EXPERIMENT 2 — SOL and WASO as regression targets
#   "can we predict how fragmented sleep will be?"
#   High WASO is associated with low HRV the following day (feedback loop).
#   We're testing the reverse: does low HRV TODAY predict high WASO TONIGHT?

print('\n' + '='*60)
print('EXPERIMENT 2: SOL and WASO as regression targets')
print('='*60)

X = df[FEATURES].values
sol_waso_targets = {
    'sleep_onset_latency_min': 'Sleep Onset Latency (SOL)',
    'waso_min'               : 'Wake After Sleep Onset (WASO)',
}

exp2_results = {}
print(f'\n{"Target":35s} {"MAE":>8} {"RMSE":>8} {"R²":>8}  Best model')
print('-'*75)

models_exp2 = {
    'Ridge'         : Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))]),
    'Random Forest' : Pipeline([('sc', StandardScaler()),
                                 ('m', RandomForestRegressor(n_estimators=200, max_depth=5,
                                                              min_samples_leaf=5,
                                                              random_state=42, n_jobs=-1))]),
    'XGBoost'       : Pipeline([('sc', StandardScaler()),
                                 ('m', XGBRegressor(n_estimators=200, max_depth=3,
                                                     learning_rate=0.05, subsample=0.8,
                                                     colsample_bytree=0.8, random_state=42,
                                                     verbosity=0, n_jobs=-1))]),
}

for col, label in sol_waso_targets.items():
    y = df[col].values
    best_r2, best_name, best_mae, best_rmse = -999, '', 0, 0
    for mname, pipe in models_exp2.items():
        mae, rmse, r2 = cv_regression(X, y, pipe, tscv)
        if r2 > best_r2:
            best_r2, best_name, best_mae, best_rmse = r2, mname, mae, rmse
    exp2_results[col] = {'label': label, 'best_model': best_name,
                          'MAE': round(best_mae,2), 'RMSE': round(best_rmse,2),
                          'R2': round(best_r2,4)}
    print(f'  {label:33s} {best_mae:>8.1f} {best_rmse:>8.1f} {best_r2:>8.4f}  {best_name}')

    # HRV correlation with this target
    r, p = sp_stats.pearsonr(df['Avg. HRV(ms)'], y)
    print(f'    → HRV correlation: r={r:.3f}, p={p:.3f}')

results_summary['exp2_sol_waso'] = exp2_results

# Figure: actual vs predicted for best model on WASO
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (col, label) in zip(axes, sol_waso_targets.items()):
    y = df[col].values
    pipe = Pipeline([('sc', StandardScaler()),
                     ('m', RandomForestRegressor(n_estimators=200, max_depth=5,
                                                  min_samples_leaf=5, random_state=42))])
    splits = list(tscv.split(X))
    tr, te = splits[-1]
    pipe.fit(X[tr], y[tr])
    y_pred = pipe.predict(X[te])
    y_true = y[te]

    ax.scatter(y_true, y_pred, alpha=0.6, color=COLORS[2], s=30, edgecolors='none')
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.2, label='perfect prediction')
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ax.set_xlabel(f'Actual {label} (min)', fontsize=9)
    ax.set_ylabel(f'Predicted {label} (min)', fontsize=9)
    ax.set_title(f'{label}\nR²={r2:.3f}, MAE={mae:.1f} min', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

fig.suptitle('Figure 22 — Experiment 2: Predicting SOL and WASO\nActual vs Predicted (Random Forest, last CV fold)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig22_sol_waso.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Figure 22 saved.')


# EXPERIMENT 3 — Weekly behavioural pattern analysis
#   Day encoding: 0=Monday, 1=Tuesday, ..., 6=Sunday

print('\n' + '='*60)
print('EXPERIMENT 3: Weekly behavioural patterns')
print('='*60)

df['dayofweek'] = df['sleep_date'].dt.dayofweek
day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

weekly_cols = {
    'Time Asleep(min)'              : 'Total Sleep (min)',
    'Sleep Stages - REM(min)'       : 'REM (min)',
    'Sleep Stages - Deep Sleep(min)': 'Deep Sleep (min)',
    'Sleep Time Ratio(%)'           : 'Sleep Efficiency (%)',
    'Avg. HRV(ms)'                  : 'Avg HRV (ms)',
    'Steps'                         : 'Steps',
}

print('\nMean ± std per day of week:')
for col, label in weekly_cols.items():
    stats_dow = df.groupby('dayofweek')[col].agg(['mean','std'])
    best_day  = day_names[stats_dow['mean'].idxmax()]
    worst_day = day_names[stats_dow['mean'].idxmin()]
    print(f'  {label:28s}: best={best_day}, worst={worst_day}')

print('\nKruskal-Wallis test (any day-of-week effect?):')
kw_results = {}
for col, label in weekly_cols.items():
    groups = [df[df['dayofweek']==d][col].dropna().values for d in range(7)]
    groups = [g for g in groups if len(g) >= 3]
    stat, p = sp_stats.kruskal(*groups)
    sig = '* significant' if p < 0.05 else 'not significant'
    kw_results[label] = {'H': round(stat,3), 'p': round(p,4)}
    print(f'  {label:28s}: H={stat:.2f}, p={p:.3f}  {sig}')

results_summary['exp3_weekly_patterns'] = kw_results

# Figure: heatmap + box plots
fig = plt.figure(figsize=(18, 10))
gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

plot_vars = [
    ('Sleep Time Ratio(%)',           'Sleep Efficiency (%)'),
    ('Sleep Stages - REM(min)',       'REM (min)'),
    ('Avg. HRV(ms)',                  'Avg HRV (ms)'),
]
for i, (col, label) in enumerate(plot_vars):
    ax = fig.add_subplot(gs[0, i])
    data_by_day = [df[df['dayofweek']==d][col].dropna().values for d in range(7)]
    bp = ax.boxplot(data_by_day, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(linewidth=1.2))
    for patch, color in zip(bp['boxes'], plt.cm.RdYlGn(np.linspace(0.2, 0.8, 7))):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax.set_xticklabels(day_names, fontsize=9)
    ax.set_title(label, fontweight='bold', fontsize=10)
    ax.set_xlabel('Day of week', fontsize=8)
    ax.grid(axis='y', alpha=0.4)
    # Add KW p-value
    p_kw = kw_results[label]['p']
    ax.text(0.98, 0.97, f'KW p={p_kw:.3f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='gray', alpha=0.8))

ax_heat = fig.add_subplot(gs[1, :])
heat_data = pd.DataFrame({
    label: df.groupby('dayofweek')[col].mean()
    for col, label in weekly_cols.items()
}).T

heat_norm = heat_data.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
heat_norm.columns = day_names

sns.heatmap(heat_norm, annot=heat_data.round(1), fmt='g',
            cmap='RdYlGn', center=0, linewidths=0.5, ax=ax_heat,
            cbar_kws={'label': 'Z-score (relative to weekly mean)'})
ax_heat.set_title('Mean values by day of week (colour = relative to weekly mean)',
                   fontweight='bold', fontsize=10)
ax_heat.set_xlabel('Day of week', fontsize=9)

fig.suptitle('Figure 23 — Experiment 3: Weekly behavioural and sleep patterns',
             fontweight='bold', fontsize=12)
plt.savefig(os.path.join(FIG_DIR, 'fig23_weekly_patterns.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Figure 23 saved.')


# EXPERIMENT 4 — Hyperparameter tuning (GridSearchCV)

print('\n' + '='*60)
print('EXPERIMENT 4: Hyperparameter tuning (GridSearchCV)')
print('='*60)
print('This may take a few minutes...')

X = df[FEATURES].values
# Use 3-fold inner CV to keep total runtime reasonable
inner_cv = TimeSeriesSplit(n_splits=3)

param_grids = {
    'Random Forest': {
        'pipeline': Pipeline([('sc', StandardScaler()),
                               ('m', RandomForestRegressor(random_state=42, n_jobs=-1))]),
        'params': {
            'm__n_estimators'  : [100, 200, 300],
            'm__max_depth'     : [3, 5, 7],
            'm__min_samples_leaf': [3, 5, 8],
        }
    },
    'XGBoost': {
        'pipeline': Pipeline([('sc', StandardScaler()),
                               ('m', XGBRegressor(random_state=42, verbosity=0, n_jobs=-1))]),
        'params': {
            'm__n_estimators'  : [100, 200],
            'm__max_depth'     : [2, 3, 4],
            'm__learning_rate' : [0.01, 0.05, 0.1],
            'm__subsample'     : [0.7, 0.9],
        }
    }
}

exp4_results = {}
print(f'\n{"Target":35s} {"Model":16s} {"Default R²":>11} {"Tuned R²":>10} {"Best params"}')
print('-'*100)

default_r2 = {
    'Time Asleep(min)'              : {'Random Forest': -0.222, 'XGBoost': -0.518},
    'Sleep Stages - REM(min)'       : {'Random Forest': -0.275, 'XGBoost': -0.440},
    'Sleep Stages - Deep Sleep(min)': {'Random Forest': -0.138, 'XGBoost': -0.338},
}

for target in REG_TARGETS:
    y = df[target].values
    exp4_results[target] = {}

    for mname, spec in param_grids.items():
        r2s_tuned = []
        best_params_list = []

        for tr, te in tscv.split(X):
            gs = GridSearchCV(
                spec['pipeline'], spec['params'],
                cv=inner_cv, scoring='r2',
                n_jobs=-1, refit=True
            )
            gs.fit(X[tr], y[tr])
            y_pred = gs.predict(X[te])
            r2s_tuned.append(r2_score(y[te], y_pred))
            best_params_list.append(gs.best_params_)

        r2_tuned   = np.mean(r2s_tuned)
        r2_default = default_r2[target][mname]
        delta      = r2_tuned - r2_default

        from collections import Counter
        best_p = best_params_list[-1]   
        exp4_results[target][mname] = {
            'default_r2': r2_default,
            'tuned_r2'  : round(r2_tuned, 4),
            'delta'     : round(delta, 4),
            'best_params': {k.replace('m__',''): v for k,v in best_p.items()}
        }
        params_str = ', '.join([f'{k.replace("m__","")}={v}'
                                  for k,v in best_p.items()])[:50]
        print(f'  {target:33s} {mname:16s} {r2_default:>11.4f} {r2_tuned:>10.4f} {params_str}')

results_summary['exp4_hyperparameter_tuning'] = exp4_results

# Figure: default vs tuned R²
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, target in zip(axes, REG_TARGETS):
    models_list = list(param_grids.keys())
    x_pos  = np.arange(len(models_list))
    width  = 0.35
    defaults = [exp4_results[target][m]['default_r2'] for m in models_list]
    tuned    = [exp4_results[target][m]['tuned_r2']   for m in models_list]

    ax.bar(x_pos-width/2, defaults, width, label='Default', color=COLORS[0], alpha=0.8)
    ax.bar(x_pos+width/2, tuned,    width, label='Tuned',   color=COLORS[1], alpha=0.8)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xticks(x_pos); ax.set_xticklabels(models_list, fontsize=9)
    ax.set_ylabel('R²'); ax.set_title(target.split('(')[0], fontweight='bold', fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.4)

fig.suptitle('Figure 24 — Experiment 4: Default vs tuned hyperparameters\n(nested CV: outer=TimeSeriesSplit k=5, inner=3-fold)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig24_hyperparameter_tuning.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Figure 24 saved.')


# EXPERIMENT 5 — Multivariate multi-output regression

print('\n' + '='*60)
print('EXPERIMENT 5: Multi-output regression')
print('='*60)

X = df[FEATURES].values
Y = df[REG_TARGETS].values   # shape (158, 3)

mo_rf = Pipeline([
    ('sc', StandardScaler()),
    ('m',  MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=5,
                               min_samples_leaf=5, random_state=42, n_jobs=-1),
        n_jobs=-1
    ))
])

mo_r2s = {t: [] for t in REG_TARGETS}
for tr, te in tscv.split(X):
    mo_rf.fit(X[tr], Y[tr])
    Y_pred = mo_rf.predict(X[te])
    for i, target in enumerate(REG_TARGETS):
        mo_r2s[target].append(r2_score(Y[te, i], Y_pred[:, i]))

# Single-output RF for comparison
so_r2s = {}
for i, target in enumerate(REG_TARGETS):
    y = df[target].values
    pipe = Pipeline([('sc', StandardScaler()),
                     ('m', RandomForestRegressor(n_estimators=200, max_depth=5,
                                                  min_samples_leaf=5, random_state=42))])
    _, _, r2 = cv_regression(X, y, pipe, tscv)
    so_r2s[target] = r2

exp5_results = {}
print(f'\n{"Target":35s} {"Single-output R²":>17} {"Multi-output R²":>16} {"Delta":>8}')
print('-'*80)
for target in REG_TARGETS:
    r2_so = so_r2s[target]
    r2_mo = np.mean(mo_r2s[target])
    delta = r2_mo - r2_so
    exp5_results[target] = {'single': round(r2_so,4), 'multi': round(r2_mo,4)}
    print(f'  {target:33s} {r2_so:>17.4f} {r2_mo:>16.4f} {delta:>+8.4f}')

results_summary['exp5_multioutput'] = exp5_results

# Target correlation matrix (justifies multi-output approach)
target_corr = df[REG_TARGETS].corr()
print('\nTarget correlation matrix (why multi-output makes sense):')
print(target_corr.round(3).to_string())

# Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar comparison
ax = axes[0]
x_pos = np.arange(len(REG_TARGETS))
width = 0.35
short = ['Total Sleep', 'REM', 'Deep Sleep']
ax.bar(x_pos-width/2, [so_r2s[t]             for t in REG_TARGETS],
       width, label='Single-output', color=COLORS[0], alpha=0.8)
ax.bar(x_pos+width/2, [np.mean(mo_r2s[t])    for t in REG_TARGETS],
       width, label='Multi-output',  color=COLORS[1], alpha=0.8)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_xticks(x_pos); ax.set_xticklabels(short)
ax.set_ylabel('R²'); ax.set_title('Single vs Multi-output RF', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.4)

# Target correlation heatmap
ax = axes[1]
short_targets = ['Total Sleep', 'REM', 'Deep Sleep']
sns.heatmap(target_corr, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=short_targets, yticklabels=short_targets,
            ax=ax, cbar_kws={'shrink':0.8})
ax.set_title('Target correlation\n(justifies multi-output approach)', fontweight='bold')

fig.suptitle('Figure 25 — Experiment 5: Single-output vs multi-output regression',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig25_multioutput.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Figure 25 saved.')


# EXPERIMENT 6 — Anomaly detection (GMM + Isolation Forest)
#
# TWO MODELS:
#   GMM (Gaussian Mixture Model): fits K Gaussian distributions to the data.
#   Nights with low probability under the fitted model = anomalies.
#
#   Isolation Forest: isolates anomalies by randomly partitioning the feature
#   space. Points that are easy to isolate (need few splits) = anomalies.

print('\n' + '='*60)
print('EXPERIMENT 6: Anomaly detection (GMM + Isolation Forest)')
print('='*60)

sleep_features_anomaly = [
    'Time Asleep(min)', 'Sleep Stages - REM(min)',
    'Sleep Stages - Deep Sleep(min)', 'waso_min',
    'Sleep Time Ratio(%)', 'sleep_onset_latency_min'
]
X_sleep = df[sleep_features_anomaly].values
scaler  = StandardScaler()
X_sleep_sc = scaler.fit_transform(X_sleep)

#GMM 
# Select optimal number of components using BIC
# Lower BIC = better model, prevents from overfitting with too many k.
bic_scores = []
n_components_range = range(1, 8)
for n in n_components_range:
    gmm = GaussianMixture(n_components=n, covariance_type='full',
                           random_state=42, n_init=3)
    gmm.fit(X_sleep_sc)
    bic_scores.append(gmm.bic(X_sleep_sc))

best_n = n_components_range[np.argmin(bic_scores)]
print(f'\nGMM optimal components (BIC): {best_n}')

gmm_final = GaussianMixture(n_components=best_n, covariance_type='full',
                              random_state=42, n_init=5)
gmm_final.fit(X_sleep_sc)

# Log-likelihood score per night == low score = anomalous
gmm_scores = gmm_final.score_samples(X_sleep_sc)
gmm_threshold = np.percentile(gmm_scores, 10)   
gmm_anomalies = (gmm_scores < gmm_threshold).astype(int)

print(f'GMM anomalies detected (bottom 10%): {gmm_anomalies.sum()} nights')

#Isolation Forest 
iso = IsolationForest(
    n_estimators=200,
    contamination=0.10,   
    random_state=42,
    n_jobs=-1
)
iso.fit(X_sleep_sc)
iso_labels = (iso.predict(X_sleep_sc) == -1).astype(int)   
iso_scores = -iso.score_samples(X_sleep_sc)   # higher = more anomalous

print(f'Isolation Forest anomalies detected: {iso_labels.sum()} nights')

agreement = (gmm_anomalies == iso_labels).mean() * 100
both_anomaly = ((gmm_anomalies == 1) & (iso_labels == 1)).sum()
print(f'Agreement between GMM and IF: {agreement:.1f}%')
print(f'Both methods flag as anomaly: {both_anomaly} nights')

poor_sleep = df[CLF_TARGET].values
print(f'\nOverlap with poor_sleep label:')
print(f'  GMM anomalies that are also poor_sleep: '
      f'{((gmm_anomalies==1) & (poor_sleep==1)).sum()} / {gmm_anomalies.sum()}')
print(f'  IF  anomalies that are also poor_sleep: '
      f'{((iso_labels==1)   & (poor_sleep==1)).sum()} / {iso_labels.sum()}')

if len(np.unique(poor_sleep)) == 2:
    auc_gmm = roc_auc_score(poor_sleep, -gmm_scores)
    auc_iso = roc_auc_score(poor_sleep, iso_scores)
    print(f'\nAUC-ROC (anomaly score vs poor_sleep label):')
    print(f'  GMM: {auc_gmm:.3f}')
    print(f'  IF : {auc_iso:.3f}')
    results_summary['exp6_anomaly_detection'] = {
        'gmm_n_components': int(best_n),
        'gmm_anomalies'   : int(gmm_anomalies.sum()),
        'iso_anomalies'   : int(iso_labels.sum()),
        'agreement_pct'   : round(agreement, 1),
        'both_anomaly'    : int(both_anomaly),
        'auc_gmm'         : round(auc_gmm, 4),
        'auc_iso'         : round(auc_iso, 4)
    }

# Figure: 3 panels
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# BIC curve
ax = axes[0]
ax.plot(list(n_components_range), bic_scores, marker='o', color=COLORS[2],
        linewidth=2, markersize=7)
ax.axvline(best_n, color=COLORS[3], linewidth=1.5, linestyle='--',
           label=f'Optimal k={best_n}')
ax.set_xlabel('Number of GMM components', fontsize=9)
ax.set_ylabel('BIC score (lower = better)', fontsize=9)
ax.set_title('GMM component selection\n(BIC criterion)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.4)

# Scatter: GMM score vs IF score, coloured by poor_sleep
ax = axes[1]
colors_pts = np.where(poor_sleep == 1, COLORS[3], COLORS[0])
ax.scatter(gmm_scores, iso_scores, c=colors_pts, alpha=0.55, s=30, edgecolors='none')
ax.axvline(gmm_threshold, color=COLORS[3], linewidth=1.2,
           linestyle='--', label=f'GMM threshold (10th pct)')
ax.set_xlabel('GMM log-likelihood (lower = more anomalous)', fontsize=9)
ax.set_ylabel('Isolation Forest anomaly score\n(higher = more anomalous)', fontsize=9)
ax.set_title('GMM vs Isolation Forest\nanomaly scores', fontweight='bold')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=COLORS[3], label='poor sleep night'),
                    Patch(color=COLORS[0], label='good sleep night'),
                    plt.Line2D([0],[0], color=COLORS[3], linestyle='--',
                               label='GMM threshold')],
          fontsize=8)
ax.grid(alpha=0.4)

# Time series of anomaly scores
ax = axes[2]
ax2_twin = ax.twinx()
ax.plot(df['sleep_date'], gmm_scores, color=COLORS[2], alpha=0.5,
        linewidth=0.8, label='GMM score')
ax2_twin.plot(df['sleep_date'], iso_scores, color=COLORS[4], alpha=0.5,
              linewidth=0.8, label='IF score')
# Mark detected anomalies
both_idx = np.where((gmm_anomalies == 1) & (iso_labels == 1))[0]
ax.scatter(df['sleep_date'].iloc[both_idx], gmm_scores[both_idx],
           color=COLORS[3], zorder=5, s=50, label='Both methods: anomaly')
ax.set_ylabel('GMM log-likelihood', fontsize=8, color=COLORS[2])
ax2_twin.set_ylabel('IF score', fontsize=8, color=COLORS[4])
ax.set_title('Anomaly scores over time\n(red = flagged by both methods)', fontweight='bold')
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.legend(fontsize=7, loc='upper left')
ax.grid(alpha=0.3)

fig.suptitle('Figure 26 — Experiment 6: Anomaly detection (GMM + Isolation Forest)\n'
             'Identifying unusual sleep nights without supervision',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig26_anomaly_detection.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Figure 26 saved.')



out_path = os.path.join(OUTPUT_DIR, 'experiment_results.json')
with open(out_path, 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f'\nAll experiment results saved → {out_path}')

print('\n' + '='*60)
print('SUMMARY OF ALL EXPERIMENTS')
print('='*60)

print('\nExp 1 — Rolling features (R² improvement):')
for t, v in exp1_results.items():
    d = v['enriched'] - v['original']
    print(f'  {t.split("(")[0]:35s}: {v["original"]:+.4f} → {v["enriched"]:+.4f}  (Δ{d:+.4f})')

print('\nExp 2 — SOL/WASO prediction:')
for col, v in exp2_results.items():
    print(f'  {v["label"]:30s}: R²={v["R2"]:.4f}, MAE={v["MAE"]:.1f} min ({v["best_model"]})')

print('\nExp 3 — Weekly patterns (Kruskal-Wallis significant?):')
for label, v in kw_results.items():
    sig = 'YES *' if v['p'] < 0.05 else 'no'
    print(f'  {label:28s}: p={v["p"]:.3f}  {sig}')

print('\nExp 4 — Hyperparameter tuning (best delta R²):')
for t in REG_TARGETS:
    for m in ['Random Forest', 'XGBoost']:
        d = exp4_results[t][m]['delta']
        print(f'  {t.split("(")[0]:25s} {m:16s}: Δ={d:+.4f}')

print('\nExp 5 — Multi-output regression:')
for t, v in exp5_results.items():
    d = v['multi'] - v['single']
    print(f'  {t.split("(")[0]:35s}: single={v["single"]:+.4f}, multi={v["multi"]:+.4f}  (Δ{d:+.4f})')

print('\nExp 6 — Anomaly detection:')
print(f'  GMM AUC vs poor_sleep label: {results_summary["exp6_anomaly_detection"]["auc_gmm"]:.3f}')
print(f'  IF  AUC vs poor_sleep label: {results_summary["exp6_anomaly_detection"]["auc_iso"]:.3f}')
print(f'  Nights flagged by both: {results_summary["exp6_anomaly_detection"]["both_anomaly"]}')
