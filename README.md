# Sleep Quality Prediction from Daytime Wearable Metrics

N-of-1 machine learning study predicting next-night sleep quality from daytime physiological and behavioural data collected via a consumer wrist-worn device (Didiconn) over 6 months (158 nights, from November 2025 to May 2026).

**Courses:** Machine Learning & Pattern Recognition / Computational Data Analytics  
**Institution:** Faculty of Electrical Engineering, University of Sarajevo, 2025/26  
**Author:** Azra Kurić

---

## Research Questions

1. Can next-night sleep quality be predicted from daytime wearable metrics?
2. Which physiological indicators are early predictors of poor sleep?
3. Does low daytime HRV precede fragmented sleep?

---

## Setup

Python 3.12+. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

Place the three CSV files exported from the Didiconn device into the `data/` folder:

```
data/
├── Sleep-Didiconn-2025-11-01-2026-05-04.csv
├── Vital-Signs-Didiconn-2025-11-01-2026-05-04.csv
└── Activity-Didiconn-2025-11-01-2026-05-04.csv
```

---

## Usage

Run scripts in order. Each script reads from `output/` and saves results back to `output/`.

```bash
python src/01_data_loading_and_cleaning.py
python src/02_eda.py
python src/03a_ml.py
python src/03b_dl.py
python src/04_shap_xai.py
python src/05_exp.py
```

---

## Project Structure

```
sleep-quality-prediction/
├── src/
│   ├── 01_data_loading_and_cleaning.py   Data loading, cleaning, t-1 lag alignment
│   ├── 02_eda.py                         Exploratory data analysis (15 figures)
│   ├── 03a_ml.py                         ML regression and classification models
│   ├── 03b_dl.py                         Deep learning models (LSTM, 1D-CNN)
│   ├── 04_shap_xai.py                    SHAP explainability analysis
│   └── 05_exp.py                         6 additional experiments
├── data/                                 Raw CSV files from wearable device
├── output/                               Generated figures and result JSONs
├── report/
│   └── KuricAzra_Report.tex              LaTeX paper (Nature Scientific Reports format)
├── poster/
│   └── KuricAzra_Poster.pptx             A3 conference poster
├── requirements.txt
└── README.md
```

---

## Methods

**Preprocessing:** t-1 lag/delay alignment (daytime features paired with following night's sleep), nap deduplication, HRV range feature engineering, data-driven poor-sleep threshold (25th percentile, ≤82.25% efficiency).

**Validation:** Walk-forward cross-validation (TimeSeriesSplit, k=5) - random k-fold excluded to prevent temporal leakage.

**Models:**
- Regression: Linear Regression, Ridge, Lasso, Random Forest, XGBoost
- Classification: Logistic Regression, SVM (RBF), Random Forest, XGBoost
- Deep learning: LSTM, 1D-CNN (7-day sliding window, N=151 sequences)
- Anomaly detection: GMM (BIC component selection), Isolation Forest

**XAI:** TreeSHAP - global feature importance, HRV dependence plots, local waterfall explanations (CLIX-M compliant).

**Additional experiments:** Rolling/trend features (49-dim), SOL/WASO as targets, weekly pattern analysis, nested GridSearchCV hyperparameter tuning, multi-output regression.

---

## Key Results

| Task | Best model | Metric |
|------|------------|--------|
| Total sleep regression | XGBoost (tuned) | R² = −0.179 |
| Deep sleep regression | RF + rolling features | R² = −0.043 |
| REM regression | Random Forest | R² = −0.275 |
| Classification (poor sleep) | SVM RBF | AUC = 0.622 |
| **Anomaly detection** | **Isolation Forest** | **AUC = 0.728** |

Unsupervised anomaly detection outperformed all supervised classifiers. SHAP identified caloric expenditure, maximum HR, and HRV range as the most influential predictors. Rolling 3-day/7-day physiological trend features provided the largest single improvement in deep sleep prediction (ΔR² = +0.095).

---

## Reporting Standards

- [TRIPOD+AI](https://www.equator-network.org/reporting-guidelines/tripod-statement/) — clinical prediction model reporting
- [CLIX-M](https://www.equator-network.org/reporting-guidelines/clinician-informed-explainable-artificial-intelligence-evaluation-checklist-with-metrics-clix-m-for-ai-powered-clinical-decision-support-systems/) — XAI component reporting
