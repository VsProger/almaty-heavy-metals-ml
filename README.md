# Heavy Metal Air Pollution Forecasting with Explainable ML

Machine learning pipeline for predicting atmospheric heavy-metal concentrations in Almaty, Kazakhstan, from 15 years of state monitoring data (2006–2020), with SHAP/LIME explainability, deep-learning baselines, and formal structural-break testing.

This repository contains the modelling code and result artefacts of an ongoing dissertation study. The underlying raw observations belong to RSE *Kazhydromet* and are not redistributed here — see [Data](#data).

---

## Overview

Almaty sits in a mountain basin that traps pollutants: frequent temperature inversions and poor ventilation make heavy-metal accumulation a recurring public-health problem. The national monitoring network measures six metals — Cd, Pb, As, Cr, Cu, Ni — at two fixed stations, but the records exist only as 32 heterogeneous Excel workbooks spanning 15 years, with inconsistent station naming, five distinct classes of date typos, censored values below the detection limit, and duplicated reporting periods.

The project answers three questions:

1. **Can decadal heavy-metal concentrations be predicted from their own recent history and co-pollutants?** Lead (Pb) is the primary target, copper (Cu) a secondary one.
2. **Which features actually drive those predictions?** Regulatory use requires an auditable model, not a black box — hence SHAP and LIME rather than accuracy alone.
3. **Does the 9.5× historical decline in Pb (following Kazakhstan's 2003 leaded-petrol ban) break the modelling assumptions?** A non-stationary series can make train and test sets belong to different pollution regimes.

The benchmark methodology is adapted from Chen et al. (2025), *Transparent and reliable construction cost prediction using advanced machine learning and explainable AI* (Engineering Science and Technology, 70, 102159). Extending it to recurrent networks and to formal structural-break testing is the contribution of this work.

## Technical Approach

**Data engineering.** A single script reconstructs the modelling table from the raw workbooks: sheet-type detection, station-name normalisation (`ПНЗ №1` / `ПНЗ№1` / `№1` → `1`), a decade-date parser tolerant to five observed typo classes, censoring flags for below-detection-limit readings, physically-impossible-outlier filtering, and deduplication that keeps the most complete row per `(station, start_date, end_date)`. Feature construction adds cyclic seasonality (`sin`/`cos` of month and day-of-year), per-metal lags (`lag1`, `lag2`, 3-period rolling mean), and MPC (maximum permissible concentration) ratio and exceedance columns.

**Model benchmark.** Ten regressors — Ridge, Lasso, Elastic Net, KNN, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, CatBoost, HistGradientBoosting — are tuned by grid search and evaluated on two feature sets: **A** (metals + temporal features, n ≈ 921) and **B** (A plus meteorology, n ≈ 203, limited by the shorter climate record). Metrics: R², RMSE, MSE, MAE and MBE on train and test, plus log-scale cross-validated RMSE.

**Explainability.** SHAP values are computed for the top-4 models, with the explainer chosen per model family — `TreeExplainer` for Extra Trees and CatBoost, `LinearExplainer` for Ridge, `KernelExplainer` for KNN — producing beeswarm plots, importance rankings and a pairwise interaction heatmap. LIME supplies local explanations for four diagnostic cases per feature set: a typical prediction, the maximum predicted value, the maximum observed value and the largest-error case.

**Deep learning.** LSTM, GRU and a two-layer stacked LSTM are trained per station on 6-decade (~2-month) windows and compared against naive and window-mean baselines.

**Structural analysis.** A Chow test (1960) probes candidate break years 2008–2013; models are then retrained on several post-break segments to test whether restricting training to a "stable" regime helps.

**Uncertainty.** Gradient-boosted quantile regression at Q ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95} yields prediction intervals, which are checked for empirical calibration and mapped onto a regulatory decision rule (`Q10 > MPC` = confident exceedance, `Q90 < MPC` = confident safe).

All experiments use `random_state = 42`.

## Technologies

Python 3.9+ · pandas · NumPy · SciPy · scikit-learn · XGBoost · CatBoost · TensorFlow/Keras · SHAP · LIME · Matplotlib

## Methodology / Pipeline

```
32 Kazhydromet Excel workbooks (2006–2020)
        │
        ▼
clean_dataset.py ──────────────► ml_ready_dataset.csv  (~924 rows × 89 cols)
        │                        parsing · normalisation · censoring flags
        │                        deduplication · lags · cyclic features · MPC columns
        ├──► eda_analysis.py ──────────────► distributions, trends, seasonality, correlations
        │
        ├──► Stage 1: benchmark_10_models.py ──► 10 models × 2 feature sets, tuned
        │              benchmark_cu_target.py ──► same benchmark with Cu as target
        │                    │
        │                    └── best_params_*.json ──┐
        │                                             │
        ├──► Stage 2: shap_analysis.py ◄──────────────┘  global + interaction explanations
        │              lime_analysis.py                  local, per-case explanations
        │
        ├──► Stage 3: lstm_gru_analysis.py ──► LSTM / GRU / Stacked-LSTM vs. baselines
        │
        └──► Refinements:
                improved_pb_v1.py ──► stacking ensemble · Cu_lag1 feature · MPC classification
                improved_pb_v2.py ──► quantile regression + calibrated prediction intervals
                improved_pb_v3.py ──► Chow test + stable-period retraining
```

## Results

All figures below are test-set values reproduced from the CSV/JSON artefacts in this repository.

### Model benchmark — Pb, feature set A (no climate)

| Model | Test R² | RMSE | MAE |
|---|---|---|---|
| **Extra Trees** | **0.704** | 0.0626 | 0.0316 |
| CatBoost | 0.691 | 0.0640 | 0.0321 |
| HistGradientBoosting | 0.688 | 0.0643 | 0.0338 |
| Gradient Boosting | 0.642 | 0.0688 | 0.0338 |
| XGBoost | 0.638 | 0.0693 | 0.0336 |
| AdaBoost | 0.593 | 0.0734 | 0.0391 |
| Ridge | 0.577 | 0.0748 | 0.0408 |
| Elastic Net | 0.577 | 0.0748 | 0.0407 |
| Lasso | 0.500 | 0.0813 | 0.0435 |
| KNN | 0.403 | 0.0889 | 0.0446 |

Feature set B (with meteorology) scores lower across the board — best Extra Trees R² = 0.550 — because adding climate variables shrinks the usable sample from ~921 to ~203 rows. **More features did not compensate for less data.**

With Cu as the target, ceiling performance drops: best Gradient Boosting R² = 0.601 vs. 0.642 for the same model on Pb.

### Feature attribution (SHAP, Extra Trees, set A)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `Pb_lag1` | 0.0187 |
| 2 | `Cu` | 0.0186 |
| 3 | `Cd` | 0.0069 |
| 4 | `Pb_roll3_mean` | 0.0064 |
| 5 | `post` (station) | 0.0061 |

Copper ranks as high as lead's own previous measurement — consistent with a shared traffic/industrial emission source. That result directly motivated the next experiment.

### Refinements

| Experiment | Features | R² | RMSE |
|---|---|---|---|
| Baseline: Extra Trees | 12 | 0.707 | 0.0623 |
| Stacking (Ridge meta-model over 4 bases) | 12 | 0.691 | 0.0639 |
| **Extra Trees + `Cu_lag1`** | **13** | **0.731** | **0.0597** |
| Stacking + `Cu_lag1` | 13 | 0.713 | 0.0616 |

Adding the single SHAP-motivated feature `Cu_lag1` gained +0.024 R² — more than the stacking ensemble, which slightly *hurt* performance.

### MPC-exceedance classification (Pb > 0.3 µg/m³)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| CatBoost | 0.964 | 0.571 | 0.667 | 0.615 | 0.955 |
| XGBoost | 0.957 | 0.500 | 0.833 | 0.625 | 0.941 |
| Gradient Boosting | 0.960 | 0.538 | 0.583 | 0.560 | 0.947 |
| Random Forest | 0.964 | 0.600 | 0.500 | 0.545 | 0.953 |

Only 12 of 277 test observations exceed the MPC. Accuracy is therefore uninformative; ROC-AUC ≈ 0.95 with F1 ≈ 0.6 is the honest reading — the models rank risk well but their positive-class precision is limited by how few exceedances exist.

### Prediction intervals (quantile regression)

Median (Q50) prediction: R² = 0.672, RMSE = 0.0659.

| Interval | Expected coverage | Empirical coverage | Mean width (µg/m³) |
|---|---|---|---|
| 90% CI | 0.900 | 0.801 | 0.106 |
| 80% CI | 0.800 | 0.682 | 0.074 |
| 50% CI | 0.500 | 0.419 | 0.038 |

All intervals are **under-covering by 8–12 percentage points** — the model is overconfident, and the README states this rather than reporting the median R² alone. Applied to the regulatory rule, `Q90 < MPC` flagged 264 of 277 cases as confidently safe, of which 262 truly were; no case reached `Q10 > MPC`.

### Recurrent networks vs. classical ensembles — a negative result

| Model | Params | Test R² | RMSE |
|---|---|---|---|
| WindowMean baseline | 0 | −0.242 | 0.1145 |
| Stacked-LSTM | 31,905 | −0.400 | 0.1215 |
| GRU | 16,513 | −0.992 | 0.1449 |
| LSTM | 21,057 | −1.322 | 0.1565 |
| Naive (last value) | 0 | −1.390 | 0.1587 |

**Every recurrent architecture failed to beat a trivial window-mean baseline, and all scored negative R².** The series is non-stationary, sparsely sampled (decadal), weakly autocorrelated — `Pb_lag1` alone yields only R² ≈ 0.11 — and dominated by transport-driven noise spikes. Under these conditions the networks collapse toward predicting a near-constant. This is reported as a finding about the data regime, not hidden as a failed experiment; the same conclusion appears in the atmospheric-monitoring literature (e.g. Cabaneros et al., 2019).

### Structural break

The Chow test rejects parameter stability most strongly at **2011** (F = 23.36, p = 1.4 × 10⁻¹⁴), with all candidate years 2008–2013 significant at p < 0.005.

Retraining on post-break segments nevertheless did **not** improve accuracy — the full 2006–2020 period remained best (R² = 0.731) against 0.706 from 2010, 0.588 from 2011 and 0.325 from 2012. The break is statistically real, but the sample loss from discarding early years outweighs the gain in homogeneity.

## Data

The raw inputs are RSE *Kazhydromet* monitoring records (two Almaty stations, 2006–2020: 924 decadal metal measurements and 7,813 synoptic meteorological observations). **They are not included in this repository** and neither is the derived modelling table, because redistribution rights have not been established.

`clean_dataset.py` documents the full transformation from the original workbooks, so the pipeline is reproducible by anyone holding the same source data. Scripts expect `data/ml_ready_dataset.csv`; the benchmark stage additionally writes `benchmark_results/sample_{A,B}_*.csv`, which the SHAP and LIME stages read.

## How to Run

```bash
git clone https://github.com/VsProger/almaty-heavy-metals-ml.git
cd almaty-heavy-metals-ml

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the source workbooks in `data/datasets/`, then run the stages in order — later stages consume earlier artefacts:

```bash
python clean_dataset.py        # data/datasets/*.xls* -> data/ml_ready_dataset.csv
python eda_analysis.py         # -> eda_results/

python benchmark_10_models.py  # -> benchmark_results/  (incl. best_params_*.json)
python benchmark_cu_target.py  # -> benchmark_cu_results/

python shap_analysis.py        # needs benchmark_results/ -> shap_results/
python lime_analysis.py        # needs benchmark_results/ -> lime_results/

python lstm_gru_analysis.py    # -> lstm_results/
python improved_pb_v1.py       # -> improved_v1_results/
python improved_pb_v2.py       # -> improved_v2_results/
python improved_pb_v3.py       # -> improved_v3_results/
```

Runtime is dominated by grid search and the KNN `KernelExplainer`; the full benchmark takes a few minutes on CPU. TensorFlow is needed only for `lstm_gru_analysis.py`.

## Project Structure

```
├── clean_dataset.py            Raw Excel -> modelling table (parsing, dedup, feature build)
├── eda_analysis.py             Distributions, time series, trends, seasonality, correlations
├── benchmark_10_models.py      Stage 1 — 10 tuned regressors, Pb target, feature sets A/B
├── benchmark_cu_target.py      Stage 1 — same benchmark with Cu as target
├── shap_analysis.py            Stage 2 — global attribution + interaction heatmap
├── lime_analysis.py            Stage 2 — local explanations for 4 diagnostic cases
├── lstm_gru_analysis.py        Stage 3 — LSTM / GRU / Stacked-LSTM vs. baselines
├── improved_pb_v1.py           Stacking · Cu_lag1 feature · MPC classification
├── improved_pb_v2.py           Quantile regression + interval calibration
├── improved_pb_v3.py           Chow test + stable-period retraining
├── requirements.txt
└── *_results/                  Metrics (CSV/JSON) and figures (PNG) for each stage
```

Every script carries a self-contained docstring header stating its purpose, method, inputs, outputs and literature reference.

## Future Improvements

- **Fix interval calibration.** The 8–12 pp under-coverage should be corrected with conformal prediction, which gives finite-sample coverage guarantees that quantile regression alone does not.
- **Address class imbalance in exceedance detection.** With 12 positives in 277 test rows, threshold tuning on precision–recall rather than accuracy, plus cost-sensitive learning, should raise usable precision.
- **Widen spatial coverage.** Two stations cannot separate local emission from city-wide background; adding stations or low-cost sensor data would let station effects be modelled instead of absorbed into a categorical feature.
- **Test regime-aware models.** Since the 2011 break is statistically real but naive truncation loses too much data, sample-weighting by recency or an explicit regime-switching model is the better way to use the finding.

## References

- Chen, L. et al. (2025). Transparent and reliable construction cost prediction using advanced machine learning and explainable AI. *Engineering Science and Technology, an International Journal*, 70, 102159.
- Chow, G. C. (1960). Tests of equality between sets of coefficients in two linear regressions. *Econometrica*, 28(3), 591–605.
- Meinshausen, N. (2006). Quantile Regression Forests. *Journal of Machine Learning Research*, 7, 983–999.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
- Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. *EMNLP*.
- Lundberg, S. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*.
