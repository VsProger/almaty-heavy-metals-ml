"""
====================================================================
QUANTILE REGRESSION для Pb с доверительными интервалами (CI)
====================================================================

Эксперимент 4 из плана улучшений.

В отличие от обычной регрессии (предсказывает одну точку — среднее
или медиану), квантильная регрессия обучает отдельные модели для
разных квантилей распределения целевой переменной. Это позволяет:

  1) ПОЛУЧИТЬ ТОЧЕЧНОЕ ПРЕДСКАЗАНИЕ (медиана = Q50) с обычной
     метрикой R².

  2) ПОСТРОИТЬ ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ:
     - 80% CI = [Q10, Q90] — широкий интервал, ~80% случаев попадают
     - 50% CI = [Q25, Q75] — узкий интерквартильный диапазон
  
  3) ОЦЕНИТЬ КАЛИБРОВКУ — доля наблюдений, действительно попадающих
     в каждый интервал.

  4) ПРАКТИЧЕСКИЕ ПРИМЕНЕНИЯ:
     - "Уверенное превышение ПДК" = Q10 > MPC (даже нижняя граница
       уверенности модели выше нормы) — это сильный сигнал для
       раннего предупреждения регуляторных органов.
     - "Уверенно ниже ПДК" = Q90 < MPC — безопасное состояние.

МОДЕЛЬ: Gradient Boosting Regressor с loss="quantile" для каждого
квантиля Q ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}.

ССЫЛКА: Meinshausen, N. (2006). Quantile Regression Forests.
        Journal of Machine Learning Research, 7, 983–999.

====================================================================
ИНСТРУКЦИЯ ПО ЗАПУСКУ
====================================================================

1) Положите ml_ready_dataset.csv в data/ (или рядом со скриптом).

2) Установите зависимости:
       pip install pandas numpy scikit-learn matplotlib

3) Запустите:
       python improved_pb_v2.py

4) Результаты — в папке improved_v2_results/:
     - quantile_predictions.csv — все квантили для тестовой выборки
     - quantile_calibration.csv — табл. покрытий CI
     - quantile_summary.json — основные метрики
     - fig_quantile_calibration.png — calibration plot
     - fig_quantile_predictions.png — предсказания с CI
     - fig_quantile_intervals_sorted.png — отсортированные интервалы

Время работы: 1–2 минуты.
====================================================================
"""

import sys
import json
import warnings
from pathlib import Path

REQUIRED = {
    "pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
}
missing = []
for mod, pip in REQUIRED.items():
    try:
        __import__(mod)
    except ImportError:
        missing.append(pip)
if missing:
    print("ОШИБКА: установите:", " ".join(missing))
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

RANDOM_STATE = 42
TEST_SIZE = 0.30
MPC_PB = 0.3  # µg/m³
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_PATHS = [
    SCRIPT_DIR / "data" / "ml_ready_dataset.csv",
    SCRIPT_DIR / "ml_ready_dataset.csv",
    SCRIPT_DIR.parent / "data" / "ml_ready_dataset.csv",
]
INPUT_CSV = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])
OUTPUT_DIR = SCRIPT_DIR / "improved_v2_results"

FEATURES = [
    "post", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "Pb_lag1", "Pb_lag2", "Pb_roll3_mean",
    "Cd", "As", "Cr", "Cu", "Cu_lag1",
]


def load_and_prepare():
    if not INPUT_CSV.exists():
        print(f"ОШИБКА: не найден ml_ready_dataset.csv. Искал:")
        for p in CANDIDATE_PATHS:
            print(f"  {p}")
        sys.exit(1)
    print(f"Использую: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, parse_dates=["start_date"])

    A = df.dropna(subset=["Pb"] + FEATURES).copy()
    X = A[FEATURES].values.astype(float)
    y_log = np.log1p(A["Pb"].values.astype(float))
    y_orig = A["Pb"].values.astype(float)
    dates = A["start_date"].values

    Xt, Xe, yt_log, ye_log, yt_orig, ye_orig, dt_train, dt_test = train_test_split(
        X, y_log, y_orig, dates,
        test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE
    )
    sc = MinMaxScaler()
    Xt_s = sc.fit_transform(Xt)
    Xe_s = sc.transform(Xe)
    return Xt_s, Xe_s, yt_log, ye_log, ye_orig, dt_test


def train_quantile_models(X_train, y_train, X_test, quantiles):
    """Тренируем GBR для каждого квантиля и возвращаем предсказания."""
    predictions = {}
    for q in quantiles:
        print(f"  Q{int(100*q):02d}: training...", end=" ", flush=True)
        model = GradientBoostingRegressor(
            loss="quantile", alpha=q,
            n_estimators=300, max_depth=4, learning_rate=0.05,
            min_samples_leaf=5, random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        # Предсказание в log-пространстве, переводим обратно
        pred = np.expm1(model.predict(X_test))
        # Не должно быть отрицательных
        pred = np.maximum(pred, 0)
        predictions[q] = pred
        print("done")
    return predictions


def calibration_table(y_true, predictions):
    """Считаем покрытие для разных интервалов."""
    rows = []
    intervals = [
        (0.05, 0.95, "90% CI"),
        (0.10, 0.90, "80% CI"),
        (0.25, 0.75, "50% CI (IQR)"),
    ]
    for q_lo, q_hi, label in intervals:
        if q_lo not in predictions or q_hi not in predictions:
            continue
        lo = predictions[q_lo]
        hi = predictions[q_hi]
        coverage = np.mean((y_true >= lo) & (y_true <= hi))
        mean_width = np.mean(hi - lo)
        expected = q_hi - q_lo
        rows.append({
            "Interval": label,
            "Expected_coverage": expected,
            "Empirical_coverage": coverage,
            "Calibration_error": coverage - expected,
            "Mean_width_µg/m³": mean_width,
        })
    return pd.DataFrame(rows)


def practical_use_analysis(y_true, predictions):
    """Сколько раз модель уверенно прогнозирует превышение/безопасность."""
    q10 = predictions[0.10]
    q50 = predictions[0.50]
    q90 = predictions[0.90]

    actual_exceed = (y_true > MPC_PB)
    pred_q50_exceed = (q50 > MPC_PB)
    confident_exceed = (q10 > MPC_PB)
    confident_safe = (q90 < MPC_PB)
    uncertain = ~(confident_exceed | confident_safe)

    return {
        "n_test": len(y_true),
        "actually_exceed_MPC": int(actual_exceed.sum()),
        "median_predicts_exceed": int(pred_q50_exceed.sum()),
        "Q10>MPC (confident exceedance)": int(confident_exceed.sum()),
        "Q90<MPC (confident safe)": int(confident_safe.sum()),
        "uncertain": int(uncertain.sum()),
        "confident_safe_actually_safe": int((confident_safe & ~actual_exceed).sum()),
        "confident_safe_actually_exceed": int((confident_safe & actual_exceed).sum()),
    }


# --------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------
def plot_calibration(df_calib):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax.scatter(df_calib["Expected_coverage"], df_calib["Empirical_coverage"],
                s=150, color="#3498db", zorder=5)
    for _, r in df_calib.iterrows():
        ax.annotate(r["Interval"], xy=(r["Expected_coverage"], r["Empirical_coverage"]),
                     xytext=(10, -5), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration plot — quantile regression intervals")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_quantile_calibration.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Сохранено: fig_quantile_calibration.png")


def plot_predictions_with_ci(y_true, predictions, dates):
    """Predicted vs True с доверительными интервалами."""
    order = np.argsort(y_true)
    y_sorted = y_true[order]
    q50 = predictions[0.50][order]
    q10 = predictions[0.10][order]
    q90 = predictions[0.90][order]
    q25 = predictions[0.25][order]
    q75 = predictions[0.75][order]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(y_sorted))
    ax.fill_between(x, q10, q90, alpha=0.25, color="#3498db", label="80% CI")
    ax.fill_between(x, q25, q75, alpha=0.4, color="#3498db", label="50% CI")
    ax.plot(x, q50, "-", color="#3498db", lw=1.5, label="Median (Q50)")
    ax.scatter(x, y_sorted, color="#e74c3c", s=15, alpha=0.7,
                label="True value", zorder=5)
    ax.axhline(MPC_PB, color="orange", ls="--", lw=1.5, label=f"MPC = {MPC_PB}")
    ax.set_xlabel("Test sample (sorted by true Pb)")
    ax.set_ylabel("Pb concentration (µg/m³)")
    ax.set_title("Quantile regression: predicted median + 80%/50% confidence intervals")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_quantile_predictions.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Сохранено: fig_quantile_predictions.png")


def plot_intervals_log_scale(y_true, predictions):
    """Логарифмическая шкала — лучше видно работу с малыми значениями."""
    order = np.argsort(y_true)
    y_sorted = y_true[order]
    q50 = predictions[0.50][order]
    q10 = predictions[0.10][order]
    q90 = predictions[0.90][order]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(y_sorted))
    # Защита от log(0) — заменяем нули на маленькое
    eps = 1e-4
    ax.fill_between(x, q10 + eps, q90 + eps, alpha=0.3, color="#3498db", label="80% CI")
    ax.plot(x, q50 + eps, "-", color="#3498db", lw=1.5, label="Q50")
    ax.scatter(x, y_sorted + eps, color="#e74c3c", s=15, alpha=0.7, label="True", zorder=5)
    ax.axhline(MPC_PB, color="orange", ls="--", lw=1.5, label=f"MPC = {MPC_PB}")
    ax.set_yscale("log")
    ax.set_xlabel("Test sample (sorted by true Pb)")
    ax.set_ylabel("Pb concentration, log scale (µg/m³)")
    ax.set_title("Quantile regression: log-scale view")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_quantile_intervals_logscale.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Сохранено: fig_quantile_intervals_logscale.png")


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    print("=" * 78)
    print("QUANTILE REGRESSION для Pb — доверительные интервалы")
    print("=" * 78)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\nЗагрузка данных...")
    X_train, X_test, y_train_log, _, y_test_orig, dates_test = load_and_prepare()
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, признаков: {len(FEATURES)}")

    print(f"\nОбучение квантильных моделей для квантилей {QUANTILES}...")
    predictions = train_quantile_models(X_train, y_train_log, X_test, QUANTILES)

    # Метрики на медиане
    y_pred_median = predictions[0.50]
    r2 = r2_score(y_test_orig, y_pred_median)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_median))
    mae = mean_absolute_error(y_test_orig, y_pred_median)
    print(f"\n--- Median (Q50) prediction metrics ---")
    print(f"  R²   = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f} µg/m³")
    print(f"  MAE  = {mae:.4f} µg/m³")

    # Калибровка
    df_calib = calibration_table(y_test_orig, predictions)
    df_calib.to_csv(OUTPUT_DIR / "quantile_calibration.csv", index=False)
    print("\n--- Calibration of confidence intervals ---")
    print(df_calib.round(4).to_string(index=False))

    # Сохраняем все предсказания
    pred_df = pd.DataFrame({f"Q{int(100*q):02d}": predictions[q] for q in QUANTILES})
    pred_df["y_true"] = y_test_orig
    pred_df["start_date"] = dates_test
    pred_df.to_csv(OUTPUT_DIR / "quantile_predictions.csv", index=False)

    # Практическое применение
    practical = practical_use_analysis(y_test_orig, predictions)
    print("\n--- Practical use analysis ---")
    for k, v in practical.items():
        print(f"  {k}: {v}")

    # JSON summary
    summary = {
        "median_metrics": {"R2": float(r2), "RMSE": float(rmse), "MAE": float(mae)},
        "calibration": df_calib.to_dict(orient="records"),
        "practical": practical,
        "features": FEATURES,
        "quantiles": QUANTILES,
        "random_state": RANDOM_STATE,
    }
    with open(OUTPUT_DIR / "quantile_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Графики
    plot_calibration(df_calib)
    plot_predictions_with_ci(y_test_orig, predictions, dates_test)
    plot_intervals_log_scale(y_test_orig, predictions)

    print(f"\nВсе результаты сохранены в: {OUTPUT_DIR}")
    print("Готово.")


if __name__ == "__main__":
    main()
