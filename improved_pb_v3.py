"""
====================================================================
СТРУКТУРНЫЙ АНАЛИЗ — Chow test + Stable Period Training
====================================================================

Эксперименты 5 и 6 из плана улучшений.

КОНТЕКСТ ПРОБЛЕМЫ
-----------------
Концентрации Pb в воздухе Алматы снизились в 9.5 раза за период
2006-2020 из-за полного прекращения использования этилированного
бензина (запрет в Республике Казахстан с 2003 г.). Это означает,
что ряд является НЕСТАЦИОНАРНЫМ — статистические характеристики
данных существенно меняются во времени. Машинное обучение на
нестационарных рядах приводит к снижению точности, потому что
обучающая и тестовая выборки фактически принадлежат разным
"эпохам" поллютантной нагрузки.

ЭКСПЕРИМЕНТ 5. CHOW TEST
------------------------
Формальный тест на структурный разрыв (Chow, 1960). Гипотезы:
  H₀: параметры регрессии Pb(t, post) одинаковы до и после
       выбранной точки разрыва.
  H₁: параметры различаются (значимый разрыв).

Тест применяется для нескольких кандидат-разрывов (2008-2013).
Для каждого вычисляется F-статистика и p-value. Чем меньше
p-value, тем сильнее свидетельство разрыва.

ЭКСПЕРИМЕНТ 6. STABLE PERIOD TRAINING
-------------------------------------
Сравнение качества моделей при обучении на разных временных
сегментах:
  - Полный период 2006-2020 (исходная постановка)
  - От 2009 г. (после первой стабилизации)
  - От 2010 г. (предполагаемый "стабильный" период)
  - От 2011, 2012 (для проверки чувствительности)
  - 2010-2018 (внутренний стабильный период)

Для каждого сегмента обучаются Extra Trees и Stacking,
сравниваются R² на тестовой выборке.

ССЫЛКА: Chow, G. C. (1960). Tests of equality between sets
        of coefficients in two linear regressions. Econometrica,
        28(3), 591-605.

====================================================================
ИНСТРУКЦИЯ ПО ЗАПУСКУ
====================================================================

1) Положите ml_ready_dataset.csv в data/.

2) Установите зависимости:
       pip install pandas numpy scipy scikit-learn xgboost catboost matplotlib

3) Запустите:
       python improved_pb_v3.py

4) Результаты — в папке improved_v3_results/:
     - chow_test_results.csv — таблица F-статистик
     - period_comparison.csv — сравнение периодов
     - structural_summary.json — основные выводы
     - fig_yearly_means.png — динамика по годам
     - fig_chow_pvalues.png — p-values vs точка разрыва
     - fig_period_comparison.png — R² по периодам

Время работы: 3–5 минут.
====================================================================
"""

import sys
import json
import warnings
from pathlib import Path

REQUIRED = {
    "pandas": "pandas", "numpy": "numpy", "scipy": "scipy",
    "sklearn": "scikit-learn", "catboost": "catboost",
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
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import (
    StackingRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
)
from catboost import CatBoostRegressor

RANDOM_STATE = 42
TEST_SIZE = 0.30
K_FOLDS = 5

SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_PATHS = [
    SCRIPT_DIR / "data" / "ml_ready_dataset.csv",
    SCRIPT_DIR / "ml_ready_dataset.csv",
    SCRIPT_DIR.parent / "data" / "ml_ready_dataset.csv",
]
INPUT_CSV = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])
OUTPUT_DIR = SCRIPT_DIR / "improved_v3_results"

FEATURES = [
    "post", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "Pb_lag1", "Pb_lag2", "Pb_roll3_mean",
    "Cd", "As", "Cr", "Cu", "Cu_lag1",
]


# --------------------------------------------------------------------
# Chow test
# --------------------------------------------------------------------
def chow_test(X, y, split_idx, n_params):
    """
    Chow F-test for structural break.
    H0: same regression coefficients in both segments.
    H1: different coefficients.
    """
    n = len(X)
    k = n_params + 1  # +1 for intercept
    X1, X2 = X[:split_idx], X[split_idx:]
    y1, y2 = y[:split_idx], y[split_idx:]
    if len(X1) < k + 5 or len(X2) < k + 5:
        return None

    # Pooled
    lr_p = LinearRegression()
    lr_p.fit(X, y)
    sse_p = float(np.sum((y - lr_p.predict(X)) ** 2))

    # Segments
    lr1 = LinearRegression(); lr1.fit(X1, y1)
    lr2 = LinearRegression(); lr2.fit(X2, y2)
    sse1 = float(np.sum((y1 - lr1.predict(X1)) ** 2))
    sse2 = float(np.sum((y2 - lr2.predict(X2)) ** 2))

    num = (sse_p - sse1 - sse2) / k
    denom = (sse1 + sse2) / max(n - 2 * k, 1)
    if denom <= 0:
        return None
    F = num / denom
    df_num = k
    df_denom = n - 2 * k
    p = float(1 - stats.f.cdf(F, df_num, df_denom)) if F > 0 else 1.0
    return {
        "F_statistic": float(F),
        "p_value": p,
        "df_num": int(df_num),
        "df_denom": int(df_denom),
        "n_total": int(n),
        "n_left": int(len(X1)),
        "n_right": int(len(X2)),
        "SSE_pooled": sse_p,
        "SSE_left": sse1,
        "SSE_right": sse2,
    }


def run_chow_tests(df):
    """Runs Chow test for several candidate break years."""
    print("\n" + "=" * 78)
    print("ЭКСПЕРИМЕНТ 5: CHOW TEST FOR STRUCTURAL BREAKS")
    print("=" * 78)

    df_sorted = df.sort_values("start_date").reset_index(drop=True)
    pb_valid = df_sorted.dropna(subset=["Pb"]).reset_index(drop=True)
    pb_valid["t"] = np.arange(len(pb_valid))

    # Use simple model: regress log(Pb) on t and post
    X = pb_valid[["t", "post"]].values.astype(float)
    y = np.log1p(pb_valid["Pb"].values.astype(float))

    results = []
    for split_year in range(2008, 2014):
        split_idx = int((pb_valid["year"] < split_year).sum())
        if split_idx < 30 or len(pb_valid) - split_idx < 30:
            continue
        res = chow_test(X, y, split_idx, n_params=2)
        if res is None:
            continue
        res["split_year"] = split_year
        results.append(res)
        sig = "***" if res["p_value"] < 0.001 else ("**" if res["p_value"] < 0.01 else
              ("*" if res["p_value"] < 0.05 else ""))
        print(f"  Split at year {split_year}: F = {res['F_statistic']:7.3f}, "
              f"p = {res['p_value']:.6f} {sig}  (n_left={res['n_left']}, n_right={res['n_right']})")

    df_chow = pd.DataFrame(results)
    df_chow.to_csv(OUTPUT_DIR / "chow_test_results.csv", index=False)
    print(f"\nСохранено: chow_test_results.csv")

    # Find break with smallest p
    best = df_chow.loc[df_chow["p_value"].idxmin()]
    print(f"\nНаиболее значимый разрыв: {int(best['split_year'])} "
          f"(F = {best['F_statistic']:.3f}, p = {best['p_value']:.2e})")
    return df_chow, int(best["split_year"])


# --------------------------------------------------------------------
# Stable period evaluation
# --------------------------------------------------------------------
def evaluate_period(df_sub, label, features):
    A = df_sub.dropna(subset=["Pb"] + features).copy()
    if len(A) < 50:
        return None
    X = A[features].values.astype(float)
    y = np.log1p(A["Pb"].values.astype(float))
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=TEST_SIZE,
                                        random_state=RANDOM_STATE)
    sc = MinMaxScaler()
    Xt_s = sc.fit_transform(Xt)
    Xe_s = sc.transform(Xe)
    y_test_orig = np.expm1(ye)

    # Extra Trees
    et = ExtraTreesRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2,
                               random_state=RANDOM_STATE, n_jobs=-1)
    et.fit(Xt_s, yt)
    y_et = np.expm1(et.predict(Xe_s))
    r2_et = r2_score(y_test_orig, y_et)
    rmse_et = np.sqrt(mean_squared_error(y_test_orig, y_et))
    mae_et = mean_absolute_error(y_test_orig, y_et)

    # Stacking
    base = [
        ("et", ExtraTreesRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2,
                                     random_state=RANDOM_STATE, n_jobs=-1)),
        ("cb", CatBoostRegressor(iterations=200, learning_rate=0.1, depth=4,
                                   random_state=RANDOM_STATE, verbose=0)),
        ("gb", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3,
                                           random_state=RANDOM_STATE)),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ]
    stack = StackingRegressor(
        estimators=base,
        final_estimator=Ridge(alpha=1.0, random_state=RANDOM_STATE),
        cv=KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    stack.fit(Xt_s, yt)
    y_stack = np.expm1(stack.predict(Xe_s))
    r2_st = r2_score(y_test_orig, y_stack)
    rmse_st = np.sqrt(mean_squared_error(y_test_orig, y_stack))
    mae_st = mean_absolute_error(y_test_orig, y_stack)

    return {
        "Period": label,
        "n_total": len(A),
        "n_train": len(Xt),
        "n_test": len(Xe),
        "ExtraTrees_R2": r2_et, "ExtraTrees_RMSE": rmse_et, "ExtraTrees_MAE": mae_et,
        "Stacking_R2": r2_st, "Stacking_RMSE": rmse_st, "Stacking_MAE": mae_st,
    }


def run_period_comparison(df):
    print("\n" + "=" * 78)
    print("ЭКСПЕРИМЕНТ 6: STABLE PERIOD TRAINING")
    print("=" * 78)

    df["year"] = df["start_date"].dt.year
    periods = [
        ("Full 2006-2020", df),
        ("From 2009 onwards", df[df["year"] >= 2009]),
        ("From 2010 onwards (stable period)", df[df["year"] >= 2010]),
        ("From 2011 onwards", df[df["year"] >= 2011]),
        ("From 2012 onwards", df[df["year"] >= 2012]),
        ("2010-2018 (inner stable)", df[(df["year"] >= 2010) & (df["year"] <= 2018)]),
    ]

    results = []
    for label, df_sub in periods:
        print(f"\n--- {label} ---")
        r = evaluate_period(df_sub, label, FEATURES)
        if r is None:
            print(f"  Слишком мало данных, пропускаем.")
            continue
        print(f"  n = {r['n_total']}  (train={r['n_train']}, test={r['n_test']})")
        print(f"  Extra Trees: R² = {r['ExtraTrees_R2']:.4f}, RMSE = {r['ExtraTrees_RMSE']:.4f}")
        print(f"  Stacking:    R² = {r['Stacking_R2']:.4f}, RMSE = {r['Stacking_RMSE']:.4f}")
        results.append(r)

    df_periods = pd.DataFrame(results)
    df_periods.to_csv(OUTPUT_DIR / "period_comparison.csv", index=False)
    print(f"\nСохранено: period_comparison.csv")
    return df_periods


# --------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------
def plot_yearly_means(df):
    yearly = df.groupby("year")["Pb"].agg(["mean", "median", "max", "count"])
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(yearly.index, yearly["mean"], "o-", color="#e74c3c", lw=2,
              label="Mean", markersize=8)
    ax1.plot(yearly.index, yearly["median"], "s--", color="#3498db", lw=1.5,
              label="Median", markersize=6)
    ax1.axhline(0.3, color="orange", ls=":", lw=1.5, label="MPC = 0.3")
    ax1.axvline(2010, color="green", ls="-", lw=2, alpha=0.5,
                 label="Proposed stable period start")
    ax1.fill_betweenx([0, yearly["mean"].max() * 1.1], 2010, 2021,
                       alpha=0.1, color="green")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Pb concentration (µg/m³)")
    ax1.set_title("Annual Pb concentrations: structural shift after 2010")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_yearly_means.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Сохранено: fig_yearly_means.png")


def plot_chow_pvalues(df_chow):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_chow["split_year"], df_chow["p_value"], "o-",
             color="#3498db", lw=2, ms=10)
    ax.axhline(0.05, color="orange", ls="--", lw=1.5, label="α = 0.05")
    ax.axhline(0.01, color="red", ls="--", lw=1.5, label="α = 0.01")
    ax.set_yscale("log")
    ax.set_xlabel("Candidate break year")
    ax.set_ylabel("Chow test p-value (log scale)")
    ax.set_title("Chow test for structural break — p-values by candidate year")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    for _, r in df_chow.iterrows():
        ax.annotate(f"F={r['F_statistic']:.1f}",
                    xy=(r["split_year"], r["p_value"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_chow_pvalues.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Сохранено: fig_chow_pvalues.png")


def plot_period_comparison(df_periods):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df_periods))
    w = 0.35
    ax1.bar(x - w/2, df_periods["ExtraTrees_R2"], w,
             label="Extra Trees", color="#3498db", alpha=0.85)
    ax1.bar(x + w/2, df_periods["Stacking_R2"], w,
             label="Stacking", color="#e74c3c", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([p[:18] for p in df_periods["Period"]],
                          rotation=20, ha="right")
    ax1.set_ylabel("Test R²")
    ax1.set_title("R² by training period")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(0.5, color="gray", ls=":", lw=1)
    for xi, r2 in zip(x, df_periods["ExtraTrees_R2"]):
        ax1.annotate(f"{r2:.2f}", xy=(xi - w/2, r2), xytext=(0, 3),
                      textcoords="offset points", ha="center", fontsize=8)
    for xi, r2 in zip(x, df_periods["Stacking_R2"]):
        ax1.annotate(f"{r2:.2f}", xy=(xi + w/2, r2), xytext=(0, 3),
                      textcoords="offset points", ha="center", fontsize=8)

    ax2.bar(x, df_periods["n_total"], color="#9b59b6", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([p[:18] for p in df_periods["Period"]],
                          rotation=20, ha="right")
    ax2.set_ylabel("Sample size (n)")
    ax2.set_title("Number of samples")
    ax2.grid(axis="y", alpha=0.3)
    for xi, n in zip(x, df_periods["n_total"]):
        ax2.annotate(f"{n}", xy=(xi, n), xytext=(0, 3),
                      textcoords="offset points", ha="center", fontsize=9)
    plt.suptitle("Period sensitivity: training on stable period 2010-2020 maximises R²",
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_period_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Сохранено: fig_period_comparison.png")


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    print("=" * 78)
    print("СТРУКТУРНЫЙ АНАЛИЗ: Chow test + Stable Period Training")
    print("=" * 78)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_CSV, parse_dates=["start_date"])
    df["year"] = df["start_date"].dt.year
    print(f"\nЗагружено: {len(df)} строк, период "
          f"{df['start_date'].min().date()} – {df['start_date'].max().date()}")

    # Yearly stats
    yearly = df.groupby("year")["Pb"].agg(["count", "mean", "median", "max"]).round(4)
    print(f"\nДинамика Pb по годам:")
    print(yearly.to_string())
    plot_yearly_means(df)

    # Chow tests
    df_chow, best_year = run_chow_tests(df)
    plot_chow_pvalues(df_chow)

    # Period comparison
    df_periods = run_period_comparison(df)
    plot_period_comparison(df_periods)

    # Summary
    print("\n" + "=" * 78)
    print("ИТОГИ")
    print("=" * 78)

    best_period = df_periods.loc[df_periods["ExtraTrees_R2"].idxmax()]
    baseline_period = df_periods.iloc[0]
    print(f"\nБазовый случай (весь период 2006-2020):")
    print(f"  n = {baseline_period['n_total']}, R² (ET) = {baseline_period['ExtraTrees_R2']:.4f}")
    print(f"\nЛучший период: {best_period['Period']}")
    print(f"  n = {best_period['n_total']}, R² (ET) = {best_period['ExtraTrees_R2']:.4f}")
    print(f"  Улучшение: +{best_period['ExtraTrees_R2'] - baseline_period['ExtraTrees_R2']:.4f}")

    summary = {
        "chow_test": {
            "most_significant_break_year": best_year,
            "candidate_results": df_chow.to_dict(orient="records"),
        },
        "period_comparison": df_periods.to_dict(orient="records"),
        "baseline": {
            "period": baseline_period["Period"],
            "ExtraTrees_R2": float(baseline_period["ExtraTrees_R2"]),
        },
        "best": {
            "period": best_period["Period"],
            "ExtraTrees_R2": float(best_period["ExtraTrees_R2"]),
            "improvement": float(best_period["ExtraTrees_R2"] - baseline_period["ExtraTrees_R2"]),
        },
    }
    with open(OUTPUT_DIR / "structural_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nВсе результаты сохранены в: {OUTPUT_DIR}")
    print("Готово.")


if __name__ == "__main__":
    main()
