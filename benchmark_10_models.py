"""
====================================================================
БЕНЧМАРК 10 МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ
для прогноза концентрации свинца (Pb) в атмосферном воздухе Алматы
====================================================================

Методология адаптирована из статьи:
    Chen, L. et al. (2025). Transparent and reliable construction cost
    prediction using advanced machine learning and explainable AI.
    Engineering Science and Technology, an International Journal, 70, 102159.
    https://doi.org/10.1016/j.jestch.2025.102159

10 моделей: Ridge, Lasso, Elastic Net, KNN, Extra Trees, Gradient Boosting,
            AdaBoost, XGBoost, CatBoost, HistGradientBoosting

====================================================================
ИНСТРУКЦИЯ ПО ЗАПУСКУ
====================================================================

1) Установите Python 3.9+ (рекомендуется 3.10 или 3.11).

2) Установите все нужные библиотеки одной командой:

    pip install pandas numpy scikit-learn xgboost catboost matplotlib

   Если у вас Anaconda:
    conda install pandas numpy scikit-learn matplotlib
    pip install xgboost catboost

3) Положите файл ml_ready_dataset.csv ЛИБО в подпапку data/ рядом со
   скриптом, ЛИБО просто рядом со скриптом. Структура папки может быть
   любой из этих двух:

   Вариант 1:                       Вариант 2:
   benchmark/                       benchmark/
   ├── benchmark_10_models.py       ├── benchmark_10_models.py
   ├── ml_ready_dataset.csv         └── data/
   └── requirements.txt                 └── ml_ready_dataset.csv

4) Запустите:

    python benchmark_10_models.py

5) Результаты сохранятся в папку benchmark_results/:
    - benchmark_A_no_climate.csv     — метрики по выборке A
    - benchmark_B_with_climate.csv   — метрики по выборке B
    - summary_table.csv              — сводная таблица
    - best_params_*.json             — лучшие гиперпараметры
    - fig_metrics_A.png              — графики (стиль Fig.4 Chen et al.)
    - fig_metrics_B.png
    - fig_compare_A_vs_B.png

Время работы: ~3-7 минут на ноутбуке среднего класса.
====================================================================
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path

# --------------------------------------------------------------------
# Проверка зависимостей с понятным сообщением об ошибке
# --------------------------------------------------------------------
REQUIRED = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "matplotlib": "matplotlib",
}
missing = []
for module_name, pip_name in REQUIRED.items():
    try:
        __import__(module_name)
    except ImportError:
        missing.append(pip_name)
if missing:
    print("ОШИБКА: не установлены библиотеки:", ", ".join(missing))
    print("\nУстановите их командой:")
    print(f"    pip install {' '.join(missing)}")
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------
RANDOM_STATE = 42
K_FOLDS = 5
TEST_SIZE = 0.30

SCRIPT_DIR = Path(__file__).resolve().parent

# Скрипт ищет ml_ready_dataset.csv в нескольких типичных местах:
#   1) ./data/ml_ready_dataset.csv  (в подпапке data рядом со скриптом)
#   2) ./ml_ready_dataset.csv       (рядом со скриптом)
#   3) ../data/ml_ready_dataset.csv (в data на уровень выше)
# Если найден — используется первый найденный.
CANDIDATE_PATHS = [
    SCRIPT_DIR / "data" / "ml_ready_dataset.csv",
    SCRIPT_DIR / "ml_ready_dataset.csv",
    SCRIPT_DIR.parent / "data" / "ml_ready_dataset.csv",
]
INPUT_CSV = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])

OUTPUT_DIR = SCRIPT_DIR / "benchmark_results"

FEATURES_NO_CLIMATE = [
    "post", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "Pb_lag1", "Pb_lag2", "Pb_roll3_mean",
    "Cd", "As", "Cr", "Cu",
]
CLIMATE_FEATURES = [
    "temp_mean", "temp_min", "temp_max", "temp_std",
    "wind_speed_mean", "wind_speed_max", "wind_dir_mean",
    "rainy_share", "snowy_share", "foggy_share", "dust_share", "haze_share",
]
FEATURES_WITH_CLIMATE = FEATURES_NO_CLIMATE + CLIMATE_FEATURES


# --------------------------------------------------------------------
# Загрузка датасета
# --------------------------------------------------------------------
def load_dataset():
    if not INPUT_CSV.exists():
        print("ОШИБКА: не удалось найти ml_ready_dataset.csv")
        print("\nСкрипт искал файл в следующих местах:")
        for p in CANDIDATE_PATHS:
            print(f"  - {p}")
        print("\nПоложите ml_ready_dataset.csv в любую из этих папок и запустите снова.")
        sys.exit(1)

    print(f"Использую датасет: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, parse_dates=["start_date"])
    # Обязательные колонки — только без климата
    required_cols = ["Pb"] + FEATURES_NO_CLIMATE
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"ОШИБКА: в датасете отсутствуют ОБЯЗАТЕЛЬНЫЕ колонки: {missing_cols}")
        sys.exit(1)
    # Климатические колонки — опционально
    has_climate = all(c in df.columns for c in CLIMATE_FEATURES)
    if not has_climate:
        absent = [c for c in CLIMATE_FEATURES if c not in df.columns]
        print(f"Внимание: в датасете нет климатических колонок ({len(absent)} отсутствует).")
        print(f"  Выборка B (с климатом) будет пропущена.")
    print(f"Датасет загружен: {len(df)} строк, {len(df.columns)} колонок")
    return df, has_climate


# --------------------------------------------------------------------
# Подготовка выборок A и B
# --------------------------------------------------------------------
def prepare_samples(df, has_climate):
    # A: без климата
    A = df.dropna(subset=["Pb"] + FEATURES_NO_CLIMATE).copy()
    print(f"Выборка A (без климата): {len(A)} строк, {len(FEATURES_NO_CLIMATE)} признаков")

    B = None
    if has_climate:
        # B: с климатом. Атмосферные явления — нулём, остальное — медианой
        B = df.dropna(subset=["Pb", "Pb_lag1", "Pb_lag2", "Pb_roll3_mean",
                              "Cd", "As", "Cr", "Cu",
                              "temp_mean", "wind_speed_mean"]).copy()
        for col in ["rainy_share", "snowy_share", "foggy_share", "dust_share", "haze_share"]:
            B[col] = B[col].fillna(0.0)
        for col in ["wind_dir_mean", "temp_min", "temp_max", "temp_std", "wind_speed_max"]:
            if B[col].isna().any():
                B[col] = B[col].fillna(B[col].median())
        print(f"Выборка B (с климатом):  {len(B)} строк, {len(FEATURES_WITH_CLIMATE)} признаков")
    else:
        print("Выборка B (с климатом):  пропущена (нет климатических колонок в датасете)")
    return A, B


# --------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "MSE":  float(mse),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "MBE":  float(np.mean(np.asarray(y_true) - np.asarray(y_pred))),
    }


# --------------------------------------------------------------------
# 10 моделей и сетки гиперпараметров
# --------------------------------------------------------------------
def get_models_and_grids():
    return {
        "Ridge Regression": (
            Ridge(random_state=RANDOM_STATE),
            {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "Lasso Regression": (
            Lasso(random_state=RANDOM_STATE, max_iter=20000),
            {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]},
        ),
        "Elastic Net Regression": (
            ElasticNet(random_state=RANDOM_STATE, max_iter=20000),
            {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
             "l1_ratio": [0.1, 0.5, 0.9]},
        ),
        "KNN Regression": (
            KNeighborsRegressor(),
            {"n_neighbors": [3, 5, 7, 10, 15],
             "weights": ["uniform", "distance"]},
        ),
        "Extra Trees Regression": (
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [100, 300],
             "max_depth": [None, 10, 20],
             "min_samples_leaf": [1, 2, 4]},
        ),
        "Gradient Boosting Regression": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [100, 300],
             "learning_rate": [0.05, 0.1],
             "max_depth": [3, 5, 7]},
        ),
        "AdaBoost Regression": (
            AdaBoostRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [50, 100, 200],
             "learning_rate": [0.1, 0.5, 1.0]},
        ),
        "XGBoost Regression": (
            XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
                         objective="reg:squarederror"),
            {"n_estimators": [100, 300],
             "learning_rate": [0.05, 0.1],
             "max_depth": [3, 5, 7],
             "subsample": [0.8, 1.0]},
        ),
        "CatBoost Regression": (
            CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
            {"iterations": [200, 500],
             "learning_rate": [0.05, 0.1],
             "depth": [4, 6, 8]},
        ),
        "HistGradientBoosting Regression": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {"max_iter": [200, 500],
             "learning_rate": [0.05, 0.1],
             "max_depth": [None, 5, 10],
             "min_samples_leaf": [10, 20]},
        ),
    }


# --------------------------------------------------------------------
# Бенчмарк одной выборки
# --------------------------------------------------------------------
def run_benchmark(sample_name, df, feature_cols):
    print("\n" + "=" * 78)
    print(f"БЕНЧМАРК: {sample_name}")
    print(f"Строк: {len(df)}, Признаков: {len(feature_cols)}")
    print("=" * 78)

    X = df[feature_cols].values.astype(float)
    y_log = np.log1p(df["Pb"].values.astype(float))

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE
    )

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    y_train_orig = np.expm1(y_train_log)
    y_test_orig = np.expm1(y_test_log)

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    best_params_all = {}

    for name, (estimator, grid) in get_models_and_grids().items():
        print(f"--- {name} ---", flush=True)
        t0 = time.time()
        gs = GridSearchCV(
            estimator, grid,
            scoring="neg_root_mean_squared_error",
            cv=kf, n_jobs=-1, refit=True,
        )
        gs.fit(X_train_s, y_train_log)
        train_time = time.time() - t0

        best_model = gs.best_estimator_
        best_params_all[name] = gs.best_params_

        y_pred_train = np.expm1(best_model.predict(X_train_s))
        y_pred_test = np.expm1(best_model.predict(X_test_s))
        m_train = evaluate(y_train_orig, y_pred_train)
        m_test = evaluate(y_test_orig, y_pred_test)

        results.append({
            "Model": name,
            "Train_R2": m_train["R2"], "Train_RMSE": m_train["RMSE"],
            "Train_MSE": m_train["MSE"], "Train_MAE": m_train["MAE"],
            "Train_MBE": m_train["MBE"],
            "Test_R2": m_test["R2"], "Test_RMSE": m_test["RMSE"],
            "Test_MSE": m_test["MSE"], "Test_MAE": m_test["MAE"],
            "Test_MBE": m_test["MBE"],
            "CV_RMSE_log_scale": float(-gs.best_score_),
            "Train_time_s": train_time,
            "Best_params": json.dumps(gs.best_params_, ensure_ascii=False),
        })
        print(f"  Test  R²={m_test['R2']:.3f}  RMSE={m_test['RMSE']:.4f}  "
              f"MSE={m_test['MSE']:.5f}  MAE={m_test['MAE']:.4f}  "
              f"MBE={m_test['MBE']:+.4f}  (time={train_time:.1f}s)")

    res_df = pd.DataFrame(results).sort_values("Test_R2", ascending=False).reset_index(drop=True)
    csv_path = OUTPUT_DIR / f"benchmark_{sample_name}.csv"
    json_path = OUTPUT_DIR / f"best_params_{sample_name}.json"
    res_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_params_all, f, ensure_ascii=False, indent=2)
    print(f"\nСохранены: {csv_path.name}, {json_path.name}")
    return res_df


# --------------------------------------------------------------------
# Графики (стиль Fig. 4 Chen et al.)
# --------------------------------------------------------------------
def shorten_name(name):
    return (name.replace("Regression", "")
                .replace("HistGradientBoosting", "HistGB")
                .replace("Gradient Boosting", "GradBoost").strip())


def plot_metrics(df, title, fname):
    df = df.copy()
    df["short"] = df["Model"].apply(shorten_name)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    w = 0.35

    ax1.bar(x - w/2, df["Test_RMSE"], w, label="RMSE", color="#5dade2", alpha=0.85)
    ax1.bar(x + w/2, df["Test_MAE"], w, label="MAE", color="#f5b041", alpha=0.85)
    ax1.set_ylabel("RMSE, MAE (мкг/м³)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["short"], rotation=35, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(x, df["Test_R2"], color="#229954", marker="o", lw=2, label="R²")
    ax2.set_ylabel("R²", color="#229954")
    ax2.set_ylim(min(df["Test_R2"].min() - 0.1, 0), 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Сохранён график: {fname.name}")


def plot_compare(A, B, fname):
    order = A["Model"].apply(shorten_name).tolist()
    A_s = A.copy(); A_s["short"] = A["Model"].apply(shorten_name)
    B_s = B.copy(); B_s["short"] = B["Model"].apply(shorten_name)
    B_s = B_s.set_index("short").reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(order))
    w = 0.4
    ax.bar(x - w/2, A_s["Test_R2"], w,
           label="Выборка A (без климата)", color="#3498db")
    ax.bar(x + w/2, B_s["Test_R2"], w,
           label="Выборка B (с климатом)", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("R² (Test)")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title("Сравнение моделей: с климатом vs без климата (Pb, тест)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Сохранён график: {fname.name}")


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    print("=" * 78)
    print("БЕНЧМАРК 10 МОДЕЛЕЙ ДЛЯ ПРОГНОЗА Pb В ВОЗДУХЕ АЛМАТЫ")
    print("=" * 78)

    OUTPUT_DIR.mkdir(exist_ok=True)
    df, has_climate = load_dataset()
    A, B = prepare_samples(df, has_climate)

    A.to_csv(OUTPUT_DIR / "sample_A_no_climate.csv", index=False)

    res_A = run_benchmark("A_no_climate", A, FEATURES_NO_CLIMATE)

    res_B = None
    if B is not None:
        B.to_csv(OUTPUT_DIR / "sample_B_with_climate.csv", index=False)
        res_B = run_benchmark("B_with_climate", B, FEATURES_WITH_CLIMATE)

    # Сводная таблица
    summary = pd.DataFrame({
        "Model": res_A["Model"],
        "A_R2": res_A["Test_R2"].values,
        "A_RMSE": res_A["Test_RMSE"].values,
        "A_MAE": res_A["Test_MAE"].values,
    })
    if res_B is not None:
        res_B_sorted = res_B.set_index("Model").reindex(res_A["Model"]).reset_index()
        summary["B_R2"] = res_B_sorted["Test_R2"].values
        summary["B_RMSE"] = res_B_sorted["Test_RMSE"].values
        summary["B_MAE"] = res_B_sorted["Test_MAE"].values
    summary.to_csv(OUTPUT_DIR / "summary_table.csv", index=False)

    # Графики
    plot_metrics(res_A,
                 f"Сравнение 10 моделей на тестовой выборке A (n={len(A)}, без климата)",
                 OUTPUT_DIR / "fig_metrics_A.png")
    if res_B is not None:
        plot_metrics(res_B,
                     f"Сравнение 10 моделей на тестовой выборке B (n={len(B)}, с климатом)",
                     OUTPUT_DIR / "fig_metrics_B.png")
        plot_compare(res_A, res_B, OUTPUT_DIR / "fig_compare_A_vs_B.png")

    # Итог в консоль
    print("\n" + "=" * 78)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 78)
    print(f"\nВыборка A (без климата, n={len(A)}):")
    print(res_A[["Model", "Test_R2", "Test_RMSE", "Test_MSE", "Test_MAE"]]
          .round(4).to_string(index=False))
    if res_B is not None:
        print(f"\nВыборка B (с климатом, n={len(B)}):")
        print(res_B[["Model", "Test_R2", "Test_RMSE", "Test_MSE", "Test_MAE"]]
              .round(4).to_string(index=False))
    else:
        print("\nВыборка B (с климатом): не запускалась — нет климатических колонок в датасете.")
    print(f"\nВсе результаты сохранены в: {OUTPUT_DIR}")
    print("Готово.")


if __name__ == "__main__":
    main()
