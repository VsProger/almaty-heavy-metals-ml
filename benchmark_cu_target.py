"""
====================================================================
БЕНЧМАРК 10 МОДЕЛЕЙ — Cu (МЕДЬ) КАК ЦЕЛЕВАЯ
====================================================================

Дополнительный эксперимент по запросу научного руководителя:
сравнить результаты с прогнозом Pb (свинца).

Методология идентична benchmark_10_models.py:
  - те же 10 моделей (Ridge, Lasso, ElasticNet, KNN, ExtraTrees,
    GradientBoosting, AdaBoost, XGBoost, CatBoost, HistGradientBoosting)
  - те же 2 выборки: без климата (n≈919) и с климатом (n≈203)
  - K-Fold CV (K=5), GridSearchCV, MinMaxScaler, log1p-таргет
  - random_state=42 для воспроизводимости

Изменения:
  - целевая переменная: Cu (медь) вместо Pb
  - используется обновлённый датасет ml_ready_dataset_v2.csv
    (NaN заменены на 1e-9, добавлены ПДК-колонки)
  - в признаках Pb теперь используется вместо Cu (логично: при
    предсказании Cu использовать Pb как индикатор-предиктор)

Цель эксперимента:
  Сравнить, насколько хорошо предсказываются концентрации меди (источник —
  автотранспорт: тормозные колодки, шины) по сравнению со свинцом
  (источник — историческое наследие этилированного бензина). Гипотеза:
  Cu должен предсказываться с похожей или лучшей точностью, т.к. это
  более "стационарный" процесс без сильного исторического тренда.

====================================================================
ИНСТРУКЦИЯ ПО ЗАПУСКУ
====================================================================

1) Сначала запустите clean_dataset.py для создания
   ml_ready_dataset_v2.csv (см. инструкцию к нему).

2) Установите библиотеки (если ещё не):
   pip install pandas numpy scikit-learn xgboost catboost matplotlib

3) Запустите:
   python benchmark_cu_target.py

4) Результаты — в папке benchmark_cu_results/:
   - benchmark_cu_A_no_climate.csv      — метрики на выборке A
   - benchmark_cu_B_with_climate.csv    — метрики на выборке B
   - summary_table_cu.csv               — сводка по двум выборкам
   - best_params_cu_*.json              — лучшие гиперпараметры
   - fig_metrics_cu_A.png               — графики (стиль Fig.4 Chen et al.)
   - fig_metrics_cu_B.png
   - fig_compare_cu_A_vs_B.png
   - comparison_cu_vs_pb.csv            — прямое сравнение с Pb-бенчмарком
                                          (если есть benchmark_results/)
   - fig_cu_vs_pb.png                   — график сравнения

Время работы: ~3-7 минут.
====================================================================
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path

# --------------------------------------------------------------------
# Проверка зависимостей
# --------------------------------------------------------------------
REQUIRED = {
    "pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn",
    "xgboost": "xgboost", "catboost": "catboost", "matplotlib": "matplotlib",
}
missing = []
for module_name, pip_name in REQUIRED.items():
    try:
        __import__(module_name)
    except ImportError:
        missing.append(pip_name)
if missing:
    print("ОШИБКА: не установлены библиотеки:", ", ".join(missing))
    print(f"\nУстановите: pip install {' '.join(missing)}")
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
    ExtraTreesRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, HistGradientBoostingRegressor,
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
TARGET = "Cu"  # целевая переменная — теперь МЕДЬ

SCRIPT_DIR = Path(__file__).resolve().parent

# Ищем датасет: сначала v2, затем обычный (clean_from_scratch.py пишет в ml_ready_dataset.csv,
# но он уже содержит ПДК-колонки и заменённые NaN — поэтому это тот же v2)
CANDIDATE_PATHS = [
    SCRIPT_DIR / "data" / "ml_ready_dataset_v2.csv",
    SCRIPT_DIR / "ml_ready_dataset_v2.csv",
    SCRIPT_DIR.parent / "data" / "ml_ready_dataset_v2.csv",
    SCRIPT_DIR / "data" / "ml_ready_dataset.csv",
    SCRIPT_DIR / "ml_ready_dataset.csv",
    SCRIPT_DIR.parent / "data" / "ml_ready_dataset.csv",
]
INPUT_CSV = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])
OUTPUT_DIR = SCRIPT_DIR / "benchmark_cu_results"

# Признаки (как в benchmark_10_models.py, но Cu выбывает из признаков —
# она теперь таргет — а Pb используется как один из металлов-предикторов)
FEATURES_NO_CLIMATE = [
    "post", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "Cu_lag1", "Cu_lag2", "Cu_roll3_mean",  # лаги ТЕПЕРЬ для Cu
    "Cd", "As", "Cr", "Pb",  # металлы-предикторы (Pb теперь в признаках)
]
CLIMATE_FEATURES = [
    "temp_mean", "temp_min", "temp_max", "temp_std",
    "wind_speed_mean", "wind_speed_max", "wind_dir_mean",
    "rainy_share", "snowy_share", "foggy_share", "dust_share", "haze_share",
]
FEATURES_WITH_CLIMATE = FEATURES_NO_CLIMATE + CLIMATE_FEATURES


def load_dataset():
    if not INPUT_CSV.exists():
        print(f"ОШИБКА: не найден ml_ready_dataset.csv (или ml_ready_dataset_v2.csv)")
        print("Запустите сначала clean_from_scratch.py, чтобы создать датасет.")
        print("Искал в:")
        for p in CANDIDATE_PATHS:
            print(f"  - {p}")
        sys.exit(1)
    print(f"Использую датасет: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, parse_dates=["start_date"])
    # Обязательные — только без климата
    required = [TARGET] + FEATURES_NO_CLIMATE
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"ОШИБКА: в датасете нет ОБЯЗАТЕЛЬНЫХ колонок: {missing_cols}")
        sys.exit(1)
    # Климат — опционально
    has_climate = all(c in df.columns for c in CLIMATE_FEATURES)
    if not has_climate:
        absent = [c for c in CLIMATE_FEATURES if c not in df.columns]
        print(f"Внимание: в датасете нет климатических колонок ({len(absent)} отсутствует).")
        print(f"  Выборка B (с климатом) будет пропущена.")
    print(f"Загружено: {len(df)} строк × {len(df.columns)} колонок")
    return df, has_climate


def prepare_samples(df, has_climate):
    """Аналогично benchmark_10_models.py, но для Cu."""
    A = df.dropna(subset=[TARGET] + FEATURES_NO_CLIMATE).copy()
    print(f"Выборка A (без климата): {len(A)} строк, "
          f"{len(FEATURES_NO_CLIMATE)} признаков")

    B = None
    if has_climate:
        B = df.dropna(subset=[TARGET, "Cu_lag1", "Cu_lag2", "Cu_roll3_mean",
                              "Cd", "As", "Cr", "Pb",
                              "temp_mean", "wind_speed_mean"]).copy()
        for col in ["rainy_share", "snowy_share", "foggy_share",
                    "dust_share", "haze_share"]:
            B[col] = B[col].fillna(0.0)
        for col in ["wind_dir_mean", "temp_min", "temp_max", "temp_std",
                    "wind_speed_max"]:
            if B[col].isna().any():
                B[col] = B[col].fillna(B[col].median())
        print(f"Выборка B (с климатом):  {len(B)} строк, "
              f"{len(FEATURES_WITH_CLIMATE)} признаков")
    else:
        print("Выборка B (с климатом):  пропущена (нет климатических колонок в датасете)")
    return A, B


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "MSE":  float(mse),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "MBE":  float(np.mean(np.asarray(y_true) - np.asarray(y_pred))),
    }


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
            {"n_estimators": [100, 300], "max_depth": [None, 10, 20],
             "min_samples_leaf": [1, 2, 4]},
        ),
        "Gradient Boosting Regression": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1],
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
            {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1],
             "max_depth": [3, 5, 7], "subsample": [0.8, 1.0]},
        ),
        "CatBoost Regression": (
            CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
            {"iterations": [200, 500], "learning_rate": [0.05, 0.1],
             "depth": [4, 6, 8]},
        ),
        "HistGradientBoosting Regression": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {"max_iter": [200, 500], "learning_rate": [0.05, 0.1],
             "max_depth": [None, 5, 10], "min_samples_leaf": [10, 20]},
        ),
    }


def run_benchmark(sample_name, df, feature_cols):
    print("\n" + "=" * 78)
    print(f"БЕНЧМАРК Cu: {sample_name}")
    print(f"Строк: {len(df)}, Признаков: {len(feature_cols)}")
    print("=" * 78)

    X = df[feature_cols].values.astype(float)
    y_log = np.log1p(df[TARGET].values.astype(float))

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
        gs = GridSearchCV(estimator, grid,
                          scoring="neg_root_mean_squared_error",
                          cv=kf, n_jobs=-1, refit=True)
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
              f"MAE={m_test['MAE']:.4f}  MBE={m_test['MBE']:+.4f}  "
              f"(time={train_time:.1f}s)")

    res_df = pd.DataFrame(results).sort_values("Test_R2", ascending=False).reset_index(drop=True)
    csv_path = OUTPUT_DIR / f"benchmark_cu_{sample_name}.csv"
    json_path = OUTPUT_DIR / f"best_params_cu_{sample_name}.json"
    res_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_params_all, f, ensure_ascii=False, indent=2)
    print(f"\nСохранены: {csv_path.name}, {json_path.name}")
    return res_df


# --------------------------------------------------------------------
# Графики
# --------------------------------------------------------------------
def shorten(name):
    return (name.replace("Regression", "")
                .replace("HistGradientBoosting", "HistGB")
                .replace("Gradient Boosting", "GradBoost").strip())


def plot_metrics(df, title, fname):
    df = df.copy()
    df["short"] = df["Model"].apply(shorten)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    w = 0.35
    ax1.bar(x - w/2, df["Test_RMSE"], w, label="RMSE", color="#5dade2", alpha=0.85)
    ax1.bar(x + w/2, df["Test_MAE"],  w, label="MAE",  color="#f5b041", alpha=0.85)
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
    order = A["Model"].apply(shorten).tolist()
    A_s = A.copy(); A_s["short"] = A["Model"].apply(shorten)
    B_s = B.copy(); B_s["short"] = B["Model"].apply(shorten)
    B_s = B_s.set_index("short").reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(order))
    w = 0.4
    ax.bar(x - w/2, A_s["Test_R2"], w,
           label=f"Cu / Выборка A (n={len(A)})", color="#3498db")
    ax.bar(x + w/2, B_s["Test_R2"], w,
           label=f"Cu / Выборка B (n={len(B)})", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("R² (Test)")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title("Cu: сравнение моделей с климатом vs без климата")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Сохранён график: {fname.name}")


def plot_cu_vs_pb(res_cu_A, res_pb_A_path, fname):
    """Прямое сравнение Cu vs Pb на выборке A (если есть данные по Pb)."""
    if not res_pb_A_path.exists():
        print(f"  Пропускаю сравнение Cu vs Pb: не найден {res_pb_A_path}")
        return None

    res_pb_A = pd.read_csv(res_pb_A_path)
    # Объединяем по моделям
    cmp = res_cu_A[["Model", "Test_R2", "Test_RMSE", "Test_MAE"]].copy()
    cmp.columns = ["Model", "Cu_R2", "Cu_RMSE", "Cu_MAE"]
    pb_short = res_pb_A[["Model", "Test_R2", "Test_RMSE", "Test_MAE"]].copy()
    pb_short.columns = ["Model", "Pb_R2", "Pb_RMSE", "Pb_MAE"]
    merged = cmp.merge(pb_short, on="Model")

    # Сохраняем CSV
    merged.to_csv(OUTPUT_DIR / "comparison_cu_vs_pb.csv", index=False)
    print(f"Сохранена сравнительная таблица: comparison_cu_vs_pb.csv")

    # График
    merged = merged.sort_values("Cu_R2", ascending=False).reset_index(drop=True)
    short = [shorten(m) for m in merged["Model"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(merged))
    w = 0.4
    ax.bar(x - w/2, merged["Cu_R2"], w, label="Cu (медь)", color="#e67e22")
    ax.bar(x + w/2, merged["Pb_R2"], w, label="Pb (свинец)", color="#3498db")
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=35, ha="right")
    ax.set_ylabel("R² (Test) — выборка без климата")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title("Прямое сравнение прогнозирования Cu vs Pb (по R²)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Сохранён график: {fname.name}")
    return merged


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    print("=" * 78)
    print("БЕНЧМАРК 10 МОДЕЛЕЙ — Cu (МЕДЬ) КАК ЦЕЛЕВАЯ")
    print("=" * 78)

    OUTPUT_DIR.mkdir(exist_ok=True)
    df, has_climate = load_dataset()
    A, B = prepare_samples(df, has_climate)

    res_A = run_benchmark("A_no_climate", A, FEATURES_NO_CLIMATE)

    res_B = None
    if B is not None:
        res_B = run_benchmark("B_with_climate", B, FEATURES_WITH_CLIMATE)

    # Сводная
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
    summary.to_csv(OUTPUT_DIR / "summary_table_cu.csv", index=False)

    # Графики
    plot_metrics(res_A,
                 f"Cu: сравнение 10 моделей на выборке A (n={len(A)}, без климата)",
                 OUTPUT_DIR / "fig_metrics_cu_A.png")
    if res_B is not None:
        plot_metrics(res_B,
                     f"Cu: сравнение 10 моделей на выборке B (n={len(B)}, с климатом)",
                     OUTPUT_DIR / "fig_metrics_cu_B.png")
        plot_compare(res_A, res_B, OUTPUT_DIR / "fig_compare_cu_A_vs_B.png")

    # Сравнение с Pb (если есть результаты Этапа 1)
    pb_csv = SCRIPT_DIR / "benchmark_results" / "benchmark_A_no_climate.csv"
    plot_cu_vs_pb(res_A, pb_csv, OUTPUT_DIR / "fig_cu_vs_pb.png")

    # Итог
    print("\n" + "=" * 78)
    print(f"ИТОГОВАЯ СВОДКА — Cu как целевая")
    print("=" * 78)
    print(f"\nВыборка A (без климата, n={len(A)}):")
    print(res_A[["Model", "Test_R2", "Test_RMSE", "Test_MSE", "Test_MAE"]]
          .round(4).to_string(index=False))
    if res_B is not None:
        print(f"\nВыборка B (с климатом, n={len(B)}):")
        print(res_B[["Model", "Test_R2", "Test_RMSE", "Test_MSE", "Test_MAE"]]
              .round(4).to_string(index=False))
    else:
        print("\nВыборка B (с климатом): не запускалась — нет климатических колонок.")
    print(f"\nВсе результаты сохранены в: {OUTPUT_DIR}")
    print("Готово.")


if __name__ == "__main__":
    main()