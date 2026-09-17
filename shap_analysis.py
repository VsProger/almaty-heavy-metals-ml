"""
====================================================================
SHAP-АНАЛИЗ для топ-4 моделей прогноза концентрации Pb
====================================================================

Этап 2 диссертации. Воспроизводит SHAP-анализ из статьи:
    Chen, L. et al. (2025). Transparent and reliable construction cost
    prediction using advanced machine learning and explainable AI.
    Engineering Science and Technology, 70, 102159.
    Раздел 3.7 (SHAP-based Explainability), Fig. 7-8.

ЧТО ДЕЛАЕТ СКРИПТ
-----------------
Для двух выборок (A — без климата, B — с климатом) обучает топ-4 модели
с гиперпараметрами, найденными на Этапе 1 (берёт из best_params_*.json),
и считает SHAP-значения. Для каждой модели сохраняет:

  • beeswarm-plot (Fig. 7 статьи) — какой признак сколько вкладывает
    в каждое предсказание и в каком направлении (рост/снижение Pb)
  • bar-plot — среднее |SHAP|, чисто ранжирование важности признаков
  • CSV с числовыми значениями важности

Для лучшей модели дополнительно строится SHAP interaction heatmap
(Fig. 8 статьи) — взаимодействия между парами признаков.

ВЫБОР EXPLAINER-А
-----------------
SHAP — это framework с несколькими методами расчёта в зависимости от
типа модели. Скрипт автоматически выбирает оптимальный:
  • Tree-based (Extra Trees, CatBoost)  → TreeExplainer (точный, быстрый)
  • Linear (Ridge)                       → LinearExplainer (точный, моментальный)
  • Прочие (KNN)                         → KernelExplainer (приближённый, медленный)

ИНСТРУКЦИЯ ПО ЗАПУСКУ
=====================

1) Сначала должен быть выполнен Этап 1 (benchmark_10_models.py).
   В подпапке benchmark_results/ должны лежать:
     - sample_A_no_climate.csv
     - sample_B_with_climate.csv
     - best_params_A_no_climate.json
     - best_params_B_with_climate.json

2) Установите дополнительную библиотеку (если ещё не установлена):

       pip install shap

3) Запустите:

       python shap_analysis.py

4) Результаты сохранятся в подпапку shap_results/:
     - beeswarm_<sample>_<model>.png
     - bar_<sample>_<model>.png
     - feature_importance_<sample>_<model>.csv
     - interaction_heatmap_<sample>.png
     - summary_feature_importance.csv

Время работы: ~2-5 минут.
====================================================================
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

# --------------------------------------------------------------------
# Проверка зависимостей
# --------------------------------------------------------------------
REQUIRED = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "matplotlib": "matplotlib",
    "shap": "shap",
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

import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------
# Конфигурация — те же значения, что в benchmark_10_models.py
# --------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30

SCRIPT_DIR = Path(__file__).resolve().parent

# Скрипт ищет данные Этапа 1 в типичных местах
def find_dir(candidates):
    for c in candidates:
        if c.exists():
            return c
    return None

INPUT_DIR = find_dir([
    SCRIPT_DIR / "benchmark_results",
    SCRIPT_DIR.parent / "benchmark_results",
])
OUTPUT_DIR = SCRIPT_DIR / "shap_results"

# Те же признаки, что на Этапе 1
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

# Какие модели объясняем (4 типа — для разнообразия)
MODELS_TO_EXPLAIN = ["Ridge Regression", "KNN Regression",
                     "Extra Trees Regression", "CatBoost Regression"]

# Лучшая модель (для дополнительного interaction heatmap)
BEST_MODEL_NAME = "Extra Trees Regression"


# --------------------------------------------------------------------
# Загрузка данных Этапа 1
# --------------------------------------------------------------------
def load_stage1_data(sample_name, features):
    if INPUT_DIR is None:
        print("ОШИБКА: не найдена папка benchmark_results/")
        print("Сначала запустите benchmark_10_models.py (Этап 1).")
        sys.exit(1)

    sample_csv = INPUT_DIR / f"sample_{sample_name}.csv"
    params_json = INPUT_DIR / f"best_params_{sample_name}.json"
    if not sample_csv.exists() or not params_json.exists():
        print(f"ОШИБКА: не найдены файлы Этапа 1 для выборки {sample_name}")
        print(f"   Ожидаются: {sample_csv.name}, {params_json.name}")
        sys.exit(1)

    df = pd.read_csv(sample_csv)
    with open(params_json, encoding="utf-8") as f:
        best_params = json.load(f)

    # Воспроизводим тот же train/test split, что в benchmark_10_models.py
    X = df[features].values.astype(float)
    y_log = np.log1p(df["Pb"].values.astype(float))
    X_train, X_test, y_train, _ = train_test_split(
        X, y_log, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE
    )
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"  Загружено: train={len(X_train_s)}, test={len(X_test_s)}, features={len(features)}")
    return X_train_s, X_test_s, y_train, features, best_params


# --------------------------------------------------------------------
# Конструирование моделей из сохранённых гиперпараметров
# --------------------------------------------------------------------
def build_model(name, params):
    if name == "Ridge Regression":
        return Ridge(random_state=RANDOM_STATE, **params)
    if name == "KNN Regression":
        return KNeighborsRegressor(**params)
    if name == "Extra Trees Regression":
        return ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if name == "CatBoost Regression":
        return CatBoostRegressor(random_state=RANDOM_STATE, verbose=0, **params)
    raise ValueError(f"Неизвестная модель: {name}")


# --------------------------------------------------------------------
# Универсальный выбор SHAP explainer
# --------------------------------------------------------------------
def compute_shap_values(name, model, X_train_s, X_test_s):
    """Возвращает (shap_values, X_explain).
    X_explain может быть меньше X_test_s, если используем KernelExplainer.
    """
    if name in ("Extra Trees Regression", "CatBoost Regression"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_s)
        return shap_values, X_test_s, explainer
    if name == "Ridge Regression":
        explainer = shap.LinearExplainer(model, X_train_s)
        shap_values = explainer.shap_values(X_test_s)
        return shap_values, X_test_s, explainer
    if name == "KNN Regression":
        # KernelExplainer — медленный, поэтому ограничиваемся
        bg = shap.sample(X_train_s, 50, random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(model.predict, bg)
        n_explain = min(150, len(X_test_s))
        X_explain = X_test_s[:n_explain]
        shap_values = explainer.shap_values(X_explain, nsamples=100, silent=True)
        return shap_values, X_explain, explainer
    raise ValueError(f"Нет explainer для модели: {name}")


# --------------------------------------------------------------------
# Графики SHAP
# --------------------------------------------------------------------
def plot_beeswarm(shap_values, X_data, feature_names, title, fname):
    plt.figure()
    shap.summary_plot(shap_values, X_data, feature_names=feature_names,
                      show=False, max_display=min(len(feature_names), 15))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  graph: {fname.name}")


def plot_bar(shap_values, X_data, feature_names, title, fname):
    plt.figure()
    shap.summary_plot(shap_values, X_data, feature_names=feature_names,
                      plot_type="bar", show=False,
                      max_display=min(len(feature_names), 15))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  graph: {fname.name}")


def plot_interaction_heatmap(interaction_values, feature_names, title, fname):
    """Как Fig. 8 у Chen et al."""
    inter_mean = np.abs(interaction_values).mean(axis=0)
    n = len(feature_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(7, n * 0.6)))
    im = ax.imshow(inter_mean, cmap="YlOrRd")
    ax.set_xticks(range(n)); ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(feature_names)
    # Подписи значений в ячейках
    max_v = inter_mean.max() if inter_mean.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            v = inter_mean[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > max_v * 0.5 else "black")
    plt.colorbar(im, label="Mean |interaction|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  graph: {fname.name}")


# --------------------------------------------------------------------
# Главная функция бенчмарка SHAP по одной выборке
# --------------------------------------------------------------------
def run_shap_for_sample(sample_name, features):
    print("\n" + "=" * 78)
    print(f"SHAP-АНАЛИЗ: выборка {sample_name}")
    print("=" * 78)

    X_train_s, X_test_s, y_train, feature_names, best_params = load_stage1_data(
        sample_name, features
    )

    importance_rows = []

    for model_name in MODELS_TO_EXPLAIN:
        if model_name not in best_params:
            print(f"  Пропускаю {model_name} — нет гиперпараметров в JSON")
            continue
        print(f"\n--- {model_name} ---")
        params = best_params[model_name]

        # Обучаем модель заново (с теми же гиперпараметрами и теми же данными)
        model = build_model(model_name, params)
        t0 = time.time()
        model.fit(X_train_s, y_train)
        fit_t = time.time() - t0

        # SHAP-значения
        t0 = time.time()
        shap_values, X_explain, _explainer = compute_shap_values(
            model_name, model, X_train_s, X_test_s
        )
        shap_t = time.time() - t0
        print(f"  fit={fit_t:.1f}s, SHAP={shap_t:.1f}s, "
              f"n_explained={len(X_explain)}")

        # Безопасные имена для файлов
        safe_name = (model_name.replace(" Regression", "")
                              .replace(" ", "_"))
        # Beeswarm + bar
        plot_beeswarm(shap_values, X_explain, feature_names,
                      title=f"SHAP Beeswarm — {model_name} ({sample_name})",
                      fname=OUTPUT_DIR / f"beeswarm_{sample_name}_{safe_name}.png")
        plot_bar(shap_values, X_explain, feature_names,
                 title=f"SHAP Feature Importance — {model_name} ({sample_name})",
                 fname=OUTPUT_DIR / f"bar_{sample_name}_{safe_name}.png")

        # Числа важности признаков
        mean_abs = np.abs(shap_values).mean(axis=0)
        fi = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        fi["rank"] = range(1, len(fi) + 1)
        fi.to_csv(OUTPUT_DIR / f"feature_importance_{sample_name}_{safe_name}.csv",
                  index=False)

        # Для сводной
        for _, r in fi.iterrows():
            importance_rows.append({
                "sample": sample_name, "model": model_name,
                "feature": r["feature"],
                "mean_abs_shap": r["mean_abs_shap"], "rank": int(r["rank"]),
            })

        # Interaction heatmap (только для tree-моделей + лучшая)
        if model_name == BEST_MODEL_NAME:
            print(f"  Считаю interaction heatmap (это дольше)...")
            t0 = time.time()
            explainer = shap.TreeExplainer(model)
            n_sub = min(150, len(X_test_s))
            interaction_values = explainer.shap_interaction_values(X_test_s[:n_sub])
            print(f"    interaction time={time.time()-t0:.1f}s")
            plot_interaction_heatmap(
                interaction_values, feature_names,
                title=f"SHAP Interaction Heatmap — {model_name} ({sample_name})",
                fname=OUTPUT_DIR / f"interaction_heatmap_{sample_name}.png"
            )
            # Сохраняем матрицу
            inter_mean = np.abs(interaction_values).mean(axis=0)
            pd.DataFrame(inter_mean,
                         index=feature_names, columns=feature_names).to_csv(
                OUTPUT_DIR / f"interaction_matrix_{sample_name}.csv"
            )

    return pd.DataFrame(importance_rows)


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    print("=" * 78)
    print("SHAP-АНАЛИЗ — Этап 2")
    print("=" * 78)
    print(f"Папка с результатами Этапа 1: {INPUT_DIR}")
    print(f"Сохранение результатов: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Выборка A — без климата
    fi_A = run_shap_for_sample("A_no_climate", FEATURES_NO_CLIMATE)

    # Выборка B — с климатом
    fi_B = run_shap_for_sample("B_with_climate", FEATURES_WITH_CLIMATE)

    # Сводная таблица feature importance по всем моделям и выборкам
    summary = pd.concat([fi_A, fi_B], ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary_feature_importance.csv", index=False)

    # Топ-5 признаков по каждой (выборка × модель)
    print("\n" + "=" * 78)
    print("ТОП-5 ПРИЗНАКОВ ПО ВАЖНОСТИ (SHAP) — для каждой модели и выборки")
    print("=" * 78)
    for (sample, model), grp in summary.groupby(["sample", "model"]):
        print(f"\n{sample}  /  {model}:")
        top5 = grp.nsmallest(5, "rank")
        for _, r in top5.iterrows():
            print(f"  {int(r['rank']):>2}. {r['feature']:<20} "
                  f"mean|SHAP| = {r['mean_abs_shap']:.4f}")

    print(f"\nВсе результаты сохранены в: {OUTPUT_DIR}")
    print("Готово.")


if __name__ == "__main__":
    main()
