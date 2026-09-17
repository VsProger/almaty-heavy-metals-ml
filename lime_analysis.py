"""
====================================================================
LIME-АНАЛИЗ — локальное объяснение предсказаний
====================================================================

LIME (Local Interpretable Model-agnostic Explanations) — методология
объяснения отдельных предсказаний модели машинного обучения путём
построения локального линейного приближения вокруг конкретной точки
данных.

В отличие от SHAP, который даёт ГЛОБАЛЬНОЕ объяснение «какие признаки
важны в среднем», LIME объясняет ЛОКАЛЬНО «почему именно ЭТО конкретное
предсказание такое». Это особенно полезно для:
  • экстремальных событий (например, превышение ПДК),
  • случаев, когда модель ошибается,
  • демонстрации работы модели стейкхолдерам на конкретных примерах.

ССЫЛКА: Ribeiro M.T., Singh S., Guestrin C. (2016). "Why should I trust you?":
        Explaining the Predictions of Any Classifier. KDD'16, 1135-1144.

====================================================================
ЧТО ДЕЛАЕТ СКРИПТ
====================================================================

Для двух выборок (A без климата, B с климатом) и обоих таргетов
(Pb и Cu, если есть результаты по обоим):

  1) Обучает лучшую модель из бенчмарка (с её оптимальными гиперпараметрами).
  2) Выбирает 4 интересных случая для объяснения:
     • Максимальное истинное значение в test (например, пик ПДК)
     • Максимальное предсказание модели
     • Случай где модель сильнее всего ошиблась
     • Случайное "среднее" наблюдение
  3) Строит LIME-объяснения для каждого случая.
  4) Сохраняет:
     • lime_explanations_<sample>_<target>.html — интерактивные объяснения
     • lime_summary_<sample>_<target>.csv — таблица весов признаков
     • lime_<sample>_<target>_case_<i>.png — bar-chart для каждого случая

====================================================================
ИНСТРУКЦИЯ ПО ЗАПУСКУ
====================================================================

1) Должны быть выполнены:
   • benchmark_10_models.py (для Pb) → benchmark_results/
   ИЛИ
   • benchmark_cu_target.py  (для Cu) → benchmark_cu_results/

   Скрипт сам определит, какие данные есть, и сделает LIME для них.

2) Установите LIME (если ещё не):
   pip install lime

3) Запустите:
   python lime_analysis.py

4) Результаты — в папке lime_results/

Время работы: ~1-3 минуты.
====================================================================
"""

import os
import sys
import json
import warnings
from pathlib import Path

# Проверка зависимостей
REQUIRED = {
    "pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn",
    "matplotlib": "matplotlib", "lime": "lime",
    "catboost": "catboost", "xgboost": "xgboost",
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
warnings.filterwarnings("ignore")

import lime
import lime.lime_tabular

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, HistGradientBoostingRegressor,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# --------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_FEATURES_TO_SHOW = 10  # топ-N признаков в LIME

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "lime_results"

# Какие модели объясняем (берём лучшие из бенчмарка)
DEFAULT_MODEL = "Extra Trees Regression"


def build_model(name, params):
    if name == "Ridge Regression":
        return Ridge(random_state=RANDOM_STATE, **params)
    if name == "Lasso Regression":
        return Lasso(random_state=RANDOM_STATE, max_iter=20000, **params)
    if name == "Elastic Net Regression":
        return ElasticNet(random_state=RANDOM_STATE, max_iter=20000, **params)
    if name == "KNN Regression":
        return KNeighborsRegressor(**params)
    if name == "Extra Trees Regression":
        return ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if name == "Gradient Boosting Regression":
        return GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    if name == "AdaBoost Regression":
        return AdaBoostRegressor(random_state=RANDOM_STATE, **params)
    if name == "XGBoost Regression":
        return XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                            verbosity=0, objective="reg:squarederror", **params)
    if name == "CatBoost Regression":
        return CatBoostRegressor(random_state=RANDOM_STATE, verbose=0, **params)
    if name == "HistGradientBoosting Regression":
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    raise ValueError(f"Неизвестная модель: {name}")


# --------------------------------------------------------------------
# Признаки для каждого таргета
# --------------------------------------------------------------------
def get_features(target, with_climate):
    if target == "Pb":
        base = ["post", "month_sin", "month_cos", "doy_sin", "doy_cos",
                "Pb_lag1", "Pb_lag2", "Pb_roll3_mean",
                "Cd", "As", "Cr", "Cu"]
    elif target == "Cu":
        base = ["post", "month_sin", "month_cos", "doy_sin", "doy_cos",
                "Cu_lag1", "Cu_lag2", "Cu_roll3_mean",
                "Cd", "As", "Cr", "Pb"]
    else:
        raise ValueError(f"Неизвестный таргет: {target}")
    if with_climate:
        base = base + [
            "temp_mean", "temp_min", "temp_max", "temp_std",
            "wind_speed_mean", "wind_speed_max", "wind_dir_mean",
            "rainy_share", "snowy_share", "foggy_share",
            "dust_share", "haze_share",
        ]
    return base


# --------------------------------------------------------------------
# Поиск данных
# --------------------------------------------------------------------
def find_target_data(target):
    """Ищет данные для конкретного таргета. Возвращает dict с путями или None."""
    if target == "Pb":
        bench_dir = SCRIPT_DIR / "benchmark_results"
    elif target == "Cu":
        bench_dir = SCRIPT_DIR / "benchmark_cu_results"
    else:
        return None

    if not bench_dir.exists():
        return None

    needed_files = {
        "A_no_climate": (
            bench_dir / f"sample_A_no_climate.csv" if target == "Pb"
            else None,  # для Cu sample не сохранён
            bench_dir / (f"best_params_A_no_climate.json" if target == "Pb"
                         else "best_params_cu_A_no_climate.json"),
        ),
        "B_with_climate": (
            bench_dir / f"sample_B_with_climate.csv" if target == "Pb"
            else None,
            bench_dir / (f"best_params_B_with_climate.json" if target == "Pb"
                         else "best_params_cu_B_with_climate.json"),
        ),
    }

    # Для Cu sample не сохранён, нужно делать заново из dataset_v2
    if target == "Cu":
        v2_paths = [
            SCRIPT_DIR / "data" / "ml_ready_dataset_v2.csv",
            SCRIPT_DIR / "ml_ready_dataset_v2.csv",
            SCRIPT_DIR.parent / "data" / "ml_ready_dataset_v2.csv",
        ]
        v2_csv = next((p for p in v2_paths if p.exists()), None)
        if not v2_csv:
            return None
        # Загружаем v2 и сами формируем sample-ы
        df = pd.read_csv(v2_csv, parse_dates=["start_date"])
        return {"target": target, "df": df, "best_params_dir": bench_dir}

    # Для Pb проверяем все файлы
    for sample_name, (sample_csv, _) in needed_files.items():
        if sample_csv is None or not sample_csv.exists():
            return None
    return {"target": target, "bench_dir": bench_dir, "needed_files": needed_files}


# --------------------------------------------------------------------
# Подготовка train/test
# --------------------------------------------------------------------
def prepare_train_test(df, target, with_climate):
    """Готовит ту же выборку, что и в бенчмарке."""
    features = get_features(target, with_climate)

    # Сначала фильтрация по обязательным колонкам
    if not with_climate:
        sample = df.dropna(subset=[target] + features).copy()
    else:
        # Для B (с климатом) — как в benchmark_10_models.py
        required = [target] + [f for f in features if not (
            f.endswith("_share") or f in ["wind_dir_mean", "temp_min",
                                          "temp_max", "temp_std",
                                          "wind_speed_max"]
        )]
        sample = df.dropna(subset=required).copy()
        for col in ["rainy_share", "snowy_share", "foggy_share",
                    "dust_share", "haze_share"]:
            if col in sample.columns:
                sample[col] = sample[col].fillna(0.0)
        for col in ["wind_dir_mean", "temp_min", "temp_max", "temp_std",
                    "wind_speed_max"]:
            if col in sample.columns and sample[col].isna().any():
                sample[col] = sample[col].fillna(sample[col].median())

    X = sample[features].values.astype(float)
    y_log = np.log1p(sample[target].values.astype(float))

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE
    )

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train_log, y_test_log, features


# --------------------------------------------------------------------
# Выбор случаев для объяснения
# --------------------------------------------------------------------
def select_cases(y_test_orig, y_pred):
    """Выбирает 4 интересных индекса в test для объяснения."""
    cases = {}
    cases["max_true"]   = int(np.argmax(y_test_orig))
    cases["max_pred"]   = int(np.argmax(y_pred))
    # Случай максимальной ошибки
    cases["max_error"]  = int(np.argmax(np.abs(y_test_orig - y_pred)))
    # Случайный «средний» (медианное реальное значение)
    diff_to_median = np.abs(y_test_orig - np.median(y_test_orig))
    cases["typical"]    = int(np.argmin(diff_to_median))
    return cases


# --------------------------------------------------------------------
# Визуализация одного LIME-объяснения
# --------------------------------------------------------------------
def plot_lime_explanation(explanation, title, fname):
    """Bar-chart весов признаков от LIME."""
    pairs = explanation.as_list()  # [(feature_name, weight), ...]
    pairs = pairs[:N_FEATURES_TO_SHOW]
    pairs = sorted(pairs, key=lambda x: x[1])  # для красоты

    fig, ax = plt.subplots(figsize=(10, max(4, len(pairs) * 0.45)))
    feats = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]
    colors = ["#e74c3c" if w < 0 else "#27ae60" for w in weights]
    bars = ax.barh(feats, weights, color=colors, alpha=0.8)

    # Подписи значений
    for bar, w in zip(bars, weights):
        ha = "right" if w < 0 else "left"
        xpos = bar.get_width()
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"  {w:+.4f}  ", va="center", ha=ha, fontsize=9)

    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Вклад признака в предсказание (LIME weight)")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=130)
    plt.close()


# --------------------------------------------------------------------
# Главная функция LIME-анализа
# --------------------------------------------------------------------
def run_lime_for_sample(df_or_sample, target, sample_name, with_climate,
                        best_params, model_name=DEFAULT_MODEL):
    print(f"\n{'='*78}")
    print(f"LIME: target={target}, sample={sample_name}, model={model_name}")
    print("=" * 78)

    X_train_s, X_test_s, y_train_log, y_test_log, features = prepare_train_test(
        df_or_sample, target, with_climate
    )

    # Обучаем модель
    if model_name not in best_params:
        # Берём первую доступную
        model_name = list(best_params.keys())[0]
        print(f"  Внимание: {DEFAULT_MODEL} нет в best_params, используем {model_name}")

    model = build_model(model_name, best_params[model_name])
    model.fit(X_train_s, y_train_log)

    # Предсказания
    y_pred_log = model.predict(X_test_s)
    y_test_orig = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)

    # Выбираем 4 случая
    cases = select_cases(y_test_orig, y_pred)
    print(f"  Выбрано случаев: {len(cases)}")
    for case_name, idx in cases.items():
        print(f"    {case_name:12s}  idx={idx:>3}  "
              f"true={y_test_orig[idx]:.4f}  pred={y_pred[idx]:.4f}  "
              f"err={y_test_orig[idx]-y_pred[idx]:+.4f}")

    # LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_s, feature_names=features, mode="regression",
        random_state=RANDOM_STATE, discretize_continuous=False
    )

    # Объясняем каждый случай и сохраняем
    summary_rows = []
    for case_name, idx in cases.items():
        exp = explainer.explain_instance(
            X_test_s[idx], model.predict,
            num_features=min(len(features), N_FEATURES_TO_SHOW)
        )

        # PNG
        title = (f"LIME: {target}/{sample_name} — случай '{case_name}'\n"
                 f"true={y_test_orig[idx]:.4f}, "
                 f"pred={y_pred[idx]:.4f} мкг/м³ "
                 f"({model_name})")
        fname = OUTPUT_DIR / f"lime_{target}_{sample_name}_{case_name}.png"
        plot_lime_explanation(exp, title, fname)
        print(f"    PNG: {fname.name}")

        # HTML (интерактив)
        html_path = OUTPUT_DIR / f"lime_{target}_{sample_name}_{case_name}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(exp.as_html())

        # Сохраняем в сводную
        for feat_name, weight in exp.as_list():
            summary_rows.append({
                "target": target, "sample": sample_name,
                "case": case_name, "true": y_test_orig[idx],
                "pred": y_pred[idx], "feature": feat_name, "weight": weight,
            })

    return summary_rows


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    print("=" * 78)
    print("LIME-АНАЛИЗ")
    print("=" * 78)
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_summary = []

    # === Pb ===
    pb_data = find_target_data("Pb")
    if pb_data:
        print(f"\n[Pb] Найдены данные в: {pb_data['bench_dir']}")
        for sample_name, (sample_csv, params_json) in pb_data["needed_files"].items():
            with_climate = (sample_name == "B_with_climate")
            df = pd.read_csv(sample_csv)
            with open(params_json, encoding="utf-8") as f:
                best_params = json.load(f)
            rows = run_lime_for_sample(df, "Pb", sample_name,
                                       with_climate, best_params)
            all_summary.extend(rows)
    else:
        print("\n[Pb] Данные не найдены — пропускаю Pb.")
        print("     (для запуска нужны benchmark_results/ из benchmark_10_models.py)")

    # === Cu ===
    cu_data = find_target_data("Cu")
    if cu_data:
        bench_dir = cu_data["best_params_dir"]
        df = cu_data["df"]
        print(f"\n[Cu] Найдены данные в: {bench_dir}")
        for sample_name in ["A_no_climate", "B_with_climate"]:
            params_json = bench_dir / f"best_params_cu_{sample_name}.json"
            if not params_json.exists():
                print(f"  Пропускаю {sample_name}: нет {params_json.name}")
                continue
            with open(params_json, encoding="utf-8") as f:
                best_params = json.load(f)
            with_climate = (sample_name == "B_with_climate")
            rows = run_lime_for_sample(df, "Cu", sample_name,
                                       with_climate, best_params)
            all_summary.extend(rows)
    else:
        print("\n[Cu] Данные не найдены — пропускаю Cu.")
        print("     (для запуска нужны benchmark_cu_results/ из benchmark_cu_target.py)")

    if all_summary:
        df_summary = pd.DataFrame(all_summary)
        df_summary.to_csv(OUTPUT_DIR / "lime_summary.csv",
                          index=False, encoding="utf-8")
        print(f"\nСводная таблица: {(OUTPUT_DIR/'lime_summary.csv').name} "
              f"({len(df_summary)} строк)")

        # Итоговая консолидированная сводка
        print("\n" + "=" * 78)
        print("ТОП-3 ПРИЗНАКА ДЛЯ КАЖДОГО СЛУЧАЯ")
        print("=" * 78)
        for (target, sample, case), grp in df_summary.groupby(
                ["target", "sample", "case"]):
            grp_sorted = grp.reindex(grp["weight"].abs().sort_values(ascending=False).index)
            top3 = grp_sorted.head(3)
            true_v = top3["true"].iloc[0]
            pred_v = top3["pred"].iloc[0]
            print(f"\n{target}/{sample}/{case}  "
                  f"(true={true_v:.4f}, pred={pred_v:.4f}):")
            for _, r in top3.iterrows():
                print(f"  {r['feature']:30s}  weight = {r['weight']:+.4f}")

    print(f"\nВсе результаты сохранены в: {OUTPUT_DIR}")
    print("Готово.")


if __name__ == "__main__":
    main()
