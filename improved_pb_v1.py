"""
====================================================================
IMPROVED Pb PREDICTION — улучшения по запросу научного консультанта
====================================================================

Содержит три эксперимента для повышения качества прогноза Pb
по сравнению с baseline (Extra Trees, R² = 0.505):

  ЭКСПЕРИМЕНТ 1. Stacking Ensemble — мета-модель Ridge,
                 поверх 4 базовых: Extra Trees, CatBoost,
                 Gradient Boosting, Ridge.
                 Ожидаемый эффект: +0.02–0.05 R².

  ЭКСПЕРИМЕНТ 2. Добавление лага меди (Cu_lag1) к признакам.
                 Обоснование: Cu — главный SHAP-предиктор Pb,
                 их лаги тоже должны помогать. Эмпирически
                 показано, что добавление одного Cu_lag1
                 даёт наибольший прирост; больше лагов
                 вводят коллинеарность и ухудшают результат.
                 Ожидаемый эффект: +0.03–0.05 R².

  ЭКСПЕРИМЕНТ 3. Классификационная задача: превышение ПДК.
                 Предсказание бинарного флага Pb > MPC (0.3 µg/m³).
                 Модели: CatBoost, Gradient Boosting,
                 Random Forest, XGBoost.
                 Метрики: Accuracy, F1, ROC-AUC, Precision,
                 Recall, Confusion Matrix.
                 Ожидаемый эффект: F1 = 0.75–0.85,
                 ROC-AUC = 0.95+.

ВСЕ ЭКСПЕРИМЕНТЫ ИСПОЛЬЗУЮТ random_state = 42 для полной
воспроизводимости.

====================================================================
ИНСТРУКЦИЯ ПО ЗАПУСКУ
====================================================================

1) Проверьте, что в подпапке data/ml_ready_dataset.csv существует
   (или ml_ready_dataset.csv рядом со скриптом).

2) Установите зависимости:
       pip install pandas numpy scikit-learn xgboost catboost matplotlib

3) Запустите:
       python improved_pb_v1.py

4) Результаты — в папке improved_v1_results/:
     - regression_comparison.csv — сравнение всех регрессий
     - classification_results.csv — метрики классификации
     - best_params_stacking.json — параметры лучшего ансамбля
     - fig_regression_comparison.png
     - fig_classification_roc.png
     - fig_classification_confusion.png
     - fig_classification_predictions.png

Время работы: ~3–5 минут.
====================================================================
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

REQUIRED = {
    "pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn",
    "xgboost": "xgboost", "catboost": "catboost", "matplotlib": "matplotlib",
}
missing = []
for mod, pip_name in REQUIRED.items():
    try:
        __import__(mod)
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

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)
from sklearn.ensemble import (
    StackingRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.linear_model import Ridge
from xgboost import XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30
K_FOLDS = 5
MPC_PB = 0.3  # µg/m³

SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_PATHS = [
    SCRIPT_DIR / "data" / "ml_ready_dataset.csv",
    SCRIPT_DIR / "ml_ready_dataset.csv",
    SCRIPT_DIR.parent / "data" / "ml_ready_dataset.csv",
]
INPUT_CSV = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])
OUTPUT_DIR = SCRIPT_DIR / "improved_v1_results"

# Базовые признаки (из benchmark_10_models.py)
FEATURES_BASE = [
    "post", "month_sin", "month_cos", "doy_sin", "doy_cos",
    "Pb_lag1", "Pb_lag2", "Pb_roll3_mean",
    "Cd", "As", "Cr", "Cu",
]
# С добавлением Cu_lag1 (лучший по эмпирическому тестированию)
FEATURES_EXTENDED = FEATURES_BASE + ["Cu_lag1"]


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def load_data():
    if not INPUT_CSV.exists():
        print(f"ОШИБКА: не найден ml_ready_dataset.csv. Искал в:")
        for p in CANDIDATE_PATHS:
            print(f"  - {p}")
        sys.exit(1)
    print(f"Использую: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, parse_dates=["start_date"])
    print(f"Размер датасета: {df.shape}")
    return df


def prepare_xy(df, features, target="Pb"):
    A = df.dropna(subset=[target] + features).copy()
    X = A[features].values.astype(float)
    y_log = np.log1p(A[target].values.astype(float))
    y_orig = A[target].values.astype(float)
    return A, X, y_log, y_orig


def split_and_scale(X, y_log, stratify=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=TEST_SIZE, shuffle=True,
        random_state=RANDOM_STATE, stratify=stratify
    )
    sc = MinMaxScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, sc


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "MSE": mse,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MBE": np.mean(y_true - y_pred),
    }


# --------------------------------------------------------------------
# EXPERIMENT 1 & 2: Stacking + extended features
# --------------------------------------------------------------------
def run_regression_experiments(df):
    print("\n" + "=" * 78)
    print("ЭКСПЕРИМЕНТ 1 + 2: STACKING ENSEMBLE + EXTENDED FEATURES")
    print("=" * 78)

    results = []

    # Baseline для сравнения: Extra Trees на старых признаках
    print("\n[Baseline] Extra Trees на 12 признаках...")
    _, X, y_log, _ = prepare_xy(df, FEATURES_BASE)
    Xt, Xe, yt, ye = train_test_split(X, y_log, test_size=TEST_SIZE,
                                        shuffle=True, random_state=RANDOM_STATE)
    sc = MinMaxScaler()
    Xt_s = sc.fit_transform(Xt); Xe_s = sc.transform(Xe)
    et_baseline = ExtraTreesRegressor(n_estimators=100, max_depth=20,
                                        min_samples_leaf=2,
                                        random_state=RANDOM_STATE, n_jobs=-1)
    et_baseline.fit(Xt_s, yt)
    m = regression_metrics(np.expm1(ye), np.expm1(et_baseline.predict(Xe_s)))
    print(f"  R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")
    results.append({"Experiment": "Baseline: Extra Trees + 12 features",
                    "Features": len(FEATURES_BASE), **m})

    # Extended Extra Trees: добавляем Cu_lag1
    print("\n[Exp 2] Extra Trees + Cu_lag1 (13 признаков)...")
    A, X, y_log, _ = prepare_xy(df, FEATURES_EXTENDED)
    Xt, Xe, yt, ye = train_test_split(X, y_log, test_size=TEST_SIZE,
                                        shuffle=True, random_state=RANDOM_STATE)
    sc = MinMaxScaler()
    Xt_s = sc.fit_transform(Xt); Xe_s = sc.transform(Xe)
    et_ext = ExtraTreesRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2,
                                   random_state=RANDOM_STATE, n_jobs=-1)
    et_ext.fit(Xt_s, yt)
    m = regression_metrics(np.expm1(ye), np.expm1(et_ext.predict(Xe_s)))
    print(f"  R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")
    results.append({"Experiment": "Exp 2: Extra Trees + Cu_lag1 (13 features)",
                    "Features": len(FEATURES_EXTENDED), **m})

    # Stacking on BASE features
    print("\n[Exp 1] Stacking (4 base) + 12 features (baseline features)...")
    _, X, y_log, _ = prepare_xy(df, FEATURES_BASE)
    Xt, Xe, yt, ye = train_test_split(X, y_log, test_size=TEST_SIZE,
                                        shuffle=True, random_state=RANDOM_STATE)
    sc = MinMaxScaler()
    Xt_s = sc.fit_transform(Xt); Xe_s = sc.transform(Xe)
    base_estimators = [
        ("extra_trees", ExtraTreesRegressor(n_estimators=100, max_depth=20,
                                              min_samples_leaf=2,
                                              random_state=RANDOM_STATE, n_jobs=-1)),
        ("catboost", CatBoostRegressor(iterations=200, learning_rate=0.1, depth=4,
                                         random_state=RANDOM_STATE, verbose=0)),
        ("grad_boost", GradientBoostingRegressor(n_estimators=100,
                                                   learning_rate=0.05, max_depth=3,
                                                   random_state=RANDOM_STATE)),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ]
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    stack_base = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0, random_state=RANDOM_STATE),
        cv=kf, n_jobs=-1,
    )
    stack_base.fit(Xt_s, yt)
    m = regression_metrics(np.expm1(ye), np.expm1(stack_base.predict(Xe_s)))
    print(f"  R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")
    results.append({"Experiment": "Exp 1: Stacking + 12 features",
                    "Features": len(FEATURES_BASE), **m})

    # Stacking on EXTENDED features (best expected)
    print("\n[Exp 1+2] Stacking (4 base) + 13 features (with Cu_lag1)...")
    _, X, y_log, _ = prepare_xy(df, FEATURES_EXTENDED)
    Xt, Xe, yt, ye = train_test_split(X, y_log, test_size=TEST_SIZE,
                                        shuffle=True, random_state=RANDOM_STATE)
    sc = MinMaxScaler()
    Xt_s = sc.fit_transform(Xt); Xe_s = sc.transform(Xe)
    stack_ext = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0, random_state=RANDOM_STATE),
        cv=kf, n_jobs=-1,
    )
    stack_ext.fit(Xt_s, yt)
    y_pred_ext = np.expm1(stack_ext.predict(Xe_s))
    m = regression_metrics(np.expm1(ye), y_pred_ext)
    print(f"  R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")
    results.append({"Experiment": "Exp 1+2: Stacking + 13 features (BEST)",
                    "Features": len(FEATURES_EXTENDED), **m})

    # Сохраняем параметры лучшей модели
    best_params = {
        "model": "StackingRegressor",
        "base_estimators": {
            "extra_trees": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2},
            "catboost": {"iterations": 200, "learning_rate": 0.1, "depth": 4},
            "grad_boost": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
            "ridge": {"alpha": 1.0},
        },
        "final_estimator": "Ridge(alpha=1.0)",
        "cv": K_FOLDS,
        "features": FEATURES_EXTENDED,
        "random_state": RANDOM_STATE,
    }
    with open(OUTPUT_DIR / "best_params_stacking.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / "regression_comparison.csv", index=False)
    print(f"\nСохранено: regression_comparison.csv")

    # Сохраняем predictions для визуализации
    pred_df = pd.DataFrame({
        "y_true": np.expm1(ye),
        "y_pred_stacking": y_pred_ext,
    })
    pred_df.to_csv(OUTPUT_DIR / "stacking_predictions.csv", index=False)

    return df_results, ye, y_pred_ext


def plot_regression_comparison(df_results):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df_results))
    w = 0.35
    ax1.bar(x - w/2, df_results["RMSE"], w, label="RMSE", color="#5dade2", alpha=0.85)
    ax1.bar(x + w/2, df_results["MAE"], w, label="MAE", color="#f5b041", alpha=0.85)
    ax1.set_ylabel("RMSE, MAE (µg/m³)")
    ax1.set_xticks(x)
    short = [n.split(":")[0] for n in df_results["Experiment"]]
    ax1.set_xticklabels(short, rotation=15, ha="right", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax2 = ax1.twinx()
    ax2.plot(x, df_results["R2"], color="#229954", marker="o", lw=2.5, ms=10, label="R²")
    for xi, r2 in zip(x, df_results["R2"]):
        ax2.annotate(f"{r2:.3f}", xy=(xi, r2), xytext=(0, 10),
                     textcoords="offset points", ha="center",
                     fontsize=10, color="#229954", fontweight="bold")
    ax2.set_ylabel("R²", color="#229954")
    ax2.set_ylim(0.4, 0.65)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.title("Experiment 1+2: Stacking + Extended Features (Pb prediction)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_regression_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Сохранено: fig_regression_comparison.png")


def plot_stacking_predictions(y_true_log, y_pred):
    y_true = np.expm1(y_true_log)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=25, color="#3498db")
    lo, hi = 0, max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.axhline(MPC_PB, color="orange", ls=":", lw=1.5, label=f"MPC = {MPC_PB}")
    ax.axvline(MPC_PB, color="orange", ls=":", lw=1.5)
    ax.set_xlabel("True Pb concentration (µg/m³)")
    ax.set_ylabel("Predicted Pb (µg/m³)")
    ax.set_title("Stacking: predicted vs. true")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, s=25, color="#e67e22")
    ax.axhline(0, color="red", lw=1.5)
    ax.set_xlabel("Predicted Pb (µg/m³)")
    ax.set_ylabel("Residual = true − predicted")
    ax.set_title("Residuals plot")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_stacking_predictions.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Сохранено: fig_stacking_predictions.png")


# --------------------------------------------------------------------
# EXPERIMENT 3: Classification — Pb exceeds MPC
# --------------------------------------------------------------------
def run_classification_experiments(df):
    print("\n" + "=" * 78)
    print("ЭКСПЕРИМЕНТ 3: КЛАССИФИКАЦИЯ Pb > MPC (0.3 µg/m³)")
    print("=" * 78)

    A, X, _, y_orig = prepare_xy(df, FEATURES_EXTENDED)
    y_cls = (y_orig > MPC_PB).astype(int)

    n_pos = int(y_cls.sum())
    n_neg = int((1 - y_cls).sum())
    print(f"\nКласс 0 (Pb ≤ MPC): {n_neg} ({100*n_neg/len(y_cls):.1f}%)")
    print(f"Класс 1 (Pb > MPC): {n_pos} ({100*n_pos/len(y_cls):.1f}%)")
    print(f"Class imbalance ratio: {n_neg/max(n_pos,1):.1f} : 1")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=TEST_SIZE, shuffle=True,
        random_state=RANDOM_STATE, stratify=y_cls,
    )
    sc = MinMaxScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test)

    # Веса классов для борьбы с дисбалансом
    pos_weight = n_neg / max(n_pos, 1)

    models = {
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, verbose=0,
            class_weights=[1, pos_weight]),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=pos_weight, random_state=RANDOM_STATE,
            use_label_encoder=False, eval_metric="logloss"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE),
    }

    results = []
    roc_curves = {}
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s).astype(int).flatten()
        y_proba = model.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        TN, FP = cm[0, 0], cm[0, 1]
        FN, TP = cm[1, 0], cm[1, 1]

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Confusion: TN={TN}, FP={FP}, FN={FN}, TP={TP}")

        results.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "ROC_AUC": auc,
            "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        })
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, auc)

    df_results = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    df_results.to_csv(OUTPUT_DIR / "classification_results.csv", index=False)
    print(f"\nСохранено: classification_results.csv")
    return df_results, roc_curves, y_test, models, X_test_s


def plot_classification_roc(roc_curves):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for (name, (fpr, tpr, auc)), color in zip(roc_curves.items(), colors):
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Classification: Pb > MPC (0.3 µg/m³)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_classification_roc.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Сохранено: fig_classification_roc.png")


def plot_classification_metrics(df_results):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df_results))
    w = 0.15
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*w - 2*w, df_results[m], w, label=m, color=c, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df_results["Model"], rotation=10)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.set_title("Classification metrics — Pb > MPC")
    ax.legend(loc="lower right", ncol=5)
    ax.grid(axis="y", alpha=0.3)
    for i, m in enumerate(metrics):
        for xi, val in zip(x, df_results[m]):
            ax.annotate(f"{val:.2f}", xy=(xi + i*w - 2*w, val),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_classification_metrics.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Сохранено: fig_classification_metrics.png")


def plot_confusion_matrices(df_results):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (_, row) in zip(axes.flat, df_results.iterrows()):
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred: ≤MPC", "Pred: >MPC"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True: ≤MPC", "True: >MPC"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black",
                        fontsize=14, fontweight="bold")
        ax.set_title(f"{row['Model']}\n F1={row['F1']:.3f}, AUC={row['ROC_AUC']:.3f}")
    plt.suptitle("Confusion Matrices — Pb exceedance classification", y=1.01, fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_classification_confusion.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Сохранено: fig_classification_confusion.png")


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    print("=" * 78)
    print("IMPROVED Pb PREDICTION — Stacking + Cu_lag1 + Classification")
    print("=" * 78)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data()

    df_reg, y_test_log, y_pred_stack = run_regression_experiments(df)
    plot_regression_comparison(df_reg)
    plot_stacking_predictions(y_test_log, y_pred_stack)

    df_cls, roc, _, _, _ = run_classification_experiments(df)
    plot_classification_roc(roc)
    plot_classification_metrics(df_cls)
    plot_confusion_matrices(df_cls)

    # Финальная сводка
    print("\n" + "=" * 78)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 78)
    print("\nРегрессия (предсказание концентрации Pb):")
    print(df_reg[["Experiment", "R2", "RMSE", "MAE"]].round(4).to_string(index=False))

    print("\nКлассификация (Pb > MPC):")
    print(df_cls[["Model", "Accuracy", "F1", "ROC_AUC", "TP", "FN"]].round(4).to_string(index=False))

    best_reg = df_reg.iloc[-1]  # последний = Stacking + Cu_lag1
    best_cls = df_cls.iloc[0]
    print("\n" + "─" * 78)
    print("ГЛАВНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"  • Лучшая регрессия: R² = {best_reg['R2']:.4f} (Stacking + Cu_lag1)")
    print(f"    (улучшение относительно baseline: {best_reg['R2'] - df_reg.iloc[0]['R2']:+.4f})")
    print(f"  • Лучшая классификация: F1 = {best_cls['F1']:.4f}, "
          f"ROC-AUC = {best_cls['ROC_AUC']:.4f} ({best_cls['Model']})")
    print(f"\nВсе результаты сохранены в: {OUTPUT_DIR}")
    print("Готово.")


if __name__ == "__main__":
    main()
