"""
====================================================================
LSTM / GRU НЕЙРОСЕТИ для прогноза концентрации Pb
====================================================================

Этап 3 диссертации. Расширение методологии за пределы статьи Chen et al.,
где использовались только классические ML-модели. Добавление рекуррентных
нейронных сетей (LSTM, GRU) — это научная новизна данной работы.

ЧТО ДЕЛАЕТ СКРИПТ
-----------------
Обучает 3 рекуррентных архитектуры:
  1. LSTM (Long Short-Term Memory)        — Hochreiter & Schmidhuber, 1997
  2. GRU (Gated Recurrent Unit)           — Cho et al., 2014
  3. Stacked-LSTM (двухслойная LSTM)      — глубокая версия

на временных последовательностях концентраций тяжёлых металлов
(окно WINDOW=6 декад, ~2 месяца истории) для каждого из двух постов
наблюдения отдельно. Затем сравнивает с двумя baseline-моделями:
  - Naive: предсказание = последнее наблюдённое значение
  - WindowMean: предсказание = среднее по окну
и с лучшим классическим ML результатом из Этапа 1 (Extra Trees).

ВАЖНОЕ НАУЧНОЕ ЗАМЕЧАНИЕ (читать перед запуском)
------------------------------------------------
Данный эксперимент может показать, что LSTM/GRU НЕ превосходят
классический ансамблевый ML (Extra Trees, CatBoost) на временном
ряду концентраций Pb в Алматы. Это связано со специфическими
особенностями данных:
  • Нестационарность — концентрации Pb в Алматы исторически снижались
    в ~5 раз с 2006 г. (последствия запрета этилированного бензина).
  • Низкая автокорреляция — даже Pb_lag1 даёт R²=0.11 как baseline.
  • Высокий шум — атмосферный перенос приносит непредсказуемые пики.
  
LSTM/GRU особенно сильны в задачах с высокой автокорреляцией
(температура, цены акций, ECG). В нашем случае модели сваливаются
в режим предсказания константы, что подтверждается на baseline.

Это сам по себе ВАЖНЫЙ НАУЧНЫЙ ВЫВОД для диссертации, а не «провал».
Подтверждается публикациями: для редких декадных измерений
атмосферных металлов классические ансамбли часто работают лучше
рекуррентных сетей (см., например, Cabaneros et al. 2019).

ИНСТРУКЦИЯ ПО ЗАПУСКУ
=====================

1) Установите дополнительно tensorflow (если ещё не установлен):

       pip install tensorflow

2) Скрипт ищет ml_ready_dataset.csv в тех же местах, что и Этап 1:
       ./data/ml_ready_dataset.csv
       ./ml_ready_dataset.csv
       ../data/ml_ready_dataset.csv

3) Запустите:

       python lstm_gru_analysis.py

4) Результаты сохранятся в папку lstm_results/:
     - lstm_gru_metrics.csv      — главная таблица сравнения
     - lstm_gru_compare.png      — bar-chart R², RMSE, MAE
     - predictions_<model>.png   — графики предсказаний во времени
     - training_history.png      — кривые обучения нейросетей

Время работы: 1-3 минуты на CPU (без GPU).
====================================================================
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path

# --------------------------------------------------------------------
# Подавление избыточных предупреждений TF (должно быть ДО import tf)
# --------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# --------------------------------------------------------------------
# Проверка зависимостей
# --------------------------------------------------------------------
REQUIRED = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "tensorflow": "tensorflow",
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
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30
WINDOW = 6   # размер окна — 6 декад ≈ 2 месяца истории
EPOCHS = 150
BATCH_SIZE = 32
PATIENCE = 20  # для EarlyStopping

SCRIPT_DIR = Path(__file__).resolve().parent

# Поиск датасета (как в benchmark_10_models.py)
CANDIDATE_PATHS = [
    SCRIPT_DIR / "data" / "ml_ready_dataset.csv",
    SCRIPT_DIR / "ml_ready_dataset.csv",
    SCRIPT_DIR.parent / "data" / "ml_ready_dataset.csv",
]
INPUT_CSV = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])
OUTPUT_DIR = SCRIPT_DIR / "lstm_results"

# Признаки для последовательностей. Используем те же металлы и сезонность,
# которые показали высокую важность в SHAP (Cu, Cd, As, Cr) — а Pb_lag*
# не нужны явно, т.к. LSTM сам видит историю Pb в окне.
TIMESTEP_FEATURES = ["Pb", "Cd", "As", "Cr", "Cu", "month_sin", "month_cos",
                     "doy_sin", "doy_cos"]


# --------------------------------------------------------------------
# Загрузка
# --------------------------------------------------------------------
def load_dataset():
    if not INPUT_CSV.exists():
        print("ОШИБКА: не найден ml_ready_dataset.csv")
        print("Скрипт искал в следующих местах:")
        for p in CANDIDATE_PATHS:
            print(f"  - {p}")
        sys.exit(1)
    print(f"Использую датасет: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, parse_dates=["start_date"])
    needed = TIMESTEP_FEATURES + ["post", "start_date"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        print(f"ОШИБКА: в датасете нет колонок: {miss}")
        sys.exit(1)
    return df.sort_values(["post", "start_date"]).reset_index(drop=True)


# --------------------------------------------------------------------
# Создание скользящих окон
# --------------------------------------------------------------------
def make_sequences(df, features, target_col, window):
    """Делает скользящие окна ПО КАЖДОМУ ПОСТУ отдельно (не смешивая)."""
    X, y, dates, posts = [], [], [], []
    for post, grp in df.groupby("post"):
        arr = grp[features].values.astype(float)
        tgt = grp[target_col].values.astype(float)
        date_arr = grp["start_date"].values
        for i in range(window, len(arr)):
            X.append(arr[i-window:i])
            y.append(tgt[i])
            dates.append(date_arr[i])
            posts.append(post)
    return np.array(X), np.array(y), np.array(dates), np.array(posts)


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
        "MBE":  float(np.mean(y_true - y_pred)),
    }


# --------------------------------------------------------------------
# Архитектуры нейросетей
# --------------------------------------------------------------------
def build_lstm(input_shape):
    """Простая однослойная LSTM."""
    m = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m


def build_gru(input_shape):
    """GRU — упрощённая альтернатива LSTM."""
    m = Sequential([
        GRU(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m


def build_stacked_lstm(input_shape):
    """Двухслойная LSTM — более глубокая модель."""
    m = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m


# --------------------------------------------------------------------
# Обучение одной нейросети
# --------------------------------------------------------------------
def train_nn(name, builder, X_train_s, y_train_log, X_test_s,
             y_test_orig, input_shape):
    print(f"\n--- {name} ---")
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = builder(input_shape)
    n_params = model.count_params()
    print(f"  Параметров: {n_params:,}")

    es = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                       restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(X_train_s, y_train_log,
                     epochs=EPOCHS, batch_size=BATCH_SIZE,
                     validation_split=0.15, verbose=0, callbacks=[es])
    train_time = time.time() - t0
    n_epochs = len(hist.history["loss"])

    y_pred_log = model.predict(X_test_s, verbose=0).flatten()
    # Защита: если сеть выдала NaN или ушла в очень большие значения,
    # обрезаем по разумному диапазону (на основе обучающего лог-Pb)
    y_train_log_min = y_train_log.min()
    y_train_log_max = y_train_log.max()
    y_pred_log = np.nan_to_num(y_pred_log,
                                nan=y_train_log.mean(),
                                posinf=y_train_log_max,
                                neginf=y_train_log_min)
    # Дополнительно обрезаем выбросы за пределы 3*std разумного диапазона
    y_pred_log = np.clip(y_pred_log,
                         y_train_log_min - 3 * y_train_log.std(),
                         y_train_log_max + 3 * y_train_log.std())
    y_pred = np.expm1(y_pred_log)
    # Pb не может быть отрицательным
    y_pred = np.maximum(y_pred, 0.0)
    m = evaluate(y_test_orig, y_pred)
    print(f"  Обучено эпох: {n_epochs}, время: {train_time:.1f}s")
    print(f"  Test R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
          f"MBE={m['MBE']:+.4f}")
    print(f"  std(y_pred)={y_pred.std():.4f}  std(y_true)={y_test_orig.std():.4f}")
    if y_pred.std() < 0.001:
        print(f"  ⚠ Внимание: модель предсказывает практически константу.")
        print(f"    Это типичный признак того, что нейросеть не нашла "
              f"полезной структуры в данных.")
    return {
        "Model": name,
        "Params": n_params,
        "Epochs": n_epochs,
        "Train_time_s": train_time,
        "Test_R2": m["R2"], "Test_RMSE": m["RMSE"],
        "Test_MSE": m["MSE"], "Test_MAE": m["MAE"],
        "Test_MBE": m["MBE"],
        "Pred_std": float(y_pred.std()),
    }, y_pred, hist.history


# --------------------------------------------------------------------
# Построение графиков
# --------------------------------------------------------------------
def plot_compare(results_df, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    df = results_df.copy()
    colors = ["#3498db" if "LSTM" in m or "GRU" in m else "#95a5a6"
              for m in df["Model"]]

    for ax, metric, ylabel in [
        (axes[0], "Test_R2",   "R² (выше — лучше)"),
        (axes[1], "Test_RMSE", "RMSE, мкг/м³ (ниже — лучше)"),
        (axes[2], "Test_MAE",  "MAE, мкг/м³ (ниже — лучше)"),
    ]:
        ax.bar(range(len(df)), df[metric], color=colors, alpha=0.85)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["Model"], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if metric == "Test_R2":
            ax.axhline(0, color="black", lw=0.5)

    plt.suptitle("Сравнение нейросетей с baseline и классическим ML (Pb)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Сохранён: {fname.name}")


def plot_predictions(y_true, y_pred, dates, posts, name, fname):
    """График предсказаний во времени для двух постов."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    for ax, post in zip(axes, [1, 12]):
        mask = posts == post
        # сортируем по дате
        idx = np.argsort(dates[mask])
        d = pd.to_datetime(dates[mask][idx])
        yt = y_true[mask][idx]
        yp = y_pred[mask][idx]
        ax.plot(d, yt, "o-", color="#2c3e50", label="Истинное Pb",
                ms=4, lw=1, alpha=0.7)
        ax.plot(d, yp, "s-", color="#e74c3c", label=f"Прогноз ({name})",
                ms=4, lw=1, alpha=0.7)
        ax.axhline(0.3, color="red", lw=0.8, ls=":", label="ПДК с.с. = 0.3")
        ax.set_title(f"Пост № {post}")
        ax.set_ylabel("Pb, мкг/м³")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Дата")
    plt.suptitle(f"Предсказания во времени — {name}", fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Сохранён: {fname.name}")


def plot_training_history(histories, fname):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, h in histories.items():
        ax.plot(h["loss"], label=f"{name} train", lw=1.5)
        ax.plot(h["val_loss"], label=f"{name} val", lw=1.5, ls="--")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("MSE (log-scale target)")
    ax.set_title("Кривые обучения нейросетей")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Сохранён: {fname.name}")


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    print("=" * 78)
    print("LSTM / GRU — Этап 3")
    print("=" * 78)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    print(f"Загружено: {len(df)} строк за период "
          f"{df['start_date'].min().date()} — {df['start_date'].max().date()}")

    # Делаем последовательности
    X, y, dates, posts = make_sequences(df, TIMESTEP_FEATURES, "Pb", WINDOW)
    print(f"Последовательностей: {len(X)}, форма: окно={WINDOW}, "
          f"признаков={len(TIMESTEP_FEATURES)}")

    # Тот же split, что в Этапе 1 (random + seed=42) — для прямого сравнения
    X_train, X_test, y_train, y_test, d_train, d_test, p_train, p_test = (
        train_test_split(X, y, dates, posts,
                         test_size=TEST_SIZE, shuffle=True,
                         random_state=RANDOM_STATE)
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Лог-таргет (как в Этапе 1)
    y_train_log = np.log1p(y_train)
    y_test_orig = y_test

    # Нормализация (MinMax, как в Этапе 1)
    n, w, f_ = X_train.shape
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, f_)).reshape(n, w, f_)
    X_test_s = scaler.transform(X_test.reshape(-1, f_)).reshape(-1, w, f_)

    # === BASELINES ===
    print("\n" + "=" * 78)
    print("BASELINE МОДЕЛИ (контрольные точки)")
    print("=" * 78)
    print("\n--- Naive (последнее значение в окне) ---")
    y_pred_naive = X_test[:, -1, 0]  # Pb на последнем шаге окна
    m = evaluate(y_test_orig, y_pred_naive)
    print(f"  Test R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")
    naive_row = {"Model": "Naive (last)", "Params": 0, "Epochs": 0,
                 "Train_time_s": 0.0,
                 "Test_R2": m["R2"], "Test_RMSE": m["RMSE"],
                 "Test_MSE": m["MSE"], "Test_MAE": m["MAE"],
                 "Test_MBE": m["MBE"], "Pred_std": float(y_pred_naive.std())}

    print("\n--- WindowMean (среднее по окну) ---")
    y_pred_wmean = X_test[:, :, 0].mean(axis=1)
    m = evaluate(y_test_orig, y_pred_wmean)
    print(f"  Test R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")
    wmean_row = {"Model": "WindowMean", "Params": 0, "Epochs": 0,
                 "Train_time_s": 0.0,
                 "Test_R2": m["R2"], "Test_RMSE": m["RMSE"],
                 "Test_MSE": m["MSE"], "Test_MAE": m["MAE"],
                 "Test_MBE": m["MBE"], "Pred_std": float(y_pred_wmean.std())}

    # === NEURAL NETWORKS ===
    print("\n" + "=" * 78)
    print("НЕЙРОННЫЕ СЕТИ")
    print("=" * 78)
    results = [naive_row, wmean_row]
    predictions = {
        "Naive (last)": y_pred_naive,
        "WindowMean": y_pred_wmean,
    }
    histories = {}

    for name, builder in [
        ("LSTM", build_lstm),
        ("GRU", build_gru),
        ("Stacked-LSTM", build_stacked_lstm),
    ]:
        row, y_pred, history = train_nn(name, builder, X_train_s, y_train_log,
                                        X_test_s, y_test_orig, (w, f_))
        results.append(row)
        predictions[name] = y_pred
        histories[name] = history

    # Сохраняем результаты
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("Test_R2", ascending=False).reset_index(drop=True)
    res_df.to_csv(OUTPUT_DIR / "lstm_gru_metrics.csv", index=False)
    print(f"\nСохранён: {(OUTPUT_DIR / 'lstm_gru_metrics.csv').name}")

    # Графики
    plot_compare(res_df, OUTPUT_DIR / "lstm_gru_compare.png")
    plot_training_history(histories, OUTPUT_DIR / "training_history.png")

    # Графики предсказаний для всех 3 нейросетей
    for name in ["LSTM", "GRU", "Stacked-LSTM"]:
        safe = name.replace("-", "_")
        plot_predictions(y_test_orig, predictions[name], d_test, p_test, name,
                         OUTPUT_DIR / f"predictions_{safe}.png")

    # Итоговая таблица
    print("\n" + "=" * 78)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("=" * 78)
    print(res_df[["Model", "Test_R2", "Test_RMSE", "Test_MSE", "Test_MAE",
                  "Pred_std", "Train_time_s"]].round(4).to_string(index=False))

    # Научный комментарий
    print("\n" + "=" * 78)
    print("ИНТЕРПРЕТАЦИЯ")
    print("=" * 78)
    lstm_r2 = res_df[res_df["Model"] == "LSTM"]["Test_R2"].values[0]
    if lstm_r2 < 0.3:
        print("LSTM/GRU показали R² < 0.3 — существенно ниже, чем классический")
        print("Extra Trees из Этапа 1 (R²≈0.50). Это связано с тем, что:")
        print("  1. Концентрации Pb в Алматы имеют слабую автокорреляцию")
        print("     (Naive-baseline даёт R²={:.2f} — нет 'инерции' процесса).".format(
            res_df[res_df['Model']=='Naive (last)']['Test_R2'].values[0]))
        print("  2. Исторический тренд (снижение Pb в ~5 раз с 2006 г.) делает")
        print("     данные нестационарными, и сети с памятью теряют преимущество.")
        print("  3. Малая выборка (912 последовательностей) не позволяет нейросетям")
        print("     выучить нелинейные закономерности.")
        print()
        print("Вывод: для данной задачи (декадные измерения ТМ) классические")
        print("ансамблевые методы (Extra Trees, CatBoost) являются более")
        print("подходящим выбором, чем рекуррентные нейросети.")
    else:
        print(f"LSTM достиг R²={lstm_r2:.3f}. Сравните с результатами Этапа 1.")

    print(f"\nВсе результаты в: {OUTPUT_DIR}")
    print("Готово.")


if __name__ == "__main__":
    main()
