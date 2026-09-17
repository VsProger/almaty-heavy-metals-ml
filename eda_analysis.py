"""
====================================================================
EDA — РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ
====================================================================

Этап 0 (по сути предшествует моделированию). Создаёт визуальный
обзор данных по тяжёлым металлам в воздухе Алматы (2006-2020):

  • Распределения концентраций (гистограммы в лог-шкале)
  • Временные ряды для всех 6 металлов и двух постов
  • Многолетние тренды (среднее/медиана по годам)
  • Корреляционная матрица между металлами
  • Сезонная изменчивость (boxplot по сезонам)
  • Различия между постами (boxplot)
  • Динамика метеопараметров (температура, скорость ветра)
  • Циклограмма по месяцам (heatmap «год × месяц»)

Все графики сохраняются как PNG для использования в диссертации,
а статистики — как CSV для дальнейшего использования.

ИНСТРУКЦИЯ ПО ЗАПУСКУ
=====================

1) Установите библиотеки (если ещё не установлены):

       pip install pandas numpy matplotlib

2) Скрипт ищет ml_ready_dataset.csv в тех же местах, что и Этап 1:
       ./data/ml_ready_dataset.csv
       ./ml_ready_dataset.csv
       ../data/ml_ready_dataset.csv

3) Запустите:

       python eda_analysis.py

4) Результаты сохранятся в папку eda_results/:
     - 01_distributions.png    — гистограммы концентраций ТМ
     - 02_timeseries.png       — временные ряды по постам
     - 03_trends.png           — многолетние тренды (среднее/медиана)
     - 04_correlation.png      — корреляционная матрица
     - 05_seasonal.png         — сезонная изменчивость
     - 06_posts.png            — сравнение постов
     - 07_meteorology.png      — температура и скорость ветра
     - 08_heatmap.png          — годовая циклограмма Pb
     - descriptive_stats.csv   — таблица описательной статистики
     - yearly_means.csv        — средние и медианы по годам
     - seasonal_means.csv      — средние по сезонам
     - correlation_matrix.csv  — числовая корреляционная матрица

Время работы: 15-30 секунд.
====================================================================
"""

import os
import sys
import warnings
from pathlib import Path

# Проверка зависимостей
REQUIRED = {
    "pandas": "pandas",
    "numpy": "numpy",
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
    print(f"\nУстановите: pip install {' '.join(missing)}")
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

CANDIDATE_PATHS = [
    SCRIPT_DIR / "data" / "ml_ready_dataset.csv",
    SCRIPT_DIR / "ml_ready_dataset.csv",
    SCRIPT_DIR.parent / "data" / "ml_ready_dataset.csv",
]
INPUT_CSV = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])
OUTPUT_DIR = SCRIPT_DIR / "eda_results"

METALS = ["Cd", "Pb", "As", "Cr", "Cu", "Ni"]
# ПДК среднесуточные (мкг/м³) для атмосферного воздуха
PDK = {"Cd": 0.3, "Pb": 0.3, "As": 0.3, "Cr": 1.5, "Cu": 2.0, "Ni": 1.0}

# Стиль научных графиков
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 130,
})


# --------------------------------------------------------------------
# Загрузка данных
# --------------------------------------------------------------------
def load_data():
    if not INPUT_CSV.exists():
        print("ОШИБКА: не найден ml_ready_dataset.csv")
        print("Скрипт искал в следующих местах:")
        for p in CANDIDATE_PATHS:
            print(f"  - {p}")
        sys.exit(1)
    print(f"Использую датасет: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, parse_dates=["start_date"])
    df["year"] = df["start_date"].dt.year
    df["month"] = df["start_date"].dt.month
    df["season"] = df["month"].map({
        12: "Зима", 1: "Зима", 2: "Зима",
        3: "Весна", 4: "Весна", 5: "Весна",
        6: "Лето", 7: "Лето", 8: "Лето",
        9: "Осень", 10: "Осень", 11: "Осень",
    })
    print(f"Загружено: {len(df)} строк, период "
          f"{df['start_date'].min().date()} — {df['start_date'].max().date()}")
    return df


# --------------------------------------------------------------------
# 1. Описательная статистика → CSV
# --------------------------------------------------------------------
def stats_descriptive(df):
    rows = []
    for m in METALS:
        s = df[m].dropna()
        rows.append({
            "Металл": m,
            "n": len(s),
            "Среднее": s.mean(),
            "Медиана": s.median(),
            "Стд_откл": s.std(),
            "Минимум": s.min(),
            "Максимум": s.max(),
            "ПДК_сс": PDK[m],
            "Превышений_ПДК": int((s > PDK[m]).sum()),
            "Доля_н/о_%": 100 * (s == 0).mean(),
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUTPUT_DIR / "descriptive_stats.csv", index=False, encoding="utf-8")
    print("CSV: descriptive_stats.csv")
    return res


def stats_yearly(df):
    rows = []
    for m in METALS:
        g = df.groupby("year")[m].agg(["count", "mean", "median", "max"])
        g["metal"] = m
        rows.append(g.reset_index())
    res = pd.concat(rows, ignore_index=True)
    res = res[["year", "metal", "count", "mean", "median", "max"]]
    res.to_csv(OUTPUT_DIR / "yearly_means.csv", index=False, encoding="utf-8")
    print("CSV: yearly_means.csv")
    return res


def stats_seasonal(df):
    rows = []
    for m in METALS:
        g = df.groupby("season")[m].agg(["count", "mean", "median", "max"])
        g["metal"] = m
        rows.append(g.reset_index())
    res = pd.concat(rows, ignore_index=True)
    res = res[["season", "metal", "count", "mean", "median", "max"]]
    res.to_csv(OUTPUT_DIR / "seasonal_means.csv", index=False, encoding="utf-8")
    print("CSV: seasonal_means.csv")
    return res


def stats_correlation(df):
    corr = df[METALS].corr()
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv", encoding="utf-8")
    print("CSV: correlation_matrix.csv")
    return corr


# --------------------------------------------------------------------
# Графики
# --------------------------------------------------------------------
def plot_distributions(df):
    """01: Гистограммы распределений 6 металлов в лог-шкале."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, m in zip(axes.flat, METALS):
        s = df[m].dropna()
        s_nonzero = s[s > 0]
        ax.hist(s_nonzero, bins=40, color="#3498db", alpha=0.75,
                edgecolor="black", lw=0.5)
        ax.set_yscale("log")
        ax.axvline(PDK[m], color="red", ls="--", lw=1.5,
                   label=f"ПДК с.с. = {PDK[m]}")
        ax.set_title(f"{m}, n={len(s)} (ненулевых: {len(s_nonzero)})")
        ax.set_xlabel("Концентрация, мкг/м³")
        ax.set_ylabel("Частота (log)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    plt.suptitle("Распределение концентраций тяжёлых металлов в воздухе Алматы (2006–2020)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_distributions.png", bbox_inches="tight")
    plt.close()
    print("PNG: 01_distributions.png")


def plot_timeseries(df):
    """02: Временные ряды концентраций по постам."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for ax, m in zip(axes.flat, METALS):
        for post, color in [(1, "#3498db"), (12, "#e74c3c")]:
            sub = df[df["post"] == post].sort_values("start_date")
            ax.plot(sub["start_date"], sub[m], "o-", color=color,
                    ms=2.5, lw=0.6, alpha=0.65, label=f"ПНЗ № {post}")
        ax.axhline(PDK[m], color="black", ls=":", lw=1.2,
                   label=f"ПДК с.с. = {PDK[m]}")
        ax.set_title(f"{m}, мкг/м³")
        ax.set_ylabel("Концентрация")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("Дата")
    plt.suptitle("Временные ряды концентраций тяжёлых металлов по постам наблюдения",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_timeseries.png", bbox_inches="tight")
    plt.close()
    print("PNG: 02_timeseries.png")


def plot_trends(df):
    """03: Многолетние тренды (среднее, медиана по годам)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, m in zip(axes.flat, METALS):
        annual = df.groupby("year")[m].agg(["mean", "median"])
        ax.plot(annual.index, annual["mean"], "o-", color="#e74c3c",
                label="Среднее", lw=2, ms=5)
        ax.plot(annual.index, annual["median"], "s--", color="#3498db",
                label="Медиана", lw=1.5, ms=4)
        ax.axhline(PDK[m], color="black", ls=":", lw=1, alpha=0.5,
                   label=f"ПДК={PDK[m]}")
        ax.set_title(f"{m}")
        ax.set_xlabel("Год")
        ax.set_ylabel("Концентрация, мкг/м³")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    plt.suptitle("Многолетние тренды концентраций ТМ (среднегодовые значения)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_trends.png", bbox_inches="tight")
    plt.close()
    print("PNG: 03_trends.png")


def plot_correlation(corr):
    """04: Корреляционная матрица."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    n = len(METALS)
    ax.set_xticks(range(n))
    ax.set_xticklabels(METALS)
    ax.set_yticks(range(n))
    ax.set_yticklabels(METALS)
    for i in range(n):
        for j in range(n):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=11,
                    fontweight="bold")
    plt.colorbar(im, label="Коэффициент корреляции Пирсона")
    plt.title("Корреляционная матрица концентраций ТМ\n"
              "(совместные источники → высокая корреляция)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_correlation.png", bbox_inches="tight")
    plt.close()
    print("PNG: 04_correlation.png")


def plot_seasonal(df):
    """05: Сезонная изменчивость (boxplot)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    season_order = ["Зима", "Весна", "Лето", "Осень"]
    season_colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    for ax, m in zip(axes.flat, METALS):
        data = [df[df["season"] == s][m].dropna() for s in season_order]
        bp = ax.boxplot(data, labels=season_order, showfliers=False,
                        patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], season_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(m)
        ax.set_ylabel("Концентрация, мкг/м³")
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Сезонная изменчивость концентраций ТМ",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_seasonal.png", bbox_inches="tight")
    plt.close()
    print("PNG: 05_seasonal.png")


def plot_posts(df):
    """06: Сравнение концентраций между постами."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, m in zip(axes.flat, METALS):
        data = [df[df["post"] == p][m].dropna() for p in [1, 12]]
        bp = ax.boxplot(data, labels=["ПНЗ № 1\n(Амангельды)",
                                       "ПНЗ № 12\n(Райымбека)"],
                        showfliers=False, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], ["#3498db", "#e74c3c"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        # Подпись с медианами
        m1 = data[0].median(); m2 = data[1].median()
        ratio = m2 / m1 if m1 > 0 else np.nan
        title = f"{m}"
        if not np.isnan(ratio):
            title += f"\n(отношение медиан: × {ratio:.1f})"
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Концентрация, мкг/м³")
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Сравнение концентраций ТМ между постами наблюдения",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_posts.png", bbox_inches="tight")
    plt.close()
    print("PNG: 06_posts.png")


def plot_meteorology(df):
    """07: Метеопараметры — температура и скорость ветра по сезонам."""
    if "temp_mean" not in df.columns or df["temp_mean"].isna().all():
        print("WARN: климатические данные отсутствуют — пропускаю 07_meteorology")
        return
    df_clim = df.dropna(subset=["temp_mean"])
    if len(df_clim) == 0:
        print("WARN: нет записей с climate features")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    season_order = ["Зима", "Весна", "Лето", "Осень"]
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

    # Температура
    data_t = [df_clim[df_clim["season"] == s]["temp_mean"].dropna() for s in season_order]
    bp = axes[0].boxplot(data_t, labels=season_order, patch_artist=True,
                          showfliers=True, widths=0.55)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    axes[0].set_title("Средняя температура воздуха")
    axes[0].set_ylabel("Температура, °C")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].axhline(0, color="black", lw=0.5, ls="--")

    # Скорость ветра
    data_w = [df_clim[df_clim["season"] == s]["wind_speed_mean"].dropna() for s in season_order]
    bp = axes[1].boxplot(data_w, labels=season_order, patch_artist=True,
                          showfliers=True, widths=0.55)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    axes[1].set_title("Средняя скорость ветра")
    axes[1].set_ylabel("Скорость ветра, м/с")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle(f"Сезонная динамика метеопараметров (n={len(df_clim)} наблюдений)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_meteorology.png", bbox_inches="tight")
    plt.close()
    print("PNG: 07_meteorology.png")


def plot_heatmap(df):
    """08: Годовая циклограмма Pb (heatmap год × месяц)."""
    pivot = df.pivot_table(index="year", columns="month", values="Pb", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    # Подписи значений
    for i in range(len(pivot.index)):
        for j in range(12):
            v = pivot.iloc[i, j]
            if pd.notna(v):
                color = "white" if v > pivot.values[~np.isnan(pivot.values)].mean() else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
    plt.colorbar(im, label="Среднемесячная концентрация Pb, мкг/м³")
    plt.title("Годовая циклограмма свинца Pb\n(тёплые цвета — высокие концентрации)")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Год")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_heatmap.png", bbox_inches="tight")
    plt.close()
    print("PNG: 08_heatmap.png")


# --------------------------------------------------------------------
# Текстовый отчёт в консоль
# --------------------------------------------------------------------
def print_report(stats_df, corr):
    print("\n" + "=" * 78)
    print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
    print("=" * 78)
    print(stats_df.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("КЛЮЧЕВЫЕ НАХОДКИ")
    print("=" * 78)

    # Кто превышает ПДК
    exceed = stats_df[stats_df["Превышений_ПДК"] > 0]
    if len(exceed) > 0:
        print("\nПревышения ПДК зарегистрированы для:")
        for _, r in exceed.iterrows():
            print(f"  • {r['Металл']}: {r['Превышений_ПДК']} случаев из {r['n']} "
                  f"({100*r['Превышений_ПДК']/r['n']:.1f}%)")
    else:
        print("\nПревышений ПДК с.с. не зафиксировано ни по одному металлу.")

    # Сильные корреляции
    print("\nСильные корреляции между металлами (|r| > 0,5):")
    n = len(METALS)
    found = False
    for i in range(n):
        for j in range(i+1, n):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                found = True
                interp = "общий источник" if r > 0 else "обратная связь"
                print(f"  • {METALS[i]} ↔ {METALS[j]}: r = {r:+.2f} ({interp})")
    if not found:
        print("  Сильных корреляций (|r| > 0,5) не найдено.")


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    print("=" * 78)
    print("EDA — РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")
    print("=" * 78)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data()

    print("\n--- Расчёт описательных статистик ---")
    stats_df = stats_descriptive(df)
    yearly = stats_yearly(df)
    seasonal = stats_seasonal(df)
    corr = stats_correlation(df)

    print("\n--- Построение графиков ---")
    plot_distributions(df)
    plot_timeseries(df)
    plot_trends(df)
    plot_correlation(corr)
    plot_seasonal(df)
    plot_posts(df)
    plot_meteorology(df)
    plot_heatmap(df)

    print_report(stats_df, corr)

    print(f"\nВсе результаты сохранены в: {OUTPUT_DIR}")
    print("Готово.")


if __name__ == "__main__":
    main()
