"""
====================================================================
ОЧИСТКА ДАННЫХ С НУЛЯ — Алматы, тяжёлые металлы (2006-2020)
====================================================================

Скрипт парсит 32 исходных Excel-файла Казгидромета и формирует
готовый к моделированию датасет ml_ready_dataset.csv.

ЭТАПЫ ОБРАБОТКИ
---------------
1. ПАРСИНГ. Чтение всех .xls/.xlsx файлов из папки. Для каждого файла
   определяется тип листов (ТМ-таблица / климат / прочее). Из ТМ-таблиц
   извлекаются строки с порядковым номером в первой колонке.

2. НОРМАЛИЗАЦИЯ ПОСТОВ. Унификация различных написаний:
   "ПНЗ №1", "ПНЗ№1", "№1" → 1
   "ПНЗ №12", "ПНЗ№12", "№12" → 12

3. ПАРСИНГ ДАТ. Парсер декад с устойчивостью к 5 типам опечаток:
   - запятая вместо точки   ("01,12-02.12" → 01.12 .. 02.12)
   - двойные точки          ("02.11-04..11" → 02.11 .. 04.11)
   - дефис вместо точки     ("22.11-29-11" → 22.11 .. 29.11)
   - три цифры в дне        ("04.125-06.12" → 04.12 .. 06.12)
   - несуществующая дата    ("31.04" → 30.04, последний день месяца)

4. ПАРСИНГ ЗНАЧЕНИЙ. "н/о" → 0 + флаг <metal>_censored = True.
   Все числовые значения — приведение к float (запятая → точка).

5. ВЫБРОСЫ. Грубая фильтрация явно невозможных концентраций (>100 мкг/м³).
   Для г. Алматы такие значения физически невозможны (потерянная запятая
   при вводе данных). Конкретный случай — Cu=564 мкг/м³ 27.11.2006.

6. ДЕДУПЛИКАЦИЯ. Многие файлы пересекаются по содержимому (например,
   "2018_свод 1 кв." содержит те же декады, что "2018+январь-март").
   Принцип: при совпадении (post, start_date, end_date) оставляем строку
   с наибольшим числом непустых значений металлов.

7. ДОБАВЛЕНИЕ ПРИЗНАКОВ:
   - циклические признаки сезонности (sin/cos месяца и дня года)
   - временные лаги для каждого металла (lag1, lag2, скользящее среднее)
   - ПДК-колонки (значение, отношение, превышение) для каждого металла

8. СОХРАНЕНИЕ. Итоговый CSV `ml_ready_dataset.csv` (~924 строки × 89 колонок).

ИСПОЛЬЗОВАНИЕ
=============

Положите все 32 Excel-файла в подпапку datasets/ рядом со скриптом.
Например, структура может быть такой:

    my_project/
    ├── clean_from_scratch.py
    └── data/
        └── datasets/
            ├── 2006-2009.xlsx
            ├── 2010.xls
            ├── ...
            └── ТМ+климат_февраль 2020.xlsx

Установите зависимости:
    pip install pandas numpy openpyxl xlrd

Запустите:
    python clean_from_scratch.py

Создаст:
    data/ml_ready_dataset.csv  — итоговый чистый датасет

Время работы: 30-60 секунд.
====================================================================
"""

import os
import sys
import re
import datetime
import warnings
from calendar import monthrange
from pathlib import Path

# --------------------------------------------------------------------
# Проверка зависимостей
# --------------------------------------------------------------------
REQUIRED = {
    "pandas":   "pandas",
    "numpy":    "numpy",
    "openpyxl": "openpyxl",  # для .xlsx
    "xlrd":     "xlrd",      # для .xls (старый формат)
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
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------
METALS = ["Cd", "Pb", "As", "Cr", "Cu", "Ni"]
# ПДК среднесуточные, мкг/м³ (ГН 2.1.6.3492-17 для воздуха)
PDK_SS = {"Cd": 0.3, "Pb": 0.3, "As": 0.3, "Cr": 1.5, "Cu": 2.0, "Ni": 1.0}
# Замена для "нет данных вообще" (не путать с "н/о = ниже LOD")
NO_DATA_VALUE = 1e-9
# Физически невозможные значения концентраций ТМ в воздухе (мкг/м³)
OUTLIER_THRESHOLD = 100

SCRIPT_DIR = Path(__file__).resolve().parent

# Где искать папку datasets/ с Excel-файлами
CANDIDATE_DATASETS_DIRS = [
    SCRIPT_DIR / "data" / "datasets",
    SCRIPT_DIR / "datasets",
    SCRIPT_DIR.parent / "data" / "datasets",
]
DATASETS_DIR = next((p for p in CANDIDATE_DATASETS_DIRS if p.exists()), None)
# Куда сохранить итоговый файл
OUTPUT_CSV = (DATASETS_DIR.parent if DATASETS_DIR else SCRIPT_DIR) / "ml_ready_dataset.csv"


# --------------------------------------------------------------------
# Парсеры
# --------------------------------------------------------------------
def has_tm_header(df):
    """Находит строку заголовка ТМ-таблицы (Декада + Cd + Pb)."""
    for i in range(min(7, len(df))):
        vals = [str(v).lower() if pd.notna(v) else "" for v in df.iloc[i].tolist()]
        joined = " | ".join(vals)
        if "декада" in joined and "cd" in joined:
            return i
    return None


def detect_year(fname, sheet):
    """Год — из имени листа (2006-2013), затем из имени файла."""
    s = str(sheet).strip()
    if re.fullmatch(r"20\d{2}", s):
        return int(s)
    m = re.search(r"\b(20\d{2})\b", fname)
    if m:
        return int(m.group(1))
    # Файлы за 2019 не имеют года в имени, но имеют его в названии
    if "2019" in fname:
        return 2019
    if "2018" in fname:
        return 2018
    return None


DATE_PAT = re.compile(r"(\d{1,2})\.(\d{1,2})")


def parse_dekada(s, year):
    """
    Парсер декады с устойчивостью к опечаткам:
      "09.01-11.01"   → (2018-01-09, 2018-01-11)
      "01,12 - 02.12" → (2018-12-01, 2018-12-02)
      "22.11-29-11"   → (2018-11-22, 2018-11-29)
      "02.11-04..11"  → (2018-11-02, 2018-11-04)
      "04.125-06.12"  → (2018-12-04, 2018-12-06)
      "31.04-10.05"   → (2018-04-30, 2018-05-10)
    Возвращает (start_date, end_date) или (None, None) если не парсится.
    """
    if not s or pd.isna(s) or year is None:
        return None, None
    s = str(s).strip()

    # 1. Запятая → точка
    s = s.replace(",", ".")
    # 2. Двойные точки → одна точка
    s = re.sub(r"\.\.+", ".", s)
    # 3. Три цифры в дне (04.125 → 04.12)
    s = re.sub(r"(\d{2})\.(\d{2})\d", r"\1.\2", s)
    # 4. Дефис вместо точки между днём и месяцем (22.11-29-11 → 22.11-29.11)
    s = re.sub(r"(\d{2})-(\d{2})$", r"\1.\2", s)
    s = re.sub(r"(\d{2})-(\d{2})(?=[^.\d]|$)", r"\1.\2", s)

    # Находим 2 даты вида dd.mm
    parts = DATE_PAT.findall(s)
    if len(parts) < 2:
        return None, None
    (d1d, d1m), (d2d, d2m) = parts[0], parts[1]

    # Начало декады
    try:
        d1 = datetime.date(int(year), int(d1m), int(d1d))
    except ValueError:
        # 31.04 → последний день месяца
        try:
            d1 = datetime.date(int(year), int(d1m),
                               monthrange(int(year), int(d1m))[1])
        except Exception:
            return None, None

    # Конец декады (с учётом перехода через Новый год для декабря)
    try:
        end_year = int(year)
        if int(d2m) < int(d1m):
            end_year += 1
        d2 = datetime.date(end_year, int(d2m), int(d2d))
    except ValueError:
        try:
            end_year = int(year)
            if int(d2m) < int(d1m):
                end_year += 1
            d2 = datetime.date(end_year, int(d2m),
                               monthrange(end_year, int(d2m))[1])
        except Exception:
            return d1, None
    return d1, d2


def normalize_post(p):
    """ПНЗ №1, ПНЗ№1, №1 → 1; ПНЗ №12, ПНЗ№12, №12 → 12."""
    s = str(p).strip()
    digits = re.findall(r"\d+", s)
    if not digits:
        return None
    return int(digits[0])


def parse_value(v):
    """
    Превращает значение из ячейки в (float, censored_flag).
    'н/о' и пустота при наличии измерения → (0.0, True).
    Пустота вообще → (None, False).
    """
    if v is None or pd.isna(v):
        return None, False
    s = str(v).strip().lower()
    if s in ("н/о", "н\\о", "но", "нет", "-", ""):
        return 0.0, True
    s = s.replace(",", ".").replace(" ", "")
    try:
        return float(s), False
    except ValueError:
        return None, False


# --------------------------------------------------------------------
# Сбор данных
# --------------------------------------------------------------------
def collect_raw_records(datasets_dir):
    """Парсит все ТМ-таблицы из всех файлов в папке."""
    files = sorted([f for f in os.listdir(datasets_dir)
                    if f.lower().endswith((".xls", ".xlsx"))])
    print(f"Найдено файлов: {len(files)}")

    records = []
    stats_file_tables = 0

    for fname in files:
        full_path = os.path.join(datasets_dir, fname)
        try:
            xl = pd.ExcelFile(full_path)
        except Exception as e:
            print(f"  ! Пропуск {fname}: {e}")
            continue

        for sheet in xl.sheet_names:
            try:
                df_sh = pd.read_excel(full_path, sheet_name=sheet, header=None)
            except Exception:
                continue
            if len(df_sh) < 4:
                continue
            hr = has_tm_header(df_sh)
            if hr is None:
                continue

            year = detect_year(fname, sheet)
            if year is None:
                print(f"  ! Не определён год для {fname}/{sheet}")
                continue

            n_cols = df_sh.shape[1]
            n_rows_added = 0
            for i in range(hr + 1, len(df_sh)):
                row = df_sh.iloc[i]
                # Первая ячейка должна быть номером строки
                try:
                    int(float(str(row.iloc[0]).strip()))
                except (ValueError, TypeError):
                    continue
                post = normalize_post(row.iloc[1])
                if post not in (1, 12):
                    continue
                dekada = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
                sd, ed = parse_dekada(dekada, year)
                if sd is None:
                    continue

                rec = {
                    "file": fname, "sheet": sheet, "year": year,
                    "post": post, "dekada_raw": dekada,
                    "start_date": sd, "end_date": ed,
                }
                for col_idx, metal in enumerate(METALS, start=3):
                    if col_idx >= n_cols:
                        rec[metal] = None
                        rec[f"{metal}_censored"] = False
                    else:
                        val, cens = parse_value(row.iloc[col_idx])
                        rec[metal] = val
                        rec[f"{metal}_censored"] = cens
                records.append(rec)
                n_rows_added += 1

            if n_rows_added > 0:
                stats_file_tables += 1
                print(f"  {fname[:50]:52s} | {sheet[:20]:22s} | "
                      f"{n_rows_added:>3} строк")

    print(f"\nОбработано ТМ-таблиц: {stats_file_tables}")
    print(f"Сырых записей собрано: {len(records)}")
    return pd.DataFrame(records)


# --------------------------------------------------------------------
# Очистка
# --------------------------------------------------------------------
def remove_outliers(df):
    """Маркируем явные физически невозможные значения как None."""
    print(f"\n--- Удаление выбросов (>{OUTLIER_THRESHOLD} мкг/м³) ---")
    for m in METALS:
        mask = df[m] > OUTLIER_THRESHOLD
        n = int(mask.sum())
        if n > 0:
            print(f"  {m}: {n} значений → None")
            df.loc[mask, m] = None
            df.loc[mask, f"{m}_censored"] = False
    return df


def deduplicate(df):
    """
    Дедупликация по (post, start_date, end_date).
    Принцип: из дубликатов оставляем строку с наибольшим числом
    непустых значений металлов.
    """
    print("\n--- Дедупликация ---")
    df["completeness"] = df[METALS].notna().sum(axis=1)
    before = len(df)
    df = df.sort_values("completeness", ascending=False).drop_duplicates(
        subset=["post", "start_date", "end_date"], keep="first"
    )
    after = len(df)
    print(f"  Было: {before}, Стало: {after} (удалено {before - after} дубликатов)")
    df = df.drop(columns=["completeness"])
    return df.sort_values(["post", "start_date"]).reset_index(drop=True)


# --------------------------------------------------------------------
# Дополнительные признаки
# --------------------------------------------------------------------
def add_calendar_features(df):
    """Циклические признаки сезонности."""
    print("\n--- Циклические признаки сезонности ---")
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["year"] = df["start_date"].dt.year
    df["month"] = df["start_date"].dt.month
    df["dayofyear"] = df["start_date"].dt.dayofyear

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"]   = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    print(f"  Добавлены: month_sin/cos, doy_sin/cos")
    return df


def add_lags(df):
    """Лаги по каждому посту для всех металлов."""
    print("\n--- Временные лаги ---")
    df = df.sort_values(["post", "start_date"]).reset_index(drop=True)
    for m in METALS:
        df[f"{m}_lag1"]       = df.groupby("post")[m].shift(1)
        df[f"{m}_lag2"]       = df.groupby("post")[m].shift(2)
        df[f"{m}_roll3_mean"] = (df.groupby("post")[m]
                                   .shift(1)
                                   .rolling(3, min_periods=1).mean()
                                   .reset_index(drop=True))
    print(f"  Для каждого металла: <m>_lag1, <m>_lag2, <m>_roll3_mean")
    return df


def replace_no_data(df):
    """NaN (не измерялось) → 1e-9. 'н/о' (=0) остаётся как есть."""
    print(f"\n--- Замена NaN → {NO_DATA_VALUE} ---")
    print("(NaN = металл не измерялся вообще; н/о остаётся = 0)")
    for m in METALS:
        n = int(df[m].isna().sum())
        if n > 0:
            df[m] = df[m].fillna(NO_DATA_VALUE)
            print(f"  {m}: {n:>4} → 1e-9")
    # Также лаги (они могут быть NaN на началах рядов)
    for m in METALS:
        for lag_col in [f"{m}_lag1", f"{m}_lag2", f"{m}_roll3_mean"]:
            if lag_col in df.columns:
                df[lag_col] = df.groupby("post")[lag_col].transform(
                    lambda s: s.bfill().fillna(NO_DATA_VALUE)
                )
    return df


def add_pdk_features(df):
    """Добавляем колонки ПДК."""
    print("\n--- Добавление ПДК-колонок ---")
    for m in METALS:
        pdk = PDK_SS[m]
        df[f"{m}_PDK_ss"]       = pdk
        df[f"{m}_PDK_ratio"]    = df[m] / pdk
        df[f"{m}_PDK_exceeded"] = df[m] > pdk
        n_exc = int(df[f"{m}_PDK_exceeded"].sum())
        max_r = df[f"{m}_PDK_ratio"].max()
        print(f"  {m}: ПДК={pdk}, превышений: {n_exc} ({100*n_exc/len(df):.1f}%), "
              f"макс. отношение: × {max_r:.2f}")
    return df


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    print("=" * 72)
    print("ОЧИСТКА ДАННЫХ С НУЛЯ — Алматы, ТМ 2006-2020")
    print("=" * 72)

    if DATASETS_DIR is None:
        print("ОШИБКА: не найдена папка datasets/. Искал в:")
        for p in CANDIDATE_DATASETS_DIRS:
            print(f"  - {p}")
        print("\nПоложите 32 исходных Excel-файла в подпапку data/datasets/")
        sys.exit(1)

    print(f"\nИсходные файлы: {DATASETS_DIR}")
    print(f"Результат:      {OUTPUT_CSV}")

    print("\n" + "=" * 72)
    print("ШАГ 1. Парсинг исходных Excel-файлов")
    print("=" * 72)
    df = collect_raw_records(DATASETS_DIR)

    print("\n" + "=" * 72)
    print("ШАГ 2. Очистка от выбросов и дубликатов")
    print("=" * 72)
    df = remove_outliers(df)
    df = deduplicate(df)

    print("\n" + "=" * 72)
    print("ШАГ 3. Добавление производных признаков")
    print("=" * 72)
    df = add_calendar_features(df)
    df = add_lags(df)

    print("\n" + "=" * 72)
    print("ШАГ 4. Обработка пропусков и ПДК")
    print("=" * 72)
    df = replace_no_data(df)
    df = add_pdk_features(df)

    # Финальный порядок колонок: id-поля → металлы → calendar → lags → ПДК
    print("\n--- Сохранение ---")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Финальный размер: {len(df)} строк × {len(df.columns)} колонок")
    print(f"Сохранено: {OUTPUT_CSV}")

    print("\n" + "=" * 72)
    print("СВОДКА")
    print("=" * 72)
    print(f"\nПо постам:")
    print(df["post"].value_counts().sort_index().to_string())
    print(f"\nПо годам:")
    print(df["year"].value_counts().sort_index().to_string())
    print(f"\nПревышения ПДК (по металлам):")
    for m in METALS:
        n = int(df[f"{m}_PDK_exceeded"].sum())
        if n > 0:
            print(f"  {m}: {n} случаев ({100*n/len(df):.1f}%)")

    print("\nГотово.")


if __name__ == "__main__":
    main()