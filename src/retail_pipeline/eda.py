from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from retail_pipeline.config import PathConfig
from retail_pipeline.io_utils import ensure_directories, write_text


def markdown_table(dataframe: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    """Создает простую Markdown-таблицу без дополнительных зависимостей."""
    visible = dataframe[columns].head(limit).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in visible.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def save_histogram(dataframe: pd.DataFrame, path: Path) -> None:
    """Сохраняет график распределения розничных продаж."""
    clipped = dataframe["rtl_sales"].clip(upper=dataframe["rtl_sales"].quantile(0.99))
    plt.figure(figsize=(10, 5))
    sns.histplot(clipped, bins=40)
    plt.title("Распределение розничных продаж")
    plt.xlabel("Розничные продажи")
    plt.ylabel("Количество строк")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_monthly_sales_plot(dataframe: pd.DataFrame, path: Path) -> None:
    """Сохраняет график динамики розничных продаж по месяцам."""
    monthly = dataframe.groupby("period_date", as_index=False)["rtl_sales"].sum().sort_values("period_date")
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=monthly, x="period_date", y="rtl_sales", marker="o")
    plt.title("Динамика розничных продаж по месяцам")
    plt.xlabel("Месяц")
    plt.ylabel("Розничные продажи")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_item_type_plot(dataframe: pd.DataFrame, path: Path) -> None:
    """Сохраняет столбчатый график продаж по типам товаров."""
    by_type = dataframe.groupby("item_type", as_index=False)["rtl_sales"].sum().sort_values("rtl_sales", ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=by_type, x="rtl_sales", y="item_type")
    plt.title("Розничные продажи по типам товаров")
    plt.xlabel("Розничные продажи")
    plt.ylabel("Тип товара")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_supplier_plot(dataframe: pd.DataFrame, path: Path) -> None:
    """Сохраняет столбчатый график топ-поставщиков по розничным продажам."""
    suppliers = dataframe.groupby("supplier", as_index=False)["rtl_sales"].sum().sort_values("rtl_sales", ascending=False).head(15)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=suppliers, x="rtl_sales", y="supplier")
    plt.title("Топ поставщиков по розничным продажам")
    plt.xlabel("Розничные продажи")
    plt.ylabel("Поставщик")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def detect_outliers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Определяет выбросы по правилу межквартильного размаха для retail sales."""
    q1 = dataframe["rtl_sales"].quantile(0.25)
    q3 = dataframe["rtl_sales"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    return dataframe[dataframe["rtl_sales"] > upper_bound]


def build_eda_report(paths: PathConfig, clean_path: str | Path) -> Path:
    """Формирует разведочный анализ данных, русскоязычный отчет и графики."""
    ensure_directories(paths)
    dataframe = pd.read_csv(clean_path, parse_dates=["period_date"])
    missing = dataframe.isna().sum().reset_index()
    missing.columns = ["Колонка", "Пропуски"]
    duplicates = int(dataframe.duplicated().sum())
    negative_rows = dataframe[(dataframe[["rtl_sales", "rtl_transfers", "whs_sales"]] < 0).any(axis=1)]
    outliers = detect_outliers(dataframe)
    monthly = dataframe.groupby("period_date", as_index=False)["rtl_sales"].sum().sort_values("period_date")
    item_types = dataframe.groupby("item_type", as_index=False)["rtl_sales"].sum().sort_values("rtl_sales", ascending=False)
    suppliers = dataframe.groupby("supplier", as_index=False)["rtl_sales"].sum().sort_values("rtl_sales", ascending=False).head(10)
    save_histogram(dataframe, paths.figures_dir / "sales_distribution.png")
    save_monthly_sales_plot(dataframe, paths.figures_dir / "monthly_sales.png")
    save_item_type_plot(dataframe, paths.figures_dir / "item_type_sales.png")
    save_supplier_plot(dataframe, paths.figures_dir / "top_suppliers.png")
    report = f"""# Разведочный анализ данных

Источник: Montgomery County Warehouse and Retail Sales.

## Размер и качество данных

- Количество строк: {len(dataframe)}
- Количество колонок: {len(dataframe.columns)}
- Полные дубликаты: {duplicates}
- Строки с отрицательными значениями продаж или перемещений: {len(negative_rows)}
- Выбросы retail sales по правилу IQR: {len(outliers)}

## Пропуски

{markdown_table(missing, ["Колонка", "Пропуски"], limit=len(missing))}

## Продажи по месяцам

{markdown_table(monthly.assign(period_date=monthly["period_date"].dt.strftime("%Y-%m")), ["period_date", "rtl_sales"], limit=12)}

## Продажи по типам товаров

{markdown_table(item_types, ["item_type", "rtl_sales"], limit=20)}

## Топ поставщиков

{markdown_table(suppliers, ["supplier", "rtl_sales"], limit=10)}

## Выводы для ML

Данные имеют выраженную сезонность по месяцам и различаются по типам товаров и поставщикам. Для модели используются агрегированные признаки, лаги, скользящие статистики и доли складских продаж и перемещений. Сырые описания товаров и технические хеш-ключи исключаются из признакового пространства.
"""
    return write_text(paths.reports_dir / "eda_report.md", report)
