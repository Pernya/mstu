from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from retail_pipeline.config import MLConfig, PathConfig
from retail_pipeline.io_utils import ensure_directories, save_dataframe


def assign_supplier_group(dataframe: pd.DataFrame, top_suppliers: int) -> pd.Series:
    """Назначает поставщикам группу top-N или OTHER по суммарным розничным продажам."""
    top_values = (
        dataframe.groupby("supplier")["rtl_sales"]
        .sum()
        .sort_values(ascending=False)
        .head(top_suppliers)
        .index
        .tolist()
    )
    return dataframe["supplier"].where(dataframe["supplier"].isin(top_values), "OTHER")


def aggregate_monthly_sales(cleaned: pd.DataFrame, top_suppliers: int) -> pd.DataFrame:
    """Агрегирует очищенные продажи до месячного уровня по типу товара и группе поставщика."""
    prepared = cleaned.copy()
    prepared["supplier_group"] = assign_supplier_group(prepared, top_suppliers)
    grouped = (
        prepared.groupby(["period_date", "calendar_year", "cal_month_num", "item_type", "supplier_group"], as_index=False)
        .agg(
            retail_sales=("rtl_sales", "sum"),
            retail_transfers=("rtl_transfers", "sum"),
            warehouse_sales=("whs_sales", "sum"),
            source_rows=("sales_link_hk", "count"),
            products_count=("item_code", "nunique"),
        )
        .sort_values(["item_type", "supplier_group", "period_date"])
    )
    return grouped


def add_calendar_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Добавляет календарные и сезонные признаки к месячной витрине."""
    featured = dataframe.copy()
    featured["quarter"] = ((featured["cal_month_num"] - 1) // 3 + 1).astype(int)
    featured["month_sin"] = np.sin(2 * np.pi * featured["cal_month_num"] / 12)
    featured["month_cos"] = np.cos(2 * np.pi * featured["cal_month_num"] / 12)
    return featured


def add_lag_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Добавляет лаги, скользящие статистики и долевые признаки по каждой товарной группе."""
    featured = dataframe.sort_values(["item_type", "supplier_group", "period_date"]).copy()
    group_keys = ["item_type", "supplier_group"]
    grouped = featured.groupby(group_keys)
    featured["sales_lag_1"] = grouped["retail_sales"].shift(1)
    featured["sales_lag_2"] = grouped["retail_sales"].shift(2)
    featured["sales_lag_3"] = grouped["retail_sales"].shift(3)
    featured["sales_rolling_mean_3"] = grouped["retail_sales"].transform(lambda series: series.shift(1).rolling(3).mean())
    featured["sales_rolling_std_3"] = grouped["retail_sales"].transform(lambda series: series.shift(1).rolling(3).std())
    featured["sales_pct_change_1"] = grouped["retail_sales"].transform(lambda series: series.shift(1).pct_change(fill_method=None)).replace([np.inf, -np.inf], np.nan)
    featured["warehouse_sales_lag_1"] = grouped["warehouse_sales"].shift(1)
    featured["retail_transfers_lag_1"] = grouped["retail_transfers"].shift(1)
    featured["source_rows_lag_1"] = grouped["source_rows"].shift(1)
    featured["products_count_lag_1"] = grouped["products_count"].shift(1)
    denominator = featured["sales_lag_1"].abs() + featured["warehouse_sales_lag_1"].abs() + featured["retail_transfers_lag_1"].abs() + 1.0
    featured["warehouse_share_lag_1"] = featured["warehouse_sales_lag_1"] / denominator
    featured["transfer_share_lag_1"] = featured["retail_transfers_lag_1"] / denominator
    return featured


def build_feature_table(paths: PathConfig, ml_config: MLConfig, clean_path: str | Path) -> Path:
    """Создает признаковую витрину для обучения модели прогноза продаж."""
    ensure_directories(paths)
    cleaned = pd.read_csv(clean_path, parse_dates=["period_date", "load_dttm"])
    monthly = aggregate_monthly_sales(cleaned, ml_config.top_suppliers)
    featured = add_lag_features(add_calendar_features(monthly))
    featured = featured.dropna(subset=["sales_lag_1", "sales_lag_2", "sales_lag_3"]).reset_index(drop=True)
    featured["sales_rolling_std_3"] = featured["sales_rolling_std_3"].fillna(0.0)
    featured["sales_pct_change_1"] = featured["sales_pct_change_1"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    save_feature_rows_plot(paths, featured)
    save_feature_correlation_plot(paths, featured)
    target_path = paths.mart_dir / "ml_features.csv"
    save_dataframe(target_path, featured)
    return target_path


def save_feature_rows_plot(paths: PathConfig, featured: pd.DataFrame) -> Path:
    """Сохраняет график количества ML-строк по типам товаров."""
    figure_path = paths.figures_dir / "feature_rows_by_item_type.png"
    plotted = featured.groupby("item_type", as_index=False).size().sort_values("size", ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plotted, x="size", y="item_type", color="#F28E2B")
    plt.title("Количество строк признаковой витрины по типам товаров")
    plt.xlabel("Строки")
    plt.ylabel("Тип товара")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def save_feature_correlation_plot(paths: PathConfig, featured: pd.DataFrame) -> Path:
    """Сохраняет тепловую карту корреляций ключевых числовых признаков."""
    figure_path = paths.figures_dir / "feature_correlation_heatmap.png"
    columns = [
        "retail_sales",
        "sales_lag_1",
        "sales_lag_2",
        "sales_lag_3",
        "sales_rolling_mean_3",
        "sales_pct_change_1",
        "warehouse_share_lag_1",
        "transfer_share_lag_1",
        "products_count_lag_1",
    ]
    available = [column for column in columns if column in featured.columns]
    corr = featured[available].corr(numeric_only=True).fillna(0.0)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False)
    plt.title("Корреляции признаков ML-витрины")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def next_month_start(period: pd.Timestamp) -> pd.Timestamp:
    """Возвращает первый день следующего месяца для заданного периода."""
    return (period + pd.offsets.MonthBegin(1)).normalize()


def build_next_month_features(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Формирует строки признаков для прогноза следующего месяца по каждой группе."""
    rows = []
    for (item_type, supplier_group), group in feature_table.groupby(["item_type", "supplier_group"]):
        ordered = group.sort_values("period_date")
        if len(ordered) < 3:
            continue
        last_rows = ordered.tail(3)
        last = ordered.iloc[-1]
        forecast_period = next_month_start(pd.Timestamp(last["period_date"]))
        sales_values = last_rows["retail_sales"].tolist()
        warehouse_lag = float(last["warehouse_sales"])
        transfers_lag = float(last["retail_transfers"])
        denominator = abs(float(last["retail_sales"])) + abs(warehouse_lag) + abs(transfers_lag) + 1.0
        rows.append(
            {
                "period_date": forecast_period,
                "calendar_year": forecast_period.year,
                "cal_month_num": forecast_period.month,
                "item_type": item_type,
                "supplier_group": supplier_group,
                "retail_sales": np.nan,
                "retail_transfers": np.nan,
                "warehouse_sales": np.nan,
                "source_rows": float(last["source_rows"]),
                "products_count": float(last["products_count"]),
                "source_rows_lag_1": float(last["source_rows"]),
                "products_count_lag_1": float(last["products_count"]),
                "quarter": int((forecast_period.month - 1) // 3 + 1),
                "month_sin": np.sin(2 * np.pi * forecast_period.month / 12),
                "month_cos": np.cos(2 * np.pi * forecast_period.month / 12),
                "sales_lag_1": float(sales_values[-1]),
                "sales_lag_2": float(sales_values[-2]),
                "sales_lag_3": float(sales_values[-3]),
                "sales_rolling_mean_3": float(np.mean(sales_values)),
                "sales_rolling_std_3": float(np.std(sales_values, ddof=1)) if len(sales_values) > 1 else 0.0,
                "sales_pct_change_1": float((sales_values[-1] - sales_values[-2]) / sales_values[-2]) if sales_values[-2] != 0 else 0.0,
                "warehouse_sales_lag_1": warehouse_lag,
                "retail_transfers_lag_1": transfers_lag,
                "warehouse_share_lag_1": warehouse_lag / denominator,
                "transfer_share_lag_1": transfers_lag / denominator,
            }
        )
    return pd.DataFrame(rows)
