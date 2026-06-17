from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from retail_pipeline.config import PathConfig
from retail_pipeline.io_utils import ensure_directories, save_dataframe


REQUIRED_COLUMNS = [
    "calendar_year",
    "cal_month_num",
    "supplier",
    "item_code",
    "item_description",
    "item_type",
    "rtl_sales",
    "rtl_transfers",
    "whs_sales",
]


def stable_hash(*values: Any) -> str:
    """Создает стабильный хеш-ключ для бизнес-сущностей DWH."""
    prepared = []
    for value in values:
        if pd.isna(value):
            prepared.append("")
        else:
            prepared.append(str(value).strip().upper())
    return hashlib.sha256("|".join(prepared).encode("utf-8")).hexdigest()[:32]


def validate_columns(dataframe: pd.DataFrame) -> None:
    """Проверяет наличие обязательных колонок в выгрузке Socrata."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing:
        raise ValueError(f"В исходных данных отсутствуют колонки: {', '.join(missing)}")


def normalize_text(value: Any, fallback: str) -> str:
    """Нормализует текстовое значение и подставляет резервную метку при пустом значении."""
    if pd.isna(value):
        return fallback
    normalized = str(value).strip()
    return normalized if normalized else fallback


def clean_sales_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Очищает и типизирует исходные строки продаж для последующей загрузки в DWH."""
    validate_columns(dataframe)
    cleaned = dataframe[REQUIRED_COLUMNS].copy()
    cleaned["calendar_year"] = pd.to_numeric(cleaned["calendar_year"], errors="coerce").astype("Int64")
    cleaned["cal_month_num"] = pd.to_numeric(cleaned["cal_month_num"], errors="coerce").astype("Int64")
    for column in ["rtl_sales", "rtl_transfers", "whs_sales"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0.0).astype(float)
    cleaned["supplier"] = cleaned["supplier"].map(lambda value: normalize_text(value, "UNKNOWN_SUPPLIER"))
    cleaned["item_code"] = cleaned["item_code"].map(lambda value: normalize_text(value, "UNKNOWN_ITEM"))
    cleaned["item_description"] = cleaned["item_description"].map(lambda value: normalize_text(value, "UNKNOWN_DESCRIPTION"))
    cleaned["item_type"] = cleaned["item_type"].map(lambda value: normalize_text(value, "UNKNOWN_TYPE"))
    cleaned = cleaned.dropna(subset=["calendar_year", "cal_month_num"])
    cleaned["calendar_year"] = cleaned["calendar_year"].astype(int)
    cleaned["cal_month_num"] = cleaned["cal_month_num"].astype(int)
    cleaned = cleaned[(cleaned["cal_month_num"] >= 1) & (cleaned["cal_month_num"] <= 12)]
    cleaned["period_date"] = pd.to_datetime(
        {
            "year": cleaned["calendar_year"],
            "month": cleaned["cal_month_num"],
            "day": np.ones(len(cleaned), dtype=int),
        }
    )
    cleaned = cleaned.drop_duplicates(
        subset=["calendar_year", "cal_month_num", "supplier", "item_code", "item_type", "rtl_sales", "rtl_transfers", "whs_sales"]
    )
    loaded_at = datetime.now(timezone.utc).replace(tzinfo=None)
    cleaned["period_hk"] = cleaned.apply(lambda row: stable_hash("PERIOD", row["calendar_year"], row["cal_month_num"]), axis=1)
    cleaned["supplier_hk"] = cleaned["supplier"].map(lambda value: stable_hash("SUPPLIER", value))
    cleaned["product_hk"] = cleaned.apply(lambda row: stable_hash("PRODUCT", row["item_code"], row["item_description"]), axis=1)
    cleaned["sales_link_hk"] = cleaned.apply(
        lambda row: stable_hash("SALE", row["calendar_year"], row["cal_month_num"], row["supplier"], row["item_code"], row["item_type"]),
        axis=1,
    )
    cleaned["metrics_hashdiff"] = cleaned.apply(
        lambda row: stable_hash(row["rtl_sales"], row["rtl_transfers"], row["whs_sales"]),
        axis=1,
    )
    cleaned["load_dttm"] = loaded_at
    cleaned["record_source"] = "montgomery_county_warehouse_retail_sales"
    return cleaned.reset_index(drop=True)


def transform_sales_file(paths: PathConfig, raw_path: str | Path) -> Path:
    """Читает raw CSV, очищает данные и сохраняет stage CSV."""
    ensure_directories(paths)
    dataframe = pd.read_csv(raw_path)
    cleaned = clean_sales_data(dataframe)
    save_transform_quality_plot(paths, dataframe, cleaned)
    save_metric_control_plot(paths, cleaned)
    target_path = paths.stage_dir / "sales_clean.csv"
    save_dataframe(target_path, cleaned)
    return target_path


def save_transform_quality_plot(paths: PathConfig, raw: pd.DataFrame, cleaned: pd.DataFrame) -> Path:
    """Сохраняет график качества transform-этапа: строки до очистки, после очистки и дубли."""
    figure_path = paths.figures_dir / "transform_quality_summary.png"
    duplicates = int(raw.duplicated().sum())
    invalid_periods = len(raw) - len(raw.dropna(subset=["calendar_year", "cal_month_num"]))
    values = pd.DataFrame(
        {
            "metric": ["строки raw", "строки stage", "полные дубли raw", "строки без периода"],
            "value": [len(raw), len(cleaned), duplicates, invalid_periods],
        }
    )
    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=values, x="value", y="metric", color="#4C78A8")
    plt.title("Контроль transform-этапа")
    plt.xlabel("Количество строк")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def save_metric_control_plot(paths: PathConfig, cleaned: pd.DataFrame) -> Path:
    """Сохраняет boxplot основных числовых метрик после очистки."""
    figure_path = paths.figures_dir / "transform_metric_boxplot.png"
    plotted = cleaned[["rtl_sales", "rtl_transfers", "whs_sales"]].copy()
    for column in plotted.columns:
        upper = plotted[column].quantile(0.99)
        plotted[column] = plotted[column].clip(upper=upper)
    long = plotted.melt(var_name="metric", value_name="value")
    plt.figure(figsize=(9, 4.8))
    sns.boxplot(data=long, x="value", y="metric")
    plt.title("Распределение метрик после очистки")
    plt.xlabel("Значение, ограничено 99 перцентилем")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path
