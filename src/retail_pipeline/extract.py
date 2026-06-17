from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

from retail_pipeline.config import PathConfig, SocrataConfig
from retail_pipeline.io_utils import ensure_directories, save_dataframe, write_json


SELECT_COLUMNS = (
    "calendar_year,cal_month_num,supplier,item_code,item_description,"
    "item_type,rtl_sales,rtl_transfers,whs_sales"
)


def request_socrata_json(url: str, params: dict[str, Any] | None, timeout: int) -> list[dict[str, Any]]:
    """Выполняет GET-запрос к Socrata API и возвращает JSON-строки датасета."""
    response = requests.get(url, params=params, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"Ошибка Socrata API {response.status_code}: {response.text[:500]}")
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Ошибка Socrata API: {payload}")
    if isinstance(payload, dict):
        return [payload]
    return payload


def dataset_resource_url(config: SocrataConfig) -> str:
    """Формирует URL ресурса данных Socrata в формате JSON."""
    return f"{config.base_url.rstrip('/')}/resource/{config.dataset_id}.json"


def columns_metadata_url(config: SocrataConfig) -> str:
    """Формирует URL метаданных колонок Socrata."""
    return f"{config.base_url.rstrip('/')}/api/views/{config.dataset_id}/columns.json"


def fetch_source_columns(paths: PathConfig, config: SocrataConfig) -> Path:
    """Загружает описание колонок официального источника и сохраняет его в raw-зону."""
    ensure_directories(paths)
    metadata = request_socrata_json(columns_metadata_url(config), None, config.request_timeout)
    target_path = paths.raw_dir / "source_columns.json"
    write_json(target_path, metadata)
    return target_path


def fetch_available_periods(config: SocrataConfig) -> list[dict[str, int]]:
    """Получает список доступных годовых и месячных периодов для ограниченной загрузки."""
    params = {
        "$select": "calendar_year,cal_month_num",
        "$where": f"calendar_year >= {config.start_year}",
        "$group": "calendar_year,cal_month_num",
        "$order": "calendar_year,cal_month_num",
    }
    rows = request_socrata_json(dataset_resource_url(config), params, config.request_timeout)
    periods = []
    for row in rows:
        periods.append(
            {
                "calendar_year": int(row["calendar_year"]),
                "cal_month_num": int(row["cal_month_num"]),
            }
        )
    return periods


def fetch_sales_for_period(config: SocrataConfig, year: int, month: int) -> list[dict[str, Any]]:
    """Загружает ограниченную выборку продаж за один календарный месяц."""
    params = {
        "$select": SELECT_COLUMNS,
        "$where": f"calendar_year = {year} AND cal_month_num = {month}",
        "$order": "supplier,item_type,item_code",
        "$limit": config.rows_per_period,
    }
    return request_socrata_json(dataset_resource_url(config), params, config.request_timeout)


def fetch_sales_data(paths: PathConfig, config: SocrataConfig) -> Path:
    """Загружает небольшую сбалансированную выборку официальных retail-данных и сохраняет CSV."""
    ensure_directories(paths)
    periods = fetch_available_periods(config)
    rows: list[dict[str, Any]] = []
    for period in periods:
        rows.extend(fetch_sales_for_period(config, period["calendar_year"], period["cal_month_num"]))
    if not rows:
        raise RuntimeError("Источник вернул пустую выборку продаж")
    dataframe = pd.DataFrame(rows)
    save_period_coverage_plot(paths, periods)
    save_extract_rows_plot(paths, dataframe)
    target_path = paths.raw_dir / "sales_raw.csv"
    save_dataframe(target_path, dataframe)
    return target_path


def save_period_coverage_plot(paths: PathConfig, periods: list[dict[str, int]]) -> Path:
    """Сохраняет график покрытия источника по годам и месяцам."""
    coverage = pd.DataFrame(periods)
    figure_path = paths.figures_dir / "extract_period_coverage.png"
    if coverage.empty:
        return figure_path
    coverage["period"] = pd.to_datetime(
        {
            "year": coverage["calendar_year"],
            "month": coverage["cal_month_num"],
            "day": 1,
        }
    )
    coverage["available_period"] = 1
    plt.figure(figsize=(11, 3.8))
    sns.scatterplot(data=coverage, x="period", y="available_period", hue="calendar_year", palette="tab10", legend=False, s=80)
    plt.title("Покрытие периодов в источнике")
    plt.xlabel("Период")
    plt.ylabel("Доступность")
    plt.yticks([1], ["месяц доступен"])
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def save_extract_rows_plot(paths: PathConfig, dataframe: pd.DataFrame) -> Path:
    """Сохраняет график количества загруженных строк по месяцам."""
    figure_path = paths.figures_dir / "extract_rows_by_month.png"
    plotted = dataframe.copy()
    plotted["calendar_year"] = pd.to_numeric(plotted["calendar_year"], errors="coerce")
    plotted["cal_month_num"] = pd.to_numeric(plotted["cal_month_num"], errors="coerce")
    plotted = plotted.dropna(subset=["calendar_year", "cal_month_num"])
    plotted["period_date"] = pd.to_datetime(
        {
            "year": plotted["calendar_year"].astype(int),
            "month": plotted["cal_month_num"].astype(int),
            "day": 1,
        }
    )
    rows_by_month = plotted.groupby("period_date", as_index=False).size()
    plt.figure(figsize=(11, 4.5))
    sns.lineplot(data=rows_by_month, x="period_date", y="size", marker="o")
    plt.title("Количество загруженных строк по месяцам")
    plt.xlabel("Месяц")
    plt.ylabel("Строки")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path
