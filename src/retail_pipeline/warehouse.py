from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from retail_pipeline.config import DatabaseConfig, PathConfig
from retail_pipeline.io_utils import ensure_directories, read_json, save_dataframe


def create_postgres_engine(config: DatabaseConfig) -> Engine:
    """Создает SQLAlchemy engine для подключения к PostgreSQL DWH."""
    return create_engine(config.database_url, pool_pre_ping=True)


def execute_sql(engine: Engine, statements: list[str]) -> None:
    """Выполняет набор SQL-операторов в одной транзакции."""
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def prepare_schemas(engine: Engine) -> None:
    """Создает схемы raw, dv, mart и ml в PostgreSQL."""
    execute_sql(
        engine,
        [
            "CREATE SCHEMA IF NOT EXISTS raw",
            "CREATE SCHEMA IF NOT EXISTS dv",
            "CREATE SCHEMA IF NOT EXISTS mart",
            "CREATE SCHEMA IF NOT EXISTS ml",
        ],
    )


def write_table(engine: Engine, dataframe: pd.DataFrame, schema: str, table: str) -> None:
    """Перезаписывает таблицу PostgreSQL содержимым DataFrame."""
    dataframe.to_sql(table, engine, schema=schema, if_exists="replace", index=False, method="multi", chunksize=1000)


def metadata_to_dataframe(metadata_path: str | Path) -> pd.DataFrame:
    """Преобразует JSON-метаданные Socrata в табличный вид."""
    rows = []
    for item in read_json(Path(metadata_path)):
        rows.append(
            {
                "name": item.get("name"),
                "field_name": item.get("fieldName"),
                "data_type": item.get("dataTypeName"),
                "description": item.get("description"),
                "position": item.get("position"),
            }
        )
    return pd.DataFrame(rows)


def build_data_vault_frames(cleaned: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Формирует Data Vault hubs, links и satellites из очищенных строк продаж."""
    hub_period = cleaned[["period_hk", "calendar_year", "cal_month_num", "period_date", "load_dttm", "record_source"]].drop_duplicates()
    hub_supplier = cleaned[["supplier_hk", "supplier", "load_dttm", "record_source"]].drop_duplicates()
    hub_product = cleaned[["product_hk", "item_code", "load_dttm", "record_source"]].drop_duplicates()
    link_sales = cleaned[["sales_link_hk", "period_hk", "supplier_hk", "product_hk", "load_dttm", "record_source"]].drop_duplicates()
    sat_product = cleaned[["product_hk", "item_description", "item_type", "load_dttm", "record_source"]].drop_duplicates()
    sat_sales_metrics = cleaned[
        ["sales_link_hk", "rtl_sales", "rtl_transfers", "whs_sales", "metrics_hashdiff", "load_dttm", "record_source"]
    ].drop_duplicates()
    return {
        "hub_period": hub_period,
        "hub_supplier": hub_supplier,
        "hub_product": hub_product,
        "link_sales": link_sales,
        "sat_product": sat_product,
        "sat_sales_metrics": sat_sales_metrics,
    }


def load_dwh(paths: PathConfig, db_config: DatabaseConfig, clean_path: str | Path, metadata_path: str | Path) -> dict[str, int]:
    """Загружает очищенные данные в raw и Data Vault слои PostgreSQL."""
    ensure_directories(paths)
    engine = create_postgres_engine(db_config)
    prepare_schemas(engine)
    cleaned = pd.read_csv(clean_path, parse_dates=["period_date", "load_dttm"])
    metadata = metadata_to_dataframe(metadata_path)
    write_table(engine, cleaned, "raw", "sales_raw")
    write_table(engine, metadata, "raw", "source_columns")
    frames = build_data_vault_frames(cleaned)
    for table, dataframe in frames.items():
        write_table(engine, dataframe, "dv", table)
    counts = {"raw.sales_raw": len(cleaned), **{f"dv.{table}": len(frame) for table, frame in frames.items()}}
    save_table_counts_plot(paths, counts, "dwh_table_counts.png", "Строки в raw и Data Vault")
    return counts


def build_marts(paths: PathConfig, db_config: DatabaseConfig) -> dict[str, int]:
    """Создает витрины продаж для дашбордов и сохраняет копии в CSV."""
    ensure_directories(paths)
    engine = create_postgres_engine(db_config)
    statements = [
        "DROP TABLE IF EXISTS mart.dm_monthly_sales",
        """
        CREATE TABLE mart.dm_monthly_sales AS
        SELECT
            period_date,
            calendar_year,
            cal_month_num,
            SUM(rtl_sales) AS retail_sales,
            SUM(rtl_transfers) AS retail_transfers,
            SUM(whs_sales) AS warehouse_sales,
            COUNT(*) AS source_rows,
            COUNT(DISTINCT supplier) AS suppliers_count,
            COUNT(DISTINCT item_code) AS products_count
        FROM raw.sales_raw
        GROUP BY period_date, calendar_year, cal_month_num
        ORDER BY period_date
        """,
        "DROP TABLE IF EXISTS mart.dm_item_type_sales",
        """
        CREATE TABLE mart.dm_item_type_sales AS
        SELECT
            period_date,
            calendar_year,
            cal_month_num,
            item_type,
            SUM(rtl_sales) AS retail_sales,
            SUM(rtl_transfers) AS retail_transfers,
            SUM(whs_sales) AS warehouse_sales,
            COUNT(*) AS source_rows
        FROM raw.sales_raw
        GROUP BY period_date, calendar_year, cal_month_num, item_type
        ORDER BY period_date, item_type
        """,
        "DROP TABLE IF EXISTS mart.dm_supplier_sales",
        """
        CREATE TABLE mart.dm_supplier_sales AS
        SELECT
            supplier,
            SUM(rtl_sales) AS retail_sales,
            SUM(rtl_transfers) AS retail_transfers,
            SUM(whs_sales) AS warehouse_sales,
            COUNT(*) AS source_rows,
            COUNT(DISTINCT item_code) AS products_count
        FROM raw.sales_raw
        GROUP BY supplier
        ORDER BY retail_sales DESC
        """,
        "DROP TABLE IF EXISTS mart.dm_ml_sales",
        """
        CREATE TABLE mart.dm_ml_sales AS
        SELECT
            period_date,
            calendar_year,
            cal_month_num,
            supplier,
            item_type,
            SUM(rtl_sales) AS retail_sales,
            SUM(rtl_transfers) AS retail_transfers,
            SUM(whs_sales) AS warehouse_sales,
            COUNT(*) AS source_rows
        FROM raw.sales_raw
        GROUP BY period_date, calendar_year, cal_month_num, supplier, item_type
        ORDER BY period_date, supplier, item_type
        """,
    ]
    execute_sql(engine, statements)
    create_support_indexes(engine)
    marts = {}
    for table in ["dm_monthly_sales", "dm_item_type_sales", "dm_supplier_sales", "dm_ml_sales"]:
        dataframe = pd.read_sql_query(f"SELECT * FROM mart.{table}", engine)
        save_dataframe(paths.mart_dir / f"{table}.csv", dataframe)
        marts[f"mart.{table}"] = len(dataframe)
    save_table_counts_plot(paths, marts, "mart_table_counts.png", "Строки в аналитических витринах")
    return marts


def create_support_indexes(engine: Engine) -> None:
    """Создает индексы для часто используемых ключей, связей и витрин."""
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_raw_sales_period ON raw.sales_raw(period_date)",
        "CREATE INDEX IF NOT EXISTS idx_raw_sales_supplier ON raw.sales_raw(supplier)",
        "CREATE INDEX IF NOT EXISTS idx_raw_sales_item_type ON raw.sales_raw(item_type)",
        "CREATE INDEX IF NOT EXISTS idx_dv_link_sales_period ON dv.link_sales(period_hk)",
        "CREATE INDEX IF NOT EXISTS idx_dv_link_sales_supplier ON dv.link_sales(supplier_hk)",
        "CREATE INDEX IF NOT EXISTS idx_dv_link_sales_product ON dv.link_sales(product_hk)",
        "CREATE INDEX IF NOT EXISTS idx_mart_monthly_sales_period ON mart.dm_monthly_sales(period_date)",
        "CREATE INDEX IF NOT EXISTS idx_mart_item_type_sales_period ON mart.dm_item_type_sales(period_date)",
    ]
    execute_sql(engine, statements)


def count_table_rows(db_config: DatabaseConfig, qualified_table: str) -> int:
    """Возвращает количество строк в указанной таблице PostgreSQL."""
    engine = create_postgres_engine(db_config)
    with engine.begin() as connection:
        result = connection.execute(text(f"SELECT COUNT(*) FROM {qualified_table}"))
        return int(result.scalar() or 0)


def write_ml_outputs(db_config: DatabaseConfig, metrics: dict[str, Any], forecast: pd.DataFrame) -> dict[str, int]:
    """Сохраняет метрики модели и прогнозы в схему ml."""
    engine = create_postgres_engine(db_config)
    prepare_schemas(engine)
    metrics_frame = pd.DataFrame([metrics])
    write_table(engine, metrics_frame, "ml", "model_metrics")
    write_table(engine, forecast, "ml", "sales_forecast")
    return {"ml.model_metrics": len(metrics_frame), "ml.sales_forecast": len(forecast)}


def write_cluster_outputs(db_config: DatabaseConfig, profiles: pd.DataFrame, members: pd.DataFrame) -> dict[str, int]:
    """Сохраняет результаты кластеризации поставщиков в схему ml."""
    engine = create_postgres_engine(db_config)
    prepare_schemas(engine)
    write_table(engine, profiles, "ml", "supplier_cluster_profiles")
    write_table(engine, members, "ml", "supplier_cluster_members")
    return {"ml.supplier_cluster_profiles": len(profiles), "ml.supplier_cluster_members": len(members)}


def save_table_counts_plot(paths: PathConfig, counts: dict[str, int], filename: str, title: str) -> Path:
    """Сохраняет график количества строк по таблицам текущего слоя DWH."""
    figure_path = paths.figures_dir / filename
    plotted = pd.DataFrame({"table": list(counts.keys()), "rows": list(counts.values())}).sort_values("rows", ascending=True)
    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=plotted, x="rows", y="table", color="#59A14F")
    plt.title(title)
    plt.xlabel("Количество строк")
    plt.ylabel("Таблица")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path
