from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task

from retail_pipeline.clustering import cluster_suppliers
from retail_pipeline.config import load_config
from retail_pipeline.eda import build_eda_report
from retail_pipeline.extract import fetch_sales_data, fetch_source_columns
from retail_pipeline.features import build_feature_table
from retail_pipeline.ml import train_model
from retail_pipeline.transform import transform_sales_file
from retail_pipeline.validation import validate_outputs
from retail_pipeline.warehouse import build_marts, load_dwh


@dag(
    dag_id="retail_sales_ml_dwh",
    description="ETL, DWH, EDA и ML-прогноз розничных продаж",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["retail", "dwh", "ml", "vkr"],
)
def retail_sales_ml_dwh() -> None:
    """Описывает полный Airflow DAG для загрузки данных, DWH, EDA, ML и проверки результатов."""

    @task
    def extract_metadata() -> str:
        """Загружает метаданные колонок официального источника Socrata."""
        config = load_config()
        return str(fetch_source_columns(config.paths, config.socrata))

    @task
    def extract_sales() -> str:
        """Загружает ограниченную выборку retail-продаж из Socrata API."""
        config = load_config()
        return str(fetch_sales_data(config.paths, config.socrata))

    @task
    def transform_sales(raw_path: str) -> str:
        """Очищает raw-данные, типизирует поля и добавляет DWH-ключи."""
        config = load_config()
        return str(transform_sales_file(config.paths, raw_path))

    @task
    def load_data_warehouse(clean_path: str, metadata_path: str) -> dict[str, int]:
        """Загружает raw и Data Vault слои в PostgreSQL DWH."""
        config = load_config()
        return load_dwh(config.paths, config.database, clean_path, metadata_path)

    @task
    def build_dashboard_marts(_: dict[str, int]) -> dict[str, int]:
        """Создает витрины данных для будущих дашбордов и анализа."""
        config = load_config()
        return build_marts(config.paths, config.database)

    @task
    def run_eda(clean_path: str, _: dict[str, int]) -> str:
        """Формирует разведочный анализ и графики по очищенным данным."""
        config = load_config()
        return str(build_eda_report(config.paths, clean_path))

    @task
    def engineer_features(clean_path: str, _: dict[str, int]) -> str:
        """Создает признаковую витрину для ML-прогноза продаж."""
        config = load_config()
        return str(build_feature_table(config.paths, config.ml, clean_path))

    @task
    def build_supplier_segments(clean_path: str, _: dict[str, int]) -> dict:
        """Сегментирует поставщиков, сохраняет портреты кластеров и бизнес-рекомендации."""
        config = load_config()
        return cluster_suppliers(config.paths, config.ml, config.database, clean_path)

    @task
    def train_forecast_model(feature_path: str, eda_path: str) -> dict:
        """Обучает модель, строит рейтинг признаков, сохраняет прогноз и отчеты."""
        config = load_config()
        result = train_model(config.paths, config.ml, config.database, feature_path)
        result["eda_report"] = eda_path
        return result

    @task
    def validate_pipeline_outputs(_: dict, __: dict) -> dict:
        """Проверяет наличие ключевых файлов и строк в DWH, витринах и ML-таблицах."""
        config = load_config()
        return validate_outputs(config.paths, config.database)

    metadata_path = extract_metadata()
    raw_path = extract_sales()
    clean_path = transform_sales(raw_path)
    dwh_result = load_data_warehouse(clean_path, metadata_path)
    marts_result = build_dashboard_marts(dwh_result)
    eda_path = run_eda(clean_path, marts_result)
    feature_path = engineer_features(clean_path, marts_result)
    cluster_result = build_supplier_segments(clean_path, marts_result)
    model_result = train_forecast_model(feature_path, eda_path)
    validate_pipeline_outputs(model_result, cluster_result)


retail_sales_ml_dwh()
