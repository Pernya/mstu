from __future__ import annotations

from pathlib import Path

from retail_pipeline.config import DatabaseConfig, PathConfig
from retail_pipeline.warehouse import count_table_rows


def assert_file_exists(path: Path) -> None:
    """Проверяет наличие файла и выбрасывает ошибку при его отсутствии."""
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")


def assert_table_has_rows(db_config: DatabaseConfig, table_name: str) -> int:
    """Проверяет, что таблица PostgreSQL существует и содержит строки."""
    row_count = count_table_rows(db_config, table_name)
    if row_count <= 0:
        raise RuntimeError(f"Таблица {table_name} не содержит строк")
    return row_count


def validate_outputs(paths: PathConfig, db_config: DatabaseConfig) -> dict[str, int | str]:
    """Проверяет ключевые артефакты пайплайна, DWH, витрины, модель и отчеты."""
    required_files = [
        paths.raw_dir / "sales_raw.csv",
        paths.stage_dir / "sales_clean.csv",
        paths.mart_dir / "ml_features.csv",
        paths.models_dir / "retail_sales_forecast.joblib",
        paths.reports_dir / "eda_report.md",
        paths.reports_dir / "model_report.md",
        paths.reports_dir / "ml_metrics.json",
        paths.reports_dir / "feature_ranking.csv",
        paths.reports_dir / "forecast_next_month.csv",
        paths.reports_dir / "cluster_report.md",
        paths.reports_dir / "supplier_cluster_profiles.csv",
        paths.reports_dir / "supplier_cluster_members.csv",
        paths.figures_dir / "monthly_sales.png",
        paths.figures_dir / "item_type_sales.png",
        paths.figures_dir / "extract_period_coverage.png",
        paths.figures_dir / "extract_rows_by_month.png",
        paths.figures_dir / "transform_quality_summary.png",
        paths.figures_dir / "transform_metric_boxplot.png",
        paths.figures_dir / "dwh_table_counts.png",
        paths.figures_dir / "mart_table_counts.png",
        paths.figures_dir / "feature_rows_by_item_type.png",
        paths.figures_dir / "feature_correlation_heatmap.png",
        paths.figures_dir / "feature_importance.png",
        paths.figures_dir / "feature_scores_heatmap.png",
        paths.figures_dir / "actual_vs_predicted.png",
        paths.figures_dir / "model_generalization.png",
        paths.figures_dir / "residual_diagnostics.png",
        paths.figures_dir / "regression_learning_curve.png",
        paths.figures_dir / "forecast_next_month.png",
        paths.figures_dir / "supplier_clusters.png",
        paths.figures_dir / "supplier_cluster_profiles.png",
    ]
    for path in required_files:
        assert_file_exists(path)
    table_counts = {
        "raw.sales_raw": assert_table_has_rows(db_config, "raw.sales_raw"),
        "mart.dm_item_type_sales": assert_table_has_rows(db_config, "mart.dm_item_type_sales"),
        "ml.sales_forecast": assert_table_has_rows(db_config, "ml.sales_forecast"),
        "ml.supplier_cluster_profiles": assert_table_has_rows(db_config, "ml.supplier_cluster_profiles"),
        "ml.supplier_cluster_members": assert_table_has_rows(db_config, "ml.supplier_cluster_members"),
    }
    return {"status": "ok", **table_counts}
