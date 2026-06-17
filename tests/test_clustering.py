from pathlib import Path

import pandas as pd

from retail_pipeline.clustering import build_cluster_profiles, build_supplier_features, fit_supplier_clusters
from retail_pipeline.config import MLConfig


def ml_config() -> MLConfig:
    """Возвращает тестовую конфигурацию кластеризации."""
    return MLConfig(
        project_root=Path("."),
        top_suppliers=3,
        test_months=2,
        random_state=42,
        estimators=10,
        max_missing_share=0.4,
        max_category_cardinality=5,
        cluster_count=3,
        cluster_top_item_types=3,
    )


def sample_clean_sales() -> pd.DataFrame:
    """Возвращает очищенные продажи для проверки кластеризации поставщиков."""
    rows = []
    for supplier_index, supplier in enumerate(["A", "B", "C", "D"], start=1):
        for month in range(1, 5):
            rows.append(
                {
                    "supplier": supplier,
                    "item_type": "WINE" if supplier_index % 2 else "BEER",
                    "item_code": f"{supplier_index}{month}",
                    "rtl_sales": float(supplier_index * month * 100),
                    "rtl_transfers": float(supplier_index * month * 10),
                    "whs_sales": float(supplier_index * month * 20),
                    "sales_link_hk": f"{supplier}-{month}",
                    "period_date": pd.Timestamp(year=2022, month=month, day=1),
                    "load_dttm": pd.Timestamp("2022-01-01"),
                }
            )
    return pd.DataFrame(rows)


def test_supplier_clustering_builds_profiles() -> None:
    """Проверяет расчет признаков, кластеров и портретов сегментов."""
    features, columns = build_supplier_features(sample_clean_sales(), ml_config())
    clustered = fit_supplier_clusters(features, columns, ml_config())
    profiles = build_cluster_profiles(clustered)
    assert "cluster_id" in clustered.columns
    assert len(profiles) <= 3
    assert {"segment_name", "business_recommendation"}.issubset(profiles.columns)
