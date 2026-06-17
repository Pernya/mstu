import pandas as pd
from pathlib import Path

from retail_pipeline.config import MLConfig
from retail_pipeline.ml import select_relevant_features


def ml_config() -> MLConfig:
    """Возвращает тестовую конфигурацию отбора признаков."""
    return MLConfig(
        project_root=Path("."),
        top_suppliers=3,
        test_months=2,
        random_state=42,
        estimators=10,
        max_missing_share=0.4,
        max_category_cardinality=3,
        cluster_count=3,
        cluster_top_item_types=3,
    )


def test_select_relevant_features_removes_leakage_and_high_cardinality() -> None:
    """Проверяет автоматическое удаление утечек, дат и высококардинальных признаков."""
    dataframe = pd.DataFrame(
        {
            "period_date": pd.date_range("2020-01-01", periods=5, freq="MS"),
            "retail_sales": [1, 2, 3, 4, 5],
            "retail_transfers": [1, 1, 1, 1, 1],
            "technical_hk": ["a", "b", "c", "d", "e"],
            "category": ["A", "B", "C", "D", "E"],
            "useful_numeric": [1, 3, 5, 7, 9],
            "useful_category": ["A", "A", "B", "B", "B"],
        }
    )
    features, target, removed = select_relevant_features(dataframe, ml_config())
    assert "retail_sales" not in features.columns
    assert "retail_transfers" not in features.columns
    assert "period_date" not in features.columns
    assert "category" not in features.columns
    assert "useful_numeric" in features.columns
    assert len(target) == 5
    assert not removed.empty
