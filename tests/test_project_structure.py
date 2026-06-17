from pathlib import Path


def test_required_project_files_exist() -> None:
    """Проверяет наличие ключевых файлов проекта."""
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "docker-compose.etl.yml",
        root / "docker-compose.superset.yml",
        root / "dags" / "retail_sales_ml_dwh.py",
        root / "docs" / "diagrams" / "solution_architecture.puml",
        root / "docs" / "diagrams" / "etl_activity.puml",
        root / "docs" / "diagrams" / "data_architecture.mmd",
        root / "docs" / "diagrams" / "ml_pipeline.mmd",
        root / "src" / "retail_pipeline" / "clustering.py",
    ]
    for path in required:
        assert path.exists(), path
