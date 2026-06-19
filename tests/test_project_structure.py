from pathlib import Path


def test_required_project_files_exist() -> None:
    """Проверяет наличие ключевых файлов проекта."""
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "docker-compose.etl.yml",
        root / "docker-compose.superset.yml",
        root / "dags" / "retail_sales_ml_dwh.py",
        root / "docker" / "superset" / "register_retail_database.py",
        root / "docker" / "superset" / "superset_config.py",
        root / "src" / "retail_pipeline" / "clustering.py",
    ]
    for path in required:
        assert path.exists(), path
