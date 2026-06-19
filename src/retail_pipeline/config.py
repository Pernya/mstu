from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class BaseConfig:
    """Базовая конфигурация с общим корнем проекта и чтением переменных окружения."""

    project_root: Path

    @staticmethod
    def env(name: str, default: str) -> str:
        """Возвращает значение переменной окружения или переданное значение по умолчанию."""
        return os.getenv(name, default)

    @staticmethod
    def env_int(name: str, default: int) -> int:
        """Возвращает целочисленное значение переменной окружения или значение по умолчанию."""
        value = os.getenv(name)
        return int(value) if value not in (None, "") else default

    @classmethod
    def resolve_project_root(cls) -> Path:
        """Определяет корневую папку проекта из окружения или расположения пакета."""
        explicit_root = os.getenv("PROJECT_ROOT")
        if explicit_root:
            return Path(explicit_root).expanduser().resolve()
        return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PathConfig(BaseConfig):
    """Конфигурация папок для исходных данных, витрин, моделей и отчетов."""

    raw_dir: Path
    stage_dir: Path
    mart_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path

    @classmethod
    def from_env(cls) -> "PathConfig":
        """Создает конфигурацию путей на основе корня проекта."""
        project_root = cls.resolve_project_root()
        return cls(
            project_root=project_root,
            raw_dir=project_root / "data" / "raw",
            stage_dir=project_root / "data" / "stage",
            mart_dir=project_root / "data" / "mart",
            models_dir=project_root / "models",
            reports_dir=project_root / "reports",
            figures_dir=project_root / "reports" / "figures",
        )


@dataclass(frozen=True)
class SocrataConfig(BaseConfig):
    """Конфигурация официального API Montgomery County Warehouse and Retail Sales."""

    base_url: str
    dataset_id: str
    start_year: int
    rows_per_period: int
    request_timeout: int

    @classmethod
    def from_env(cls) -> "SocrataConfig":
        """Создает конфигурацию API из переменных окружения."""
        project_root = cls.resolve_project_root()
        return cls(
            project_root=project_root,
            base_url=cls.env("SOCRATA_BASE_URL", "https://data.montgomerycountymd.gov"),
            dataset_id=cls.env("SOCRATA_DATASET_ID", "v76h-r7br"),
            start_year=cls.env_int("RETAIL_START_YEAR", 2018),
            rows_per_period=cls.env_int("RETAIL_ROWS_PER_PERIOD", 1500),
            request_timeout=cls.env_int("RETAIL_REQUEST_TIMEOUT", 45),
        )


@dataclass(frozen=True)
class DatabaseConfig(BaseConfig):
    """Конфигурация подключения к PostgreSQL-хранилищу."""

    database_url: str

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Создает конфигурацию базы данных из переменных окружения."""
        project_root = cls.resolve_project_root()
        return cls(
            project_root=project_root,
            database_url=cls.env(
                "RETAIL_DATABASE_URL",
                "postgresql+psycopg2://retail_user:retail_password@localhost:5433/retail_dwh",
            ),
        )


@dataclass(frozen=True)
class MLConfig(BaseConfig):
    """Конфигурация обучения модели и отбора признаков."""

    top_suppliers: int
    test_months: int
    random_state: int
    estimators: int
    max_missing_share: float
    max_category_cardinality: int
    cluster_count: int
    cluster_top_item_types: int

    @classmethod
    def from_env(cls) -> "MLConfig":
        """Создает конфигурацию ML-пайплайна из переменных окружения."""
        project_root = cls.resolve_project_root()
        return cls(
            project_root=project_root,
            top_suppliers=cls.env_int("ML_TOP_SUPPLIERS", 20),
            test_months=cls.env_int("ML_TEST_MONTHS", 4),
            random_state=cls.env_int("ML_RANDOM_STATE", 42),
            estimators=cls.env_int("ML_ESTIMATORS", 120),
            max_missing_share=float(cls.env("ML_MAX_MISSING_SHARE", "0.4")),
            max_category_cardinality=cls.env_int("ML_MAX_CATEGORY_CARDINALITY", 40),
            cluster_count=cls.env_int("ML_CLUSTER_COUNT", 4),
            cluster_top_item_types=cls.env_int("ML_CLUSTER_TOP_ITEM_TYPES", 6),
        )


@dataclass(frozen=True)
class AppConfig:
    """Единая конфигурация приложения, объединяющая пути, API, БД и ML."""

    paths: PathConfig
    socrata: SocrataConfig
    database: DatabaseConfig
    ml: MLConfig


def load_config() -> AppConfig:
    """Загружает `.env` и возвращает полную конфигурацию проекта."""
    project_root = BaseConfig.resolve_project_root()
    load_dotenv(project_root / ".env")
    return AppConfig(
        paths=PathConfig.from_env(),
        socrata=SocrataConfig.from_env(),
        database=DatabaseConfig.from_env(),
        ml=MLConfig.from_env(),
    )
