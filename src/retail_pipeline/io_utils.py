from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from retail_pipeline.config import PathConfig


def ensure_directories(paths: PathConfig) -> None:
    """Создает рабочие папки проекта, если они еще не существуют."""
    for directory in [
        paths.raw_dir,
        paths.stage_dir,
        paths.mart_dir,
        paths.models_dir,
        paths.reports_dir,
        paths.figures_dir,
        paths.diagrams_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> Path:
    """Сохраняет объект Python в JSON-файл с читаемым форматированием."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    """Читает JSON-файл и возвращает объект Python."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> Path:
    """Сохраняет текстовый файл в кодировке UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def save_dataframe(path: Path, dataframe: pd.DataFrame) -> Path:
    """Сохраняет DataFrame в CSV-файл без индекса."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path


def load_dataframe(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Загружает CSV-файл в DataFrame с опциональным разбором дат."""
    return pd.read_csv(path, parse_dates=parse_dates)
