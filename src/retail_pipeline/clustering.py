from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from retail_pipeline.config import DatabaseConfig, MLConfig, PathConfig
from retail_pipeline.io_utils import ensure_directories, save_dataframe, write_text
from retail_pipeline.warehouse import write_cluster_outputs


def safe_feature_suffix(value: Any) -> str:
    """Преобразует значение категории в стабильный суффикс имени признака."""
    prepared = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    allowed = [symbol for symbol in prepared if symbol.isalnum() or symbol == "_"]
    return "".join(allowed) or "unknown"


def supplier_base_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует продажи до уровня поставщика для дальнейшей кластеризации."""
    grouped = (
        cleaned.groupby("supplier", as_index=False)
        .agg(
            retail_sales=("rtl_sales", "sum"),
            retail_transfers=("rtl_transfers", "sum"),
            warehouse_sales=("whs_sales", "sum"),
            source_rows=("sales_link_hk", "count"),
            products_count=("item_code", "nunique"),
            active_months=("period_date", "nunique"),
            negative_rows=("rtl_sales", lambda series: int((series < 0).sum())),
        )
        .sort_values("retail_sales", ascending=False)
    )
    total_flow = grouped["retail_sales"].abs() + grouped["warehouse_sales"].abs() + grouped["retail_transfers"].abs() + 1.0
    grouped["warehouse_share"] = grouped["warehouse_sales"] / total_flow
    grouped["transfer_share"] = grouped["retail_transfers"] / total_flow
    grouped["negative_sales_share"] = grouped["negative_rows"] / grouped["source_rows"].replace(0, np.nan)
    grouped["avg_retail_sales_per_month"] = grouped["retail_sales"] / grouped["active_months"].replace(0, np.nan)
    grouped["product_variety"] = grouped["products_count"] / grouped["source_rows"].replace(0, np.nan)
    return grouped.fillna(0.0).reset_index(drop=True)


def item_type_share_features(cleaned: pd.DataFrame, top_item_types: int) -> pd.DataFrame:
    """Строит доли продаж по наиболее крупным типам товаров для каждого поставщика."""
    top_types = cleaned.groupby("item_type")["rtl_sales"].sum().abs().sort_values(ascending=False).head(top_item_types).index.tolist()
    filtered = cleaned[cleaned["item_type"].isin(top_types)]
    pivot = filtered.pivot_table(index="supplier", columns="item_type", values="rtl_sales", aggfunc="sum", fill_value=0.0)
    totals = cleaned.groupby("supplier")["rtl_sales"].sum().abs().replace(0, np.nan)
    shares = pivot.div(totals, axis=0).fillna(0.0).reset_index()
    shares.columns = ["supplier", *[f"share_{safe_feature_suffix(column)}" for column in shares.columns[1:]]]
    return shares


def build_supplier_features(cleaned: pd.DataFrame, ml_config: MLConfig) -> tuple[pd.DataFrame, list[str]]:
    """Формирует числовую матрицу признаков для кластеризации поставщиков."""
    base = supplier_base_features(cleaned)
    shares = item_type_share_features(cleaned, ml_config.cluster_top_item_types)
    features = base.merge(shares, on="supplier", how="left").fillna(0.0)
    features["log_retail_sales"] = np.log1p(features["retail_sales"].clip(lower=0.0))
    features["log_warehouse_sales"] = np.log1p(features["warehouse_sales"].clip(lower=0.0))
    features["log_retail_transfers"] = np.log1p(features["retail_transfers"].clip(lower=0.0))
    feature_columns = [
        "log_retail_sales",
        "log_warehouse_sales",
        "log_retail_transfers",
        "products_count",
        "active_months",
        "avg_retail_sales_per_month",
        "warehouse_share",
        "transfer_share",
        "negative_sales_share",
        "product_variety",
        *[column for column in features.columns if column.startswith("share_")],
    ]
    return features, feature_columns


def fit_supplier_clusters(features: pd.DataFrame, feature_columns: list[str], ml_config: MLConfig) -> pd.DataFrame:
    """Назначает поставщикам кластеры KMeans по нормированной матрице признаков."""
    clustered = features.copy()
    if clustered.empty:
        raise RuntimeError("Недостаточно данных для кластеризации поставщиков")
    cluster_count = max(1, min(ml_config.cluster_count, len(clustered)))
    if cluster_count == 1:
        clustered["cluster_id"] = 0
        return clustered
    matrix = clustered[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled = StandardScaler().fit_transform(matrix)
    model = KMeans(n_clusters=cluster_count, random_state=ml_config.random_state, n_init=10)
    clustered["cluster_id"] = model.fit_predict(scaled)
    return clustered


def cluster_segment_name(row: pd.Series, medians: dict[str, float]) -> str:
    """Возвращает бизнес-название сегмента по профилю кластера."""
    if row["retail_sales"] >= medians["retail_sales"] and row["products_count_mean"] >= medians["products_count_mean"]:
        return "Крупные мультикатегорийные поставщики"
    if row["warehouse_share_mean"] >= medians["warehouse_share_mean"]:
        return "Поставщики со складским каналом"
    if row["products_count_mean"] < medians["products_count_mean"]:
        return "Нишевые поставщики"
    return "Средний стабильный сегмент"


def cluster_recommendation(row: pd.Series) -> str:
    """Формирует краткую бизнес-рекомендацию для сегмента поставщиков."""
    name = row["segment_name"]
    if "Крупные" in name:
        return "Сохранить приоритет в планировании запасов и использовать сегмент как базу для прогноза спроса."
    if "складским" in name:
        return "Контролировать складскую нагрузку и сравнивать динамику перемещений с фактическими розничными продажами."
    if "Нишевые" in name:
        return "Оценивать ассортимент точечно и не перегружать закупку позициями с низкой повторяемостью спроса."
    return "Использовать стандартный мониторинг продаж и смотреть отклонения от месячного прогноза."


def build_cluster_profiles(clustered: pd.DataFrame) -> pd.DataFrame:
    """Строит портреты кластеров с бизнес-интерпретацией и рекомендациями."""
    profiles = (
        clustered.groupby("cluster_id", as_index=False)
        .agg(
            supplier_count=("supplier", "count"),
            retail_sales=("retail_sales", "sum"),
            warehouse_sales=("warehouse_sales", "sum"),
            retail_transfers=("retail_transfers", "sum"),
            source_rows=("source_rows", "sum"),
            products_count_mean=("products_count", "mean"),
            active_months_mean=("active_months", "mean"),
            warehouse_share_mean=("warehouse_share", "mean"),
            transfer_share_mean=("transfer_share", "mean"),
            negative_sales_share_mean=("negative_sales_share", "mean"),
            avg_retail_sales_per_month_mean=("avg_retail_sales_per_month", "mean"),
        )
        .sort_values("retail_sales", ascending=False)
        .reset_index(drop=True)
    )
    medians = {
        "retail_sales": float(profiles["retail_sales"].median()),
        "products_count_mean": float(profiles["products_count_mean"].median()),
        "warehouse_share_mean": float(profiles["warehouse_share_mean"].median()),
    }
    profiles["segment_name"] = profiles.apply(lambda row: cluster_segment_name(row, medians), axis=1)
    profiles["segment_label"] = profiles.apply(lambda row: f"Сегмент {int(row.cluster_id)}: {row.segment_name}", axis=1)
    profiles["business_recommendation"] = profiles.apply(cluster_recommendation, axis=1)
    return profiles


def attach_segment_labels(clustered: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к строкам поставщиков название сегмента и рекомендацию."""
    labels = profiles[["cluster_id", "segment_name", "segment_label", "business_recommendation"]]
    return clustered.merge(labels, on="cluster_id", how="left").sort_values(["cluster_id", "retail_sales"], ascending=[True, False])


def save_cluster_plot(paths: PathConfig, members: pd.DataFrame) -> Path:
    """Сохраняет график кластеров поставщиков по продажам и складской доле."""
    figure_path = paths.figures_dir / "supplier_clusters.png"
    plotted = members.copy()
    plotted["retail_sales_log"] = np.log1p(plotted["retail_sales"].clip(lower=0.0))
    plotted["products_count_size"] = plotted["products_count"].clip(lower=1.0)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plotted,
        x="retail_sales_log",
        y="warehouse_share",
        hue="segment_name",
        size="products_count_size",
        sizes=(40, 220),
        alpha=0.8,
    )
    plt.title("Кластеры поставщиков")
    plt.xlabel("Логарифм розничных продаж")
    plt.ylabel("Доля складских продаж")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def save_cluster_profile_plot(paths: PathConfig, profiles: pd.DataFrame) -> Path:
    """Сохраняет график размера и продаж по сегментам поставщиков."""
    figure_path = paths.figures_dir / "supplier_cluster_profiles.png"
    plotted = profiles.sort_values("retail_sales", ascending=False)
    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=plotted, x="retail_sales", y="segment_label", hue="supplier_count", dodge=False, palette="viridis")
    plt.title("Портреты кластеров поставщиков")
    plt.xlabel("Розничные продажи")
    plt.ylabel("Сегмент")
    plt.legend(title="Поставщиков")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def build_cluster_report(profiles: pd.DataFrame, members: pd.DataFrame) -> str:
    """Формирует русскоязычный отчет по сегментам поставщиков."""
    profile_lines = []
    for row in profiles.itertuples():
        profile_lines.append(
            f"- {row.segment_label}: поставщиков {row.supplier_count}, розничные продажи {row.retail_sales:.2f}, "
            f"среднее число товаров {row.products_count_mean:.1f}, складская доля {row.warehouse_share_mean:.3f}. "
            f"Рекомендация: {row.business_recommendation}"
        )
    leaders = members.sort_values("retail_sales", ascending=False).head(10)
    leader_lines = "\n".join([f"- {row.supplier}: {row.segment_label}, продажи {row.retail_sales:.2f}" for row in leaders.itertuples()])
    return f"""# Кластеризация поставщиков

## Цель

Я выделил сегменты поставщиков, чтобы кроме прогноза продаж получить управленческую интерпретацию: какие группы дают основной оборот, где заметна складская нагрузка, а где нужен точечный контроль ассортимента.

## Признаки

В кластеризацию вошли логарифмы розничных продаж, складских продаж и retail transfers, количество товаров, число активных месяцев, средние продажи в месяц, доли складского и transfer-канала, доля отрицательных строк и доли продаж по крупнейшим типам товаров.

## Портреты сегментов

{chr(10).join(profile_lines)}

## Крупнейшие поставщики

{leader_lines}

## Рекомендация бизнесу

Основной управленческий фокус стоит держать на крупных мультикатегорийных поставщиках: они формируют наибольший оборот и сильнее всего влияют на точность месячного прогноза. Для сегмента со складским каналом важно отдельно контролировать расхождение между складскими продажами, перемещениями и фактической розницей. Нишевые поставщики лучше анализировать через ассортиментную матрицу, чтобы не увеличивать запас по позициям с нерегулярным спросом.
"""


def cluster_suppliers(paths: PathConfig, ml_config: MLConfig, db_config: DatabaseConfig, clean_path: str | Path) -> dict[str, Any]:
    """Выполняет кластеризацию поставщиков, сохраняет портреты сегментов, график и таблицы PostgreSQL."""
    ensure_directories(paths)
    cleaned = pd.read_csv(clean_path, parse_dates=["period_date", "load_dttm"])
    features, feature_columns = build_supplier_features(cleaned, ml_config)
    clustered = fit_supplier_clusters(features, feature_columns, ml_config)
    profiles = build_cluster_profiles(clustered)
    members = attach_segment_labels(clustered, profiles)
    save_dataframe(paths.reports_dir / "supplier_cluster_profiles.csv", profiles)
    save_dataframe(paths.reports_dir / "supplier_cluster_members.csv", members)
    write_text(paths.reports_dir / "cluster_report.md", build_cluster_report(profiles, members))
    save_cluster_plot(paths, members)
    save_cluster_profile_plot(paths, profiles)
    write_cluster_outputs(db_config, profiles, members)
    return {
        "cluster_count": int(profiles["cluster_id"].nunique()),
        "supplier_count": int(len(members)),
        "profile_rows": int(len(profiles)),
    }
