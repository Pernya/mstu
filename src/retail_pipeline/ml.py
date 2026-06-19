from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from retail_pipeline.config import DatabaseConfig, MLConfig, PathConfig
from retail_pipeline.features import build_next_month_features
from retail_pipeline.io_utils import ensure_directories, save_dataframe, write_json, write_text
from retail_pipeline.warehouse import write_ml_outputs


TARGET_COLUMN = "retail_sales"
DATE_COLUMN = "period_date"


def detect_removed_features(dataframe: pd.DataFrame, ml_config: MLConfig) -> pd.DataFrame:
    """Определяет признаки, которые нужно исключить из обучения, и причину исключения."""
    rows = []
    leakage_columns = {
        TARGET_COLUMN,
        "rtl_sales",
        "retail_sales_current",
        "retail_transfers",
        "warehouse_sales",
        "source_rows",
        "products_count",
    }
    technical_fragments = ["_hk", "hash", "hashdiff", "record_source", "load_dttm"]
    text_columns = {"item_description", "item_code", "supplier"}
    for column in dataframe.columns:
        reason = None
        if column == DATE_COLUMN:
            reason = "дата используется только для временного разбиения"
        elif column in leakage_columns:
            reason = "утечка целевой переменной"
        elif column in text_columns:
            reason = "сырое текстовое поле заменено агрегированными признаками"
        elif any(fragment in column.lower() for fragment in technical_fragments):
            reason = "технический идентификатор или служебное поле"
        elif dataframe[column].isna().mean() > ml_config.max_missing_share:
            reason = "слишком высокая доля пропусков"
        elif dataframe[column].nunique(dropna=True) <= 1:
            reason = "нулевая или почти нулевая дисперсия"
        elif dataframe[column].dtype == "object" and dataframe[column].nunique(dropna=True) > ml_config.max_category_cardinality:
            reason = "слишком высокая кардинальность категориального признака"
        if reason:
            rows.append({"feature": column, "reason": reason})
    return pd.DataFrame(rows)


def select_relevant_features(dataframe: pd.DataFrame, ml_config: MLConfig) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Исключает нерелевантные признаки и возвращает матрицу признаков, таргет и причины удаления."""
    removed = detect_removed_features(dataframe, ml_config)
    removed_features = set(removed["feature"].tolist()) if not removed.empty else set()
    feature_columns = [column for column in dataframe.columns if column not in removed_features]
    features = dataframe[feature_columns].copy()
    target = dataframe[TARGET_COLUMN].astype(float)
    return features, target, removed


def split_temporal(features: pd.DataFrame, target: pd.Series, source: pd.DataFrame, test_months: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Делит данные на train и test по последним календарным месяцам."""
    periods = sorted(pd.to_datetime(source[DATE_COLUMN]).dropna().unique())
    if len(periods) <= test_months:
        boundary = periods[max(1, len(periods) // 2)]
    else:
        boundary = periods[-test_months]
    mask = pd.to_datetime(source[DATE_COLUMN]) < boundary
    if mask.sum() == 0 or (~mask).sum() == 0:
        mask = np.arange(len(source)) < int(len(source) * 0.8)
    return features.loc[mask], features.loc[~mask], target.loc[mask], target.loc[~mask]


def build_model(features: pd.DataFrame, ml_config: MLConfig) -> Pipeline:
    """Создает sklearn Pipeline с предобработкой и компактным RandomForestRegressor."""
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in features.columns if column not in numeric_features]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=ml_config.estimators,
        max_depth=9,
        min_samples_leaf=3,
        random_state=ml_config.random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Рассчитывает основные метрики качества регрессионной модели."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_encoded_feature_names(model: Pipeline) -> list[str]:
    """Возвращает имена признаков после preprocessing-шага sklearn."""
    preprocessor = model.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out().tolist()


def base_feature_name(feature: str) -> str:
    """Приводит имя one-hot или pipeline-признака к базовому исходному признаку."""
    cleaned = feature.replace("num__", "").replace("cat__", "")
    for prefix in ["item_type_", "supplier_group_"]:
        if cleaned.startswith(prefix):
            return prefix.rstrip("_")
    return cleaned


def normalized_scores(scores: pd.Series) -> pd.Series:
    """Нормирует значения скоринга в диапазон от нуля до единицы."""
    absolute = scores.abs().fillna(0.0)
    maximum = absolute.max()
    if maximum == 0 or pd.isna(maximum):
        return absolute
    return absolute / maximum


def correlation_scores(features: pd.DataFrame, target: pd.Series, method: str) -> pd.DataFrame:
    """Считает Pearson или Spearman корреляцию для числовых признаков."""
    numeric = features.select_dtypes(include=["number", "bool"])
    rows = []
    for column in numeric.columns:
        score = numeric[column].corr(target, method=method)
        rows.append({"feature": column, "method": method, "score": 0.0 if pd.isna(score) else float(score)})
    return pd.DataFrame(rows)


def encoded_feature_scores(model: Pipeline, features: pd.DataFrame, target: pd.Series, ml_config: MLConfig) -> pd.DataFrame:
    """Считает mutual information, f-regression и важность RandomForest на закодированных признаках."""
    transformed = model.named_steps["preprocessor"].transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = get_encoded_feature_names(model)
    mi = mutual_info_regression(transformed, target, random_state=ml_config.random_state)
    f_values, _ = f_regression(transformed, target)
    forest = model.named_steps["model"].feature_importances_
    rows = []
    for name, score in zip(names, mi):
        rows.append({"feature": base_feature_name(name), "method": "mutual_info", "score": float(score)})
    for name, score in zip(names, np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0)):
        rows.append({"feature": base_feature_name(name), "method": "f_regression", "score": float(score)})
    for name, score in zip(names, forest):
        rows.append({"feature": base_feature_name(name), "method": "random_forest", "score": float(score)})
    return pd.DataFrame(rows)


def permutation_scores(model: Pipeline, features: pd.DataFrame, target: pd.Series, ml_config: MLConfig) -> pd.DataFrame:
    """Считает permutation importance на исходных признаках test-набора."""
    result = permutation_importance(model, features, target, n_repeats=8, random_state=ml_config.random_state, n_jobs=-1)
    return pd.DataFrame(
        {
            "feature": features.columns,
            "method": "permutation",
            "score": result.importances_mean,
        }
    )


def build_feature_ranking(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, ml_config: MLConfig) -> pd.DataFrame:
    """Формирует общий рейтинг признаков по нескольким независимым скорингам."""
    frames = [
        correlation_scores(x_train, y_train, "pearson"),
        correlation_scores(x_train, y_train, "spearman"),
        encoded_feature_scores(model, x_train, y_train, ml_config),
        permutation_scores(model, x_test, y_test, ml_config),
    ]
    long_scores = pd.concat(frames, ignore_index=True)
    long_scores["feature"] = long_scores["feature"].map(base_feature_name)
    long_scores["normalized_score"] = long_scores.groupby("method")["score"].transform(normalized_scores)
    ranking = (
        long_scores.pivot_table(index="feature", columns="method", values="normalized_score", aggfunc="max")
        .fillna(0.0)
        .reset_index()
    )
    method_columns = [column for column in ranking.columns if column != "feature"]
    ranking["total_score"] = ranking[method_columns].mean(axis=1)
    return ranking.sort_values("total_score", ascending=False).reset_index(drop=True)


def save_feature_importance_plot(paths: PathConfig, ranking: pd.DataFrame) -> Path:
    """Сохраняет график итоговой важности признаков."""
    figure_path = paths.figures_dir / "feature_importance.png"
    visible = ranking.head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=visible, x="total_score", y="feature")
    plt.title("Итоговый рейтинг признаков")
    plt.xlabel("Итоговый нормированный скоринг")
    plt.ylabel("Признак")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def save_feature_score_heatmap(paths: PathConfig, ranking: pd.DataFrame) -> Path:
    """Сохраняет тепловую карту скорингов признаков по разным методам."""
    figure_path = paths.figures_dir / "feature_scores_heatmap.png"
    method_columns = [column for column in ranking.columns if column not in {"feature", "total_score"}]
    visible = ranking.head(15).set_index("feature")[method_columns]
    plt.figure(figsize=(10, 7))
    sns.heatmap(visible, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Скоринги признаков по методам")
    plt.xlabel("Метод")
    plt.ylabel("Признак")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def save_actual_vs_predicted_plot(paths: PathConfig, y_true: pd.Series, y_pred: np.ndarray) -> Path:
    """Сохраняет график сравнения фактических и прогнозных значений."""
    figure_path = paths.figures_dir / "actual_vs_predicted.png"
    plotted = pd.DataFrame({"fact": y_true.values, "forecast": y_pred})
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=plotted, x="fact", y="forecast")
    maximum = float(max(plotted["fact"].max(), plotted["forecast"].max()))
    minimum = float(min(plotted["fact"].min(), plotted["forecast"].min()))
    plt.plot([minimum, maximum], [minimum, maximum], color="red")
    plt.title("Факт против прогноза")
    plt.xlabel("Факт")
    plt.ylabel("Прогноз")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def save_model_generalization_plot(paths: PathConfig, metrics: dict[str, Any]) -> Path:
    """Сохраняет сравнение train и test метрик для контроля переобучения."""
    figure_path = paths.figures_dir / "model_generalization.png"
    plotted = pd.DataFrame(
        [
            {"metric": "MAE", "sample": "train", "value": metrics["train_mae"]},
            {"metric": "MAE", "sample": "test", "value": metrics["test_mae"]},
            {"metric": "RMSE", "sample": "train", "value": metrics["train_rmse"]},
            {"metric": "RMSE", "sample": "test", "value": metrics["test_rmse"]},
            {"metric": "R2", "sample": "train", "value": metrics["train_r2"]},
            {"metric": "R2", "sample": "test", "value": metrics["test_r2"]},
        ]
    )
    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        sns.barplot(data=plotted[plotted["metric"] == metric], x="sample", y="value", hue="sample", ax=axis, palette=["#4E79A7", "#F28E2B"], legend=False)
        axis.set_title(metric)
        axis.set_xlabel("")
        axis.set_ylabel("Значение")
    plt.suptitle("Сравнение качества на train и test")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def save_residual_diagnostics_plot(paths: PathConfig, y_true: pd.Series, y_pred: np.ndarray) -> Path:
    """Сохраняет диагностику ошибок регрессионной модели."""
    figure_path = paths.figures_dir / "residual_diagnostics.png"
    residuals = pd.Series(y_true.values - y_pred, name="residual")
    plotted = pd.DataFrame({"forecast": y_pred, "residual": residuals})
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(residuals, kde=True, ax=axes[0], color="#4E79A7")
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title("Распределение ошибок")
    axes[0].set_xlabel("Факт минус прогноз")
    axes[0].set_ylabel("Количество")
    sns.scatterplot(data=plotted, x="forecast", y="residual", ax=axes[1], color="#59A14F")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Ошибки относительно прогноза")
    axes[1].set_xlabel("Прогноз")
    axes[1].set_ylabel("Ошибка")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def save_regression_learning_curve(paths: PathConfig, model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> Path:
    """Сохраняет learning curve по MAE для проверки устойчивости модели."""
    figure_path = paths.figures_dir / "regression_learning_curve.png"
    fractions = np.linspace(0.35, 1.0, 6)
    sizes = sorted({max(8, min(len(x_train), int(len(x_train) * fraction))) for fraction in fractions})
    rows = []
    for size in sizes:
        candidate = clone(model)
        candidate.fit(x_train.iloc[:size], y_train.iloc[:size])
        train_pred = candidate.predict(x_train.iloc[:size])
        test_pred = candidate.predict(x_test)
        rows.append({"train_rows": size, "sample": "train", "mae": mean_absolute_error(y_train.iloc[:size], train_pred)})
        rows.append({"train_rows": size, "sample": "test", "mae": mean_absolute_error(y_test, test_pred)})
    plotted = pd.DataFrame(rows)
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=plotted, x="train_rows", y="mae", hue="sample", marker="o")
    plt.title("Learning curve регрессионной модели")
    plt.xlabel("Количество train-строк")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def save_forecast_plot(paths: PathConfig, forecast: pd.DataFrame) -> Path:
    """Сохраняет график прогноза следующего месяца по типам товаров."""
    figure_path = paths.figures_dir / "forecast_next_month.png"
    plotted = forecast.groupby("item_type", as_index=False)["forecast_retail_sales"].sum().sort_values("forecast_retail_sales", ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plotted, x="forecast_retail_sales", y="item_type", color="#B07AA1")
    plt.title("Прогноз розничных продаж на следующий месяц")
    plt.xlabel("Прогноз продаж")
    plt.ylabel("Тип товара")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return figure_path


def build_forecast(model: Pipeline, feature_table: pd.DataFrame, selected_columns: list[str]) -> pd.DataFrame:
    """Строит прогноз продаж на следующий месяц по последним лаговым признакам."""
    future = build_next_month_features(feature_table)
    if future.empty:
        raise RuntimeError("Недостаточно истории для прогноза следующего месяца")
    predictions = model.predict(future[selected_columns])
    forecast = future[[DATE_COLUMN, "calendar_year", "cal_month_num", "item_type", "supplier_group"]].copy()
    forecast["forecast_retail_sales"] = predictions.clip(min=0)
    return forecast


def build_model_report(metrics: dict[str, Any], removed: pd.DataFrame, ranking: pd.DataFrame) -> str:
    """Формирует русскоязычный отчет о модели, признаках и качестве прогноза."""
    removed_count = len(removed)
    top_features = ranking.head(10)
    feature_lines = "\n".join([f"- {row.feature}: {row.total_score:.4f}" for row in top_features.itertuples()])
    removed_lines = "\n".join([f"- {row.feature}: {row.reason}" for row in removed.itertuples()]) if removed_count else "- Исключенных признаков нет"
    overfit_status = "критических признаков переобучения не выявлено" if metrics["r2_gap"] <= 0.15 else "нужна дополнительная проверка из-за заметного разрыва R2"
    return f"""# Отчет по ML-модели

## Постановка задачи

Модель прогнозирует месячные розничные продажи `retail_sales` по агрегированным данным о типе товара, группе поставщика, сезонности, лагах продаж и складских показателях.

## Метрики на тестовой выборке

- MAE: {metrics["mae"]:.4f}
- RMSE: {metrics["rmse"]:.4f}
- R2: {metrics["r2"]:.4f}
- Количество train-строк: {metrics["train_rows"]}
- Количество test-строк: {metrics["test_rows"]}

## Контроль переобучения

Precision curve не используется, потому что задача является регрессионной, а не классификационной. Для проверки переобучения применены временное train/test-разбиение, сравнение train и test метрик, learning curve и анализ остатков.

- Train MAE: {metrics["train_mae"]:.4f}
- Test MAE: {metrics["test_mae"]:.4f}
- Train RMSE: {metrics["train_rmse"]:.4f}
- Test RMSE: {metrics["test_rmse"]:.4f}
- Train R2: {metrics["train_r2"]:.4f}
- Test R2: {metrics["test_r2"]:.4f}
- Разрыв R2: {metrics["r2_gap"]:.4f}
- Вывод: {overfit_status}.

Для визуальной проверки сохраняются графики `model_generalization.png`, `regression_learning_curve.png`, `residual_diagnostics.png` и `actual_vs_predicted.png`.

## Исключенные признаки

{removed_lines}

## Топ признаков по совокупному скорингу

{feature_lines}

## Интерпретация

В общий рейтинг включены Pearson, Spearman, mutual information, f-regression, permutation importance и важность RandomForest. Такой подход снижает риск опираться на один скоринг и показывает признаки, устойчиво полезные для прогноза.
"""


def train_model(paths: PathConfig, ml_config: MLConfig, db_config: DatabaseConfig, feature_path: str | Path) -> dict[str, Any]:
    """Обучает модель прогноза, сохраняет артефакты, отчеты, графики и прогнозы в DWH."""
    ensure_directories(paths)
    feature_table = pd.read_csv(feature_path, parse_dates=[DATE_COLUMN])
    features, target, removed = select_relevant_features(feature_table, ml_config)
    x_train, x_test, y_train, y_test = split_temporal(features, target, feature_table, ml_config.test_months)
    model = build_model(x_train, ml_config)
    model.fit(x_train, y_train)
    train_predictions = model.predict(x_train)
    predictions = model.predict(x_test)
    train_metrics = calculate_metrics(y_train, train_predictions)
    test_metrics = calculate_metrics(y_test, predictions)
    metrics = {
        **test_metrics,
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "train_r2": train_metrics["r2"],
        "r2_gap": train_metrics["r2"] - test_metrics["r2"],
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "model_type": "RandomForestRegressor",
    }
    ranking = build_feature_ranking(model, x_train, y_train, x_test, y_test, ml_config)
    forecast = build_forecast(model, feature_table, list(features.columns))
    model_path = paths.models_dir / "retail_sales_forecast.joblib"
    joblib.dump({"model": model, "selected_columns": list(features.columns), "metrics": metrics}, model_path)
    save_dataframe(paths.reports_dir / "feature_ranking.csv", ranking)
    save_dataframe(paths.reports_dir / "forecast_next_month.csv", forecast)
    write_json(paths.reports_dir / "ml_metrics.json", metrics)
    write_text(paths.reports_dir / "model_report.md", build_model_report(metrics, removed, ranking))
    save_feature_importance_plot(paths, ranking)
    save_feature_score_heatmap(paths, ranking)
    save_actual_vs_predicted_plot(paths, y_test, predictions)
    save_model_generalization_plot(paths, metrics)
    save_residual_diagnostics_plot(paths, y_test, predictions)
    save_regression_learning_curve(paths, model, x_train, y_train, x_test, y_test)
    save_forecast_plot(paths, forecast)
    write_ml_outputs(db_config, metrics, forecast)
    return {"model_path": str(model_path), "forecast_rows": len(forecast), **metrics}
