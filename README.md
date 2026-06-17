# Анализ и прогнозирование розничных продаж с применением Apache Airflow и методов машинного обучения

Проект реализует рабочий ETL, DWH и ML-пайплайн для анализа розничных продаж. Я использовал официальный датасет Montgomery County Warehouse and Retail Sales, загрузку через Socrata API, Apache Airflow для оркестрации, PostgreSQL для хранилища, Data Vault 2.0 как слой интеграции, Apache Superset для дашбордов и `RandomForestRegressor` для прогноза месячных retail sales.

В репозитории лежит только код, конфигурация и документация для запуска. Данные, модели, отчеты, графики и файлы защиты генерируются локально после запуска DAG и не хранятся в Git.

## Быстрый запуск

```bash
cp .env.example .env
docker compose -f docker-compose.etl.yml --env-file .env up -d --build
```

Airflow будет доступен по адресу `http://localhost:8080`.

Логин и пароль по умолчанию:

```text
airflow / airflow
```

После запуска нужно открыть DAG `retail_sales_ml_dwh` и запустить его вручную. Пайплайн загрузит ограниченную выборку данных, построит Data Vault, витрины, EDA, признаки, модель прогноза, кластеризацию поставщиков и проверки результата.

## Superset

Superset запускается отдельно от ETL:

```bash
docker compose -f docker-compose.superset.yml --env-file .env up -d --build
```

Интерфейс будет доступен по адресу `http://localhost:8088`.

Логин и пароль по умолчанию:

```text
admin / admin
```

Подключение к DWH в Superset настроено через адрес хоста:

```text
postgresql+psycopg2://retail_user:retail_password@host.docker.internal:5433/retail_dwh
```

Так Superset, запущенный в отдельном compose-проекте, обращается к PostgreSQL, который опубликован ETL compose на порту `5433` хост-машины.

## Структура

```text
dags/                     DAG Apache Airflow
src/retail_pipeline/      Python-пакет ETL, DWH, EDA, ML и кластеризации
docker/postgres/init/     инициализация PostgreSQL
docker/superset/          образ и инициализация Superset
docs/                     русская документация и исходники диаграмм
tests/                    pytest-проверки основных модулей
data/                     локальные данные после запуска
models/                   локальная модель после обучения
reports/                  локальные отчеты и графики после запуска
logs/                     локальные логи Airflow
```

## DAG

`retail_sales_ml_dwh` состоит из шагов:

- `extract_metadata` загружает метаданные колонок источника;
- `extract_sales` забирает ограниченную месячную выборку из Socrata API;
- `transform_sales` очищает данные, типизирует поля и формирует бизнес-ключи;
- `load_data_warehouse` загружает `raw` и Data Vault слой;
- `build_dashboard_marts` строит витрины для анализа и Superset;
- `run_eda` формирует EDA-отчет и графики;
- `engineer_features` создает признаки для прогноза;
- `build_supplier_segments` кластеризует поставщиков и сохраняет портреты сегментов;
- `train_forecast_model` обучает модель, считает рейтинг признаков и прогноз;
- `validate_pipeline_outputs` проверяет файлы и строки в ключевых таблицах.

## Хранилище

PostgreSQL содержит четыре схемы:

- `raw` хранит нормализованные исходные строки и метаданные источника;
- `dv` хранит Data Vault слой: `hub_period`, `hub_supplier`, `hub_product`, `link_sales`, `sat_product`, `sat_sales_metrics`;
- `mart` хранит витрины `dm_monthly_sales`, `dm_item_type_sales`, `dm_supplier_sales`, `dm_ml_sales`;
- `ml` хранит `model_metrics`, `sales_forecast`, `supplier_cluster_profiles`, `supplier_cluster_members`.

Бизнес-ключи формируются из устойчивых полей источника:

- период: `PERIOD + calendar_year + cal_month_num`;
- поставщик: `SUPPLIER + supplier`;
- товар: `PRODUCT + item_code + item_description`;
- факт продаж: `SALE + calendar_year + cal_month_num + supplier + item_code + item_type`.

Перед хешированием значения приводятся к строке, обрезаются по краям и переводятся в верхний регистр. Это нужно, чтобы одинаковый поставщик или товар не получал разные ключи из-за регистра или лишних пробелов.

## ML

Целевая переменная: месячные розничные продажи `retail_sales`.

В признаки входят календарные признаки, циклическое кодирование месяца, тип товара, группа поставщика, лаги продаж, скользящие статистики, доли складского канала и retail transfers. Технические хеши, сырые текстовые описания, дата периода, таргет, признаки с высокой долей пропусков, нулевой дисперсией и слишком высокой кардинальностью удаляются автоматически.

Рейтинг признаков строится несколькими способами: Pearson, Spearman, mutual information, f-regression, permutation importance и важность RandomForest. Итоговый рейтинг сохраняется в `reports/feature_ranking.csv`.

## Кластеризация

Поставщики сегментируются методом KMeans. В модель кластеризации входят логарифмы продаж и перемещений, количество товаров, активные месяцы, средние продажи в месяц, доли складских продаж, доли transfers, доля отрицательных строк и структура продаж по крупнейшим типам товаров.

Результаты:

- `reports/cluster_report.md`;
- `reports/supplier_cluster_profiles.csv`;
- `reports/supplier_cluster_members.csv`;
- `reports/figures/supplier_clusters.png`;
- `ml.supplier_cluster_profiles`;
- `ml.supplier_cluster_members`.

Кластеризация нужна не для прогноза как такового, а для бизнес-интерпретации: видно, какие поставщики формируют основной оборот, где высока складская нагрузка и где нужен точечный контроль ассортимента.

## Результаты после запуска

Пайплайн локально создает:

- `data/raw/sales_raw.csv`;
- `data/stage/sales_clean.csv`;
- `data/mart/*.csv`;
- `models/retail_sales_forecast.joblib`;
- `reports/eda_report.md`;
- `reports/model_report.md`;
- `reports/cluster_report.md`;
- `reports/ml_metrics.json`;
- `reports/feature_ranking.csv`;
- `reports/forecast_next_month.csv`;
- `reports/figures/*.png`.

Эти файлы намеренно исключены из Git, потому что они зависят от запуска, настроек `.env` и текущего состояния источника.

## Визуализации по шагам

Пайплайн сохраняет графики не только после обучения модели, а на каждом ключевом этапе:

- extract: `extract_period_coverage.png`, `extract_rows_by_month.png`;
- transform: `transform_quality_summary.png`, `transform_metric_boxplot.png`;
- DWH: `dwh_table_counts.png`, `mart_table_counts.png`;
- EDA: `monthly_sales.png`, `item_type_sales.png`, `sales_distribution.png`, `top_suppliers.png`;
- feature engineering: `feature_rows_by_item_type.png`, `feature_correlation_heatmap.png`;
- ML: `feature_importance.png`, `feature_scores_heatmap.png`, `actual_vs_predicted.png`, `forecast_next_month.png`;
- clustering: `supplier_clusters.png`, `supplier_cluster_profiles.png`.

Так проще объяснять работу на защите: каждый этап оставляет не только таблицу или CSV, но и визуальный контроль результата.

## Проверки

Проверка compose-файлов:

```bash
docker compose -f docker-compose.etl.yml --env-file .env config
docker compose -f docker-compose.superset.yml --env-file .env config
```

Проверка Python-модулей:

```bash
python -m pytest
```

Проверка внутри контейнера Airflow:

```bash
docker compose -f docker-compose.etl.yml --env-file .env run --rm airflow-scheduler pytest
```

Проверка таблиц:

```bash
docker compose -f docker-compose.etl.yml --env-file .env exec postgres psql -U retail_user -d retail_dwh -c "select count(*) from raw.sales_raw;"
docker compose -f docker-compose.etl.yml --env-file .env exec postgres psql -U retail_user -d retail_dwh -c "select count(*) from mart.dm_item_type_sales;"
docker compose -f docker-compose.etl.yml --env-file .env exec postgres psql -U retail_user -d retail_dwh -c "select count(*) from ml.sales_forecast;"
docker compose -f docker-compose.etl.yml --env-file .env exec postgres psql -U retail_user -d retail_dwh -c "select count(*) from ml.supplier_cluster_profiles;"
```

## Диаграммы

Исходники диаграмм находятся в `docs/diagrams`.

Опциональный рендер PlantUML:

```bash
plantuml docs/diagrams/solution_architecture.puml
plantuml docs/diagrams/etl_activity.puml
plantuml docs/diagrams/data_vault_schema.puml
plantuml docs/diagrams/archimate_solution.puml
```

Опциональный рендер Mermaid:

```bash
mmdc -i docs/diagrams/data_architecture.mmd -o docs/diagrams/data_architecture.png
mmdc -i docs/diagrams/ml_pipeline.mmd -o docs/diagrams/ml_pipeline.png
```

## Дополнительная документация

- `docs/data_dictionary.md` описывает источник, схемы, витрины, признаки и бизнес-ключи;
- `docs/implementation_notes.md` фиксирует технические тонкости реализации;
- `docs/dashboard_plan.md` описывает построение дашбордов в Superset;
- `docs/airflow_graphs.md` объясняет, что показывать в Airflow.
- `docs/visualizations.md` перечисляет графики пайплайна и смысл каждого файла.
