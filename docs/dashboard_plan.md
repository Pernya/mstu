# План построения дашбордов в Superset

## Подготовка

1. Запустить ETL compose:

```bash
docker compose -f docker-compose.etl.yml --env-file .env up -d --build
```

2. Открыть Airflow на `http://localhost:8080`.
3. Запустить DAG `retail_sales_ml_dwh`.
4. Дождаться успешного выполнения `validate_pipeline_outputs`.
5. Запустить Superset:

```bash
docker compose -f docker-compose.superset.yml --env-file .env up -d --build
```

6. Открыть `http://localhost:8088` и войти под `admin / admin`.

## Подключение к DWH

Подключение `Retail DWH` создается автоматически при старте `superset-init`. Если его нужно создать вручную, в Superset нужно открыть `Settings`, затем `Database Connections`, добавить PostgreSQL и указать SQLAlchemy URI:

```text
postgresql+psycopg2://retail_user:retail_password@host.docker.internal:5433/retail_dwh
```

Адрес `host.docker.internal` используется потому, что Superset запущен в отдельном compose-проекте. Он подключается не к контейнерному имени `postgres`, а к порту PostgreSQL, опубликованному на хосте.

## Наборы данных

Для дашборда я добавляю datasets:

- `mart.dm_monthly_sales`;
- `mart.dm_item_type_sales`;
- `mart.dm_supplier_sales`;
- `mart.dm_ml_sales`;
- `ml.sales_forecast`;
- `ml.supplier_cluster_profiles`;
- `ml.supplier_cluster_members`.

## Рекомендуемые графики

1. KPI `sum(retail_sales)` по `mart.dm_monthly_sales`.
2. Линейный график `period_date` -> `retail_sales` по `mart.dm_monthly_sales`.
3. Столбчатый график продаж по `item_type` из `mart.dm_item_type_sales`.
4. Top-N поставщиков по `retail_sales` из `mart.dm_supplier_sales`.
5. Таблица прогноза по `ml.sales_forecast`: период, тип товара, группа поставщика, прогноз.
6. Scatter-график кластеров из `ml.supplier_cluster_members`: `retail_sales`, `warehouse_share`, цвет по `segment_name`, размер по `products_count`.
7. Таблица портретов сегментов из `ml.supplier_cluster_profiles` с рекомендациями.
8. Bar chart сегментов из `ml.supplier_cluster_profiles`: `segment_label` и `retail_sales`, цвет по `supplier_count`.

## Логика дашборда

Верхний блок показывает общий оборот и динамику по месяцам. Средний блок объясняет структуру продаж по типам товаров и поставщикам. Нижний блок показывает прогноз и кластеризацию: это связывает технический ML-результат с бизнес-рекомендациями.

## Что показать на защите

Сначала я показываю Airflow `Graph`, чтобы подтвердить автоматизацию. Затем открываю Superset и показываю, что витрины читаются из PostgreSQL. После этого демонстрирую график продаж по месяцам, структуру по типам товаров, топ поставщиков, прогноз и сегменты поставщиков. Такой порядок удобен: сначала видно, как данные проходят пайплайн, потом видно, какой аналитический результат получается на выходе.
