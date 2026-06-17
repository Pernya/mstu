# Airflow: граф пайплайна и проверка запуска

## Что показывает Airflow

Airflow автоматически строит граф по зависимостям задач внутри `dags/retail_sales_ml_dwh.py`. Отдельно рисовать граф для интерфейса не нужно: вкладка `Graph` берет порядок выполнения прямо из DAG.

## Как показать пайплайн

1. Открыть `http://localhost:8080`.
2. Войти под `airflow / airflow`.
3. Открыть DAG `retail_sales_ml_dwh`.
4. Перейти на вкладку `Graph`.
5. Показать цепочку `extract_metadata`, `extract_sales`, `transform_sales`, `load_data_warehouse`, `build_dashboard_marts`, `run_eda`, `engineer_features`, `build_supplier_segments`, `train_forecast_model`, `validate_pipeline_outputs`.
6. Перейти на вкладку `Grid` и показать успешное состояние задач.
7. Открыть лог `train_forecast_model`, чтобы показать обучение модели внутри Airflow.
8. Открыть лог `build_supplier_segments`, чтобы показать кластеризацию поставщиков.
9. Открыть лог `validate_pipeline_outputs`, чтобы показать финальную проверку файлов и таблиц.

## Где находятся графики

Airflow не является BI-инструментом. В этом проекте он запускает задачи, которые строят графики и сохраняют их как файлы:

```text
reports/figures/monthly_sales.png
reports/figures/item_type_sales.png
reports/figures/sales_distribution.png
reports/figures/top_suppliers.png
reports/figures/feature_importance.png
reports/figures/actual_vs_predicted.png
reports/figures/supplier_clusters.png
reports/figures/supplier_cluster_profiles.png
```

Для защиты удобно показать `Graph` и `Grid` в Airflow как доказательство оркестрации, а сами аналитические графики открыть из `reports/figures` или собрать по витринам в Superset.

## CLI-проверки

Список DAG:

```bash
docker compose -f docker-compose.etl.yml --env-file .env exec airflow-scheduler airflow dags list
```

Запуски DAG:

```bash
docker compose -f docker-compose.etl.yml --env-file .env exec airflow-scheduler airflow dags list-runs -d retail_sales_ml_dwh
```

Ручной запуск DAG:

```bash
docker compose -f docker-compose.etl.yml --env-file .env exec airflow-scheduler airflow dags trigger retail_sales_ml_dwh
```

## Формулировка для защиты

Airflow отвечает за оркестрацию всего процесса: загрузку данных, очистку, загрузку DWH, построение витрин, EDA, генерацию признаков, кластеризацию поставщиков, обучение модели, сохранение прогноза и финальную валидацию. Граф на вкладке `Graph` строится автоматически по зависимостям задач, поэтому он показывает фактическую последовательность выполнения, а не отдельную нарисованную схему.
