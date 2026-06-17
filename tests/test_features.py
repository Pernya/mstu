import pandas as pd

from retail_pipeline.features import add_calendar_features, add_lag_features, aggregate_monthly_sales, build_next_month_features


def monthly_source() -> pd.DataFrame:
    """Возвращает тестовую историю продаж для feature engineering."""
    rows = []
    for month in range(1, 7):
        rows.append(
            {
                "period_date": pd.Timestamp(year=2020, month=month, day=1),
                "calendar_year": 2020,
                "cal_month_num": month,
                "supplier": "Supplier A",
                "item_type": "WINE",
                "item_code": f"10{month}",
                "sales_link_hk": f"link-{month}",
                "rtl_sales": float(month * 10),
                "rtl_transfers": float(month),
                "whs_sales": float(month * 2),
            }
        )
    return pd.DataFrame(rows)


def test_feature_engineering_builds_lags() -> None:
    """Проверяет расчет лагов и скользящих признаков."""
    aggregated = aggregate_monthly_sales(monthly_source(), top_suppliers=3)
    featured = add_lag_features(add_calendar_features(aggregated))
    complete = featured.dropna(subset=["sales_lag_1", "sales_lag_2", "sales_lag_3"])
    assert len(complete) == 3
    assert complete.iloc[0]["sales_lag_1"] == 30.0
    assert "warehouse_share_lag_1" in complete.columns
    assert "source_rows_lag_1" in complete.columns


def test_next_month_features_use_last_history() -> None:
    """Проверяет формирование признаков для прогноза следующего месяца."""
    aggregated = aggregate_monthly_sales(monthly_source(), top_suppliers=3)
    featured = add_lag_features(add_calendar_features(aggregated)).dropna(subset=["sales_lag_1", "sales_lag_2", "sales_lag_3"])
    future = build_next_month_features(featured)
    assert len(future) == 1
    assert future.iloc[0]["cal_month_num"] == 7
    assert future.iloc[0]["sales_lag_1"] == 60.0
