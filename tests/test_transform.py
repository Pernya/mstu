import pandas as pd

from retail_pipeline.transform import clean_sales_data, stable_hash


def sample_raw_sales() -> pd.DataFrame:
    """Возвращает минимальную тестовую выгрузку продаж."""
    return pd.DataFrame(
        [
            {
                "calendar_year": "2020",
                "cal_month_num": "1",
                "supplier": "Supplier A",
                "item_code": "100",
                "item_description": "Item A",
                "item_type": "WINE",
                "rtl_sales": "10.5",
                "rtl_transfers": "2",
                "whs_sales": "5",
            },
            {
                "calendar_year": "2020",
                "cal_month_num": "1",
                "supplier": "Supplier A",
                "item_code": "100",
                "item_description": "Item A",
                "item_type": "WINE",
                "rtl_sales": "10.5",
                "rtl_transfers": "2",
                "whs_sales": "5",
            },
            {
                "calendar_year": "2020",
                "cal_month_num": "2",
                "supplier": "",
                "item_code": "101",
                "item_description": "Item B",
                "item_type": "BEER",
                "rtl_sales": "7",
                "rtl_transfers": "1",
                "whs_sales": "3",
            },
        ]
    )


def test_stable_hash_is_reproducible() -> None:
    """Проверяет воспроизводимость бизнес-хеша."""
    assert stable_hash("A", 1) == stable_hash(" a ", "1")


def test_clean_sales_data_adds_keys_and_removes_duplicates() -> None:
    """Проверяет очистку, удаление дублей и создание ключей DWH."""
    cleaned = clean_sales_data(sample_raw_sales())
    assert len(cleaned) == 2
    assert {"period_hk", "supplier_hk", "product_hk", "sales_link_hk"}.issubset(cleaned.columns)
    assert cleaned["supplier"].str.len().min() > 0
    assert cleaned["rtl_sales"].dtype == float
