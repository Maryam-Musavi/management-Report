"""
**************************************************
Gam Chatbot Database Module
**************************************************
ماژول مدیریت پایگاه داده SQLite برای پروژه اوراق گام

چرا SQLite به‌جای Excel؟
  - داده به‌صورت پایدار روی دیسک ذخیره می‌شود؛ نیازی به آپلود
    دوباره فایل در هر بار اجرای برنامه نیست.
  - امکان فیلتر/شمارش دقیق با SQL به‌جای پردازش کامل فایل در حافظه.
  - ساختار تمیزتر و type-safe نسبت به سلول‌های آزاد اکسل.
  - آماده برای اتصال به سیستم‌های دیگر (مثلاً CRM) که مستقیماً
    در پایگاه داده می‌نویسند.
**************************************************
"""

from typing import Optional
import os
import sqlite3
import pandas as pd

# مسیر پایگاه داده — کنار همین فایل، در پوشه data/
_BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DB_PATH: str = os.path.join(_BASE_DIR, "data", "gam_customers.db")

TABLE_NAME: str = "customers"

# ستون‌های جدول (برای ساخت جدول تازه در صورت نبودن فایل)
_SCHEMA_SQL: str = f'''
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    "ردیف" INTEGER,
    "واحد درخواست کننده" TEXT,
    "نام متقاضی" TEXT,
    "نوع درخواست کننده" TEXT,
    "نوع کالای درخواستی" TEXT,
    "وضعیت تامین کننده" TEXT,
    "تاریخ" TEXT,
    "نحوه معرفی" TEXT,
    "شماره تماس" TEXT,
    "اقدامات انجام شده" TEXT
)
'''


def get_connection() -> sqlite3.Connection:
    """اتصال به پایگاه داده SQLite (فایل در صورت نبود، ساخته می‌شود)."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA_SQL)
    conn.commit()
    return conn


def load_dataframe() -> Optional[pd.DataFrame]:
    """
    خواندن کل جدول customers از SQLite به‌صورت DataFrame.
    اگر جدول خالی باشد یا فایل پایگاه داده وجود نداشته باشد، None برمی‌گردد.
    """
    if not os.path.exists(DB_PATH):
        return None

    conn = get_connection()
    try:
        df = pd.read_sql_query(f'SELECT * FROM {TABLE_NAME}', conn)
    finally:
        conn.close()

    if df.empty:
        return None

    # ستون فنی id را حذف کن — در تحلیل استفاده نمی‌شود
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df = df.fillna("")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace("None", "")

    return df


def import_dataframe(df: pd.DataFrame, replace: bool = True) -> None:
    """
    نوشتن یک DataFrame در پایگاه داده (برای وارد کردن فایل جدید از UI).

    Args:
        df: داده‌ای که باید ذخیره شود (باید همان ستون‌های جدول customers را داشته باشد)
        replace: اگر True، جدول قبلی پاک و جایگزین می‌شود
    """
    conn = get_connection()
    try:
        if replace:
            conn.execute(f'DELETE FROM {TABLE_NAME}')
            conn.commit()
        # اطمینان از وجود ستون شماره تماس، حتی اگر فایل ورودی نداشته باشد
        if "شماره تماس" not in df.columns:
            df = df.copy()
            df["شماره تماس"] = None
        df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def database_exists_with_data() -> bool:
    """بررسی اینکه آیا پایگاه داده موجود و دارای رکورد است یا نه."""
    if not os.path.exists(DB_PATH):
        return False
    conn = get_connection()
    try:
        count = conn.execute(f'SELECT COUNT(*) FROM {TABLE_NAME}').fetchone()[0]
        return count > 0
    finally:
        conn.close()
