"""
**************************************************
Build / Clean Gam Customers Database
**************************************************
اسکریپت مستقل برای تبدیل یک فایل خام SQLite (خروجی مستقیم از اکسل یا
سامانه دیگر) به پایگاه داده تمیز و استاندارد پروژه اوراق گام.

چه زمانی از این اسکریپت استفاده کنیم؟
  هر بار که گزارش جدیدی (فایل .sqlite یا .xlsx) از سامانه دیگر دریافت
  می‌شود، این اسکریپت را روی آن اجرا کنید تا در قالب پایگاه داده تمیز
  پروژه (data/gam_customers.db) ذخیره شود. برنامه اصلی (streamlit_app.py)
  همیشه از همین فایل تمیز می‌خواند.

نحوه اجرا:
    python build_database.py /path/to/raw_file.sqlite
    python build_database.py /path/to/raw_file.xlsx

نکته مهم درباره فایل‌های خام:
  بعضی خروجی‌های خام (مثل نمونه اولیه این پروژه) در واقع دو جدول را
  به‌صورت افقی کنار هم چسبانده‌اند (یک جدول اصلی + یک جدول کوچک شامل
  شماره تماس). این اسکریپت به‌طور خودکار این حالت را تشخیص می‌دهد،
  دو جدول را با احتیاط ادغام می‌کند (بدون از دست دادن داده‌های متضاد)
  و در صورت تشخیص فایل ساده (یک جدول معمولی)، مستقیماً همان را
  پاکسازی و ذخیره می‌کند.
**************************************************
"""

import sys
import os
import difflib
import sqlite3
import pandas as pd

import gam_database as db

# ستون‌های استاندارد نهایی پروژه
TARGET_COLS = [
    "ردیف", "واحد درخواست کننده", "نام متقاضی", "نوع درخواست کننده",
    "نوع کالای درخواستی", "وضعیت تامین کننده", "تاریخ", "نحوه معرفی",
    "شماره تماس", "اقدامات انجام شده",
]

# فیلدهایی که تفاوت جزئی تایپی (مثل غلط املایی) در آن‌ها قابل چشم‌پوشی است
_FUZZY_COLS = {"اقدامات انجام شده"}


def _is_blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in ("", "None", "nan")


def _text_similar(a, b, threshold: float = 0.85) -> bool:
    return difflib.SequenceMatcher(None, str(a).strip(), str(b).strip()).ratio() >= threshold


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"None": None, "nan": None, "": None})
    return df


def _clean_phone_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    نرمال‌سازی ستون شماره تماس.
    اکسل گاهی شماره تلفن را عددی (float) می‌خواند و به‌صورت نماد علمی
    یا با ".0" انتهایی نمایش می‌دهد (مثلاً 9125658183.0). این تابع آن
    را به یک رشته صحیح تبدیل می‌کند.
    """
    if "شماره تماس" not in df.columns:
        return df

    def fmt(v):
        if _is_blank(v):
            return None
        if isinstance(v, float):
            return str(int(v))
        s = str(v).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s

    df["شماره تماس"] = df["شماره تماس"].apply(fmt)
    return df


def _read_raw_table(path: str) -> pd.DataFrame:
    """خواندن فایل خام (sqlite یا excel) به‌صورت یک DataFrame تک‌جدولی."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".sqlite", ".db"):
        conn = sqlite3.connect(path)
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
        if not tables:
            raise ValueError("هیچ جدولی در فایل SQLite یافت نشد.")
        # اولین جدول را می‌خوانیم (فرض بر ساختار مشابه نمونه پروژه)
        df = pd.read_sql_query(f"SELECT * FROM {tables[0]}", conn)
        conn.close()
        return df
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"فرمت فایل پشتیبانی نمی‌شود: {ext}")


def _detect_suffix(df: pd.DataFrame) -> str | None:
    """
    تشخیص الگوی پسوند دومین جدول در یک فایل خام دوتایی.
    دو الگو دیده شده:
      - "_1"  (وقتی فایل خام sqlite با ستون‌های صریحاً نام‌گذاری‌شده باشد)
      - ".1"  (وقتی pandas یک اکسل با ستون‌های هم‌نام تکراری را می‌خواند)
    اگر هیچ‌کدام یافت نشود، None برمی‌گردد (یعنی فایل تک‌جدولی است).
    """
    if "نام متقاضی_1" in df.columns:
        return "_1"
    if "نام متقاضی.1" in df.columns:
        return ".1"
    return None


def _looks_like_double_table(df: pd.DataFrame) -> bool:
    """تشخیص اینکه آیا فایل خام دو جدول افقی چسبیده به‌هم است."""
    return _detect_suffix(df) is not None


def _split_double_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    جدا کردن دو نیمه از یک فایل خام دوتایی، هر دو با ستون‌های استاندارد.

    مهم: هر ستون فقط زمانی برای یک نیمه خوانده می‌شود که واقعاً برای همان
    نیمه در فایل خام وجود داشته باشد. مثلاً «شماره تماس» گاهی فقط متعلق
    به نیمه B است (نیمه A اصلاً چنین ستونی ندارد) و گاهی هر دو نیمه
    ستون شماره تماس واقعی و مستقل خودشان را دارند. خواندن نابه‌جا از
    یک ستون که به نیمه دیگر تعلق دارد، دقیقاً همان باگی است که قبلاً
    باعث نسبت‌دادن اشتباه شماره تماس به مشتریان غلط شد — پس اینجا هرگز
    فرض ثابتی درباره وجود/نبود یک ستون نمی‌کنیم، بلکه با .get() فقط
    در صورت وجود واقعی، آن ستون را می‌خوانیم.
    """
    suffix = _detect_suffix(raw)
    if suffix is None:
        raise ValueError("ساختار دوتایی تشخیص داده نشد.")

    A = raw[raw["ردیف"].notna()].copy()
    A_out = pd.DataFrame({col: A[col] if col in A.columns else None for col in TARGET_COLS})

    b_col_names = {col: f"{col}{suffix}" for col in TARGET_COLS}
    row_marker = b_col_names["ردیف"]
    B = raw[raw[row_marker].notna()].copy()
    B_out = pd.DataFrame({
        col: (B[bcol] if bcol in B.columns else None)
        for col, bcol in b_col_names.items()
    })

    return (
        _clean_phone_column(_clean_strings(A_out)),
        _clean_phone_column(_clean_strings(B_out)),
    )


def _merge_tables(A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    """
    ادغام محتاطانه دو جدول:
      - اگر یک رکورد در B با یک رکورد در A هم‌نام باشد و در تمام فیلدهای
        مشترکِ غیرخالی یکسان (یا بسیار مشابه) باشد → همان رکورد است؛
        فیلدهای خالیِ A از B پر می‌شود (از جمله شماره تماس).
      - در غیر این صورت (تضاد واقعی در داده، مثل نوع متقاضی متفاوت) →
        به‌عنوان یک رکورد کاملاً جدید و جداگانه اضافه می‌شود تا هیچ
        داده‌ای گم نشود.
    """
    A = A.reset_index(drop=True).copy()

    def confident_match(a_row, b_row) -> bool:
        if str(a_row["نام متقاضی"]).strip() != str(b_row["نام متقاضی"]).strip():
            return False
        for col in TARGET_COLS:
            if col in ("ردیف", "شماره تماس"):
                continue
            av, bv = a_row[col], b_row[col]
            if _is_blank(av) or _is_blank(bv):
                continue
            if col in _FUZZY_COLS:
                if not _text_similar(av, bv):
                    return False
            elif str(av).strip() != str(bv).strip():
                return False
        return True

    new_rows = []
    for _, brow in B.iterrows():
        candidates = A[A["نام متقاضی"].astype(str).str.strip() == str(brow["نام متقاضی"]).strip()]
        matched_idx = None
        for idx, arow in candidates.iterrows():
            if confident_match(arow, brow):
                matched_idx = idx
                break

        if matched_idx is not None:
            for col in TARGET_COLS:
                if col == "ردیف":
                    continue
                if _is_blank(A.loc[matched_idx, col]) and not _is_blank(brow[col]):
                    A.loc[matched_idx, col] = brow[col]
        else:
            new_rows.append(brow)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        next_id = int(A["ردیف"].max()) + 1
        new_df["ردیف"] = range(next_id, next_id + len(new_df))
        A = pd.concat([A, new_df], ignore_index=True)

    return A


def build(raw_path: str) -> None:
    """نقطه ورود اصلی: خواندن فایل خام و ذخیره نسخه تمیز در پایگاه داده پروژه."""
    print(f"در حال خواندن فایل خام: {raw_path}")
    raw = _read_raw_table(raw_path)

    if _looks_like_double_table(raw):
        print("ساختار دوتایی (دو جدول چسبیده) تشخیص داده شد — در حال جداسازی و ادغام...")
        A, B = _split_double_table(raw)
        final_df = _merge_tables(A, B)
    else:
        print("ساختار تک‌جدولی معمولی تشخیص داده شد.")
        final_df = raw.copy()
        # اطمینان از وجود همه ستون‌های استاندارد
        for col in TARGET_COLS:
            if col not in final_df.columns:
                final_df[col] = None
        final_df = final_df[TARGET_COLS]
        final_df = _clean_phone_column(_clean_strings(final_df))

    final_df["ردیف"] = pd.to_numeric(final_df["ردیف"], errors="coerce").astype("Int64")

    db.import_dataframe(final_df, replace=True)
    print(f"✅ ذخیره شد: {len(final_df)} رکورد در {db.DB_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("استفاده: python build_database.py /path/to/raw_file.(sqlite|db|xlsx)")
        sys.exit(1)
    build(sys.argv[1])
