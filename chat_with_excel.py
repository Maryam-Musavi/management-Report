import pandas as pd
import json
import requests

INPUT_FILE = r"input.xlsx"
MODEL_NAME = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"
API_PATH = "/api/generate"
REQUEST_TIMEOUT = 120
TEMPERATURE = 0.1

def load_and_analyze_data(file_path):
    print("[INFO] Loading data...")
    df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
    
    possible_columns = ['استان', 'شهر', 'محل', 'آدرس', 'موقعیت', 'منطقه', 'واحد درخواست کننده']
    province_col = None
    for col in possible_columns:
        if col in df.columns:
            province_col = col
            print(f"[INFO] Found province column: '{col}'")
            break
    
    if province_col is None:
        for col in df.columns:
            if df[col].dtype == 'object':
                province_col = col
                print(f"[INFO] Using column: '{col}' for province/city")
                break
    
    provinces = {}
    if province_col:
        for value in df[province_col].dropna():
            val_str = str(value).strip()
            if val_str and val_str != 'nan':
                provinces[val_str] = provinces.get(val_str, 0) + 1
    
    summary = {
        'total_records': len(df),
        'unique_provinces': len(provinces),
        'provinces_count': provinces,
        'province_column': province_col,
    }
    
    print(f"[INFO] Total records: {len(df)}")
    print(f"[INFO] Unique provinces/cities: {len(provinces)}")
    return summary, df

def ask_question(question, data_summary):
    context = f"""
اطلاعات خلاصه شده از فایل اکسل:

تعداد کل رکوردها: {data_summary['total_records']}
تعداد استان/شهرهای مختلف: {data_summary['unique_provinces']}

لیست استان/شهرها و تعداد تماس‌های هرکدام:
{json.dumps(data_summary['provinces_count'], ensure_ascii=False, indent=2)}
"""
    
    prompt = f"""شما یک دستیار هوشمند هستید که به سوالات کاربر درباره داده‌های یک فایل اکسل پاسخ می‌دهید.

اطلاعات داده‌ها:
{context}

سوال کاربر: {question}

لطفاً به سوال کاربر به زبان فارسی و با دقت پاسخ دهید. اگر سوال درباره تعداد استان‌ها یا تعداد تماس‌های هر استان است، از اطلاعات بالا استفاده کنید.

پاسخ:"""

    url = OLLAMA_URL.rstrip("/") + API_PATH
    body = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE},
    }
    
    try:
        response = requests.post(url, json=body, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "پاسخی دریافت نشد.")
        else:
            return f"خطا: {response.status_code}"
    except Exception as e:
        return f"خطا: {e}"

def main():
    print("[START] Chatbot for Excel data...")
    
    try:
        data_summary, df = load_and_analyze_data(INPUT_FILE)
    except Exception as e:
        print(f"[ERROR] Failed to load file: {e}")
        return
    
    print("\n" + "="*60)
    print(f"✅ تعداد کل استان‌ها: {data_summary['unique_provinces']}")
    print(f"✅ تعداد کل رکوردها: {data_summary['total_records']}")
    print("\n📊 آمار استان‌ها:")
    for province, count in sorted(data_summary['provinces_count'].items(), key=lambda x: x[1], reverse=True):
        print(f"   • {province}: {count} تماس")
    print("="*60)
    
    print("\n🤖 چت بات آماده است. سوال خود را بپرسید.")
    print("(برای خروج 'exit' یا 'خروج' را تایپ کنید)\n")
    
    while True:
        question = input("❓ سوال شما: ").strip()
        if question.lower() in ['exit', 'quit', 'خروج', 'bye']:
            print("👋 خدانگهدار!")
            break
        if question:
            print("🤔 در حال پردازش...")
            answer = ask_question(question, data_summary)
            print(f"💬 پاسخ: {answer}\n")

if __name__ == "__main__":
    main()
