import pandas as pd
import json
import requests
import re

# =========================
# Config (Editable)
# =========================
INPUT_FILE = r"input.xlsx"  # همان فایل اصلی شما
MODEL_NAME = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"
API_PATH = "/api/generate"
REQUEST_TIMEOUT = 120
TEMPERATURE = 0.1

# =========================
# Load and Analyze Data
# =========================
def load_and_analyze_data(file_path):
    """Load Excel file and extract key information"""
    print("[INFO] Loading data...")
    df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
    
    # Find columns that might contain province/city names
    # Try common Persian column names
    possible_columns = ['استان', 'شهر', 'محل', 'آدرس', 'موقعیت', 'منطقه', 
                        'واحد درخواست کننده', 'نام متقاضی']
    
    province_col = None
    for col in possible_columns:
        if col in df.columns:
            province_col = col
            print(f"[INFO] Found province/location column: '{col}'")
            break
    
    if province_col is None:
        # If no Persian column found, show all columns
        print(f"[INFO] Available columns: {df.columns.tolist()}")
        print("[WARNING] Could not find province column. Using first text column as reference.")
        # Find first string column
        for col in df.columns:
            if df[col].dtype == 'object':
                province_col = col
                break
    
    # Extract province/city information
    provinces = {}
    if province_col:
        for value in df[province_col].dropna():
            val_str = str(value).strip()
            if val_str and val_str != 'nan':
                # Simple counting
                provinces[val_str] = provinces.get(val_str, 0) + 1
    
    # Get total rows
    total_rows = len(df)
    
    # Create summary
    summary = {
        'total_records': total_rows,
        'unique_provinces': len(provinces),
        'provinces_count': provinces,
        'province_column': province_col,
        'all_columns': df.columns.tolist(),
        'sample_data': df.head(5).to_dict()
    }
    
    print(f"[INFO] Total records: {total_rows}")
    print(f"[INFO] Unique provinces/cities: {len(provinces)}")
    
    return summary, df

# =========================
# Query System
# =========================
def ask_question(question, data_summary):
    """Ask a question about the data using Ollama"""
    
    # Create a context from the data summary
    context = f"""
    اطلاعات خلاصه شده از فایل اکسل:
    
    تعداد کل رکوردها: {data_summary['total_records']}
    تعداد استان/شهرهای مختلف: {data_summary['unique_provinces']}
    
    لیست استان/شهرها و تعداد تماس‌های هرکدام:
    {json.dumps(data_summary['provinces_count'], ensure_ascii=False, indent=2)}
    
    نام ستونی که اطلاعات استان/شهر در آن قرار دارد: {data_summary['province_column']}
    
    همه ستون‌های موجود در فایل:
    {data_summary['all_columns']}
    """
    
    prompt = f"""شما یک دستیار هوشمند هستید که به سوالات کاربر درباره داده‌های یک فایل اکسل پاسخ می‌دهید.

اطلاعات داده‌ها:
{context}

سوال کاربر: {question}

لطفاً به سوال کاربر به زبان فارسی و با دقت پاسخ دهید. اگر سوال درباره تعداد استان‌ها یا تعداد تماس‌های هر استان است، از اطلاعات بالا استفاده کنید.
اگر سوال نیاز به اطلاعات بیشتری دارد که در خلاصه موجود نیست، بگویید که آن اطلاعات در دسترس نیست.

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
            answer = data.get("response", "پاسخی دریافت نشد.")
            return answer.strip()
        else:
            return f"خطا در ارتباط با مدل: {response.status_code}"
    except Exception as e:
        return f"خطا: {e}"

def answer_specific_questions(data_summary):
    """Answer common questions about provinces"""
    
    total_provinces = data_summary['unique_provinces']
    province_counts = data_summary['provinces_count']
    
    # Sort provinces by count
    sorted_provinces = sorted(province_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print("📊 آمار استان‌ها:")
    print("="*60)
    print(f"✅ تعداد کل استان‌های موجود در داده‌ها: {total_provinces} استان")
    print(f"✅ تعداد کل رکوردها: {data_summary['total_records']} تماس")
    print("\n📌 تعداد تماس‌ها بر اساس استان (به ترتیب بیشترین):")
    
    for province, count in sorted_provinces:
        print(f"   • {province}: {count} تماس")
    
    print("="*60)

def chat_mode(data_summary):
    """Interactive chat mode"""
    print("\n" + "="*60)
    print("🤖 چت بات هوشمند برای سوالات از داده‌های اکسل")
    print("="*60)
    print("💡 سوالات نمونه:")
    print("   - در کل چند استان تماس گرفته‌اند؟")
    print("   - از استان تهران چند نفر تماس گرفته‌اند؟")
    print("   - کدام استان بیشترین تماس را داشته؟")
    print("   - چند درصد تماس‌ها از استان اصفهان است؟")
    print("   - لیست همه استان‌ها را به من بده")
    print("\n⚠️  برای خروج از چت بات، 'exit' یا 'quit' یا 'خروج' را تایپ کنید.")
    print("="*60)
    
    while True:
        user_input = input("\n❓ سوال شما: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'خروج', 'bye']:
            print("👋 خدانگهدار! موفق باشید.")
            break
        
        if not user_input:
            print("❗ لطفاً یک سوال بپرسید.")
            continue
        
        print("\n🤔 در حال پردازش سوال شما...")
        answer = ask_question(user_input, data_summary)
        print(f"\n💬 پاسخ: {answer}")

def main():
    print("[START] Loading Excel data for Q&A system...")
    
    # Load and analyze data
    try:
        data_summary, df = load_and_analyze_data(INPUT_FILE)
    except Exception as e:
        print(f"[ERROR] Failed to load file: {e}")
        return
    
    # First, show basic statistics
    answer_specific_questions(data_summary)
    
    # Start interactive chat
    chat_mode(data_summary)

if __name__ == "__main__":
    main()