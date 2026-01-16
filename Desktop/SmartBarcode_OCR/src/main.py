import os
import io
import cv2
import numpy as np
import pandas as pd
import requests
import easyocr
import torch
from pyzbar.pyzbar import decode
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURATION ---
INPUT_FOLDER = os.getenv('DATA_DIR', './data')
INPUT_FILE_NAME = 'input.csv'
OUTPUT_FILE_NAME = 'result.csv'
COL_URL = 'Link'

# --- 1. SETUP SYSTEM & GPU ---
def get_device():
    """ตรวจสอบและตั้งค่า GPU"""
    use_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"⚙️ System Initializing... Processor: {device_name}")
    return use_cuda

USE_GPU = get_device()
reader = easyocr.Reader(['en'], gpu=USE_GPU)

# Setup Session with Retry
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# --- 2. CORE LOGIC ---
def preprocess_image(img_cv2):
    """ปรับแต่งภาพให้อ่านง่ายขึ้น"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        img_gray = ImageOps.grayscale(img_pil)
        enhancer = ImageEnhance.Contrast(img_gray)
        img_enhanced = enhancer.enhance(2.0)
        return np.array(img_enhanced)
    
    # คำสั่ง except ต้องตรงกับ try เป๊ะๆ
    except Exception as e:
        print(f"Error occurred: {e}")  
        return img_cv2                 
def analyze_image(img_np):
    """
    ฟังก์ชันตัดสินใจว่าจะ Scan หรือ OCR
    Returns: (text_result, method_name)
    """
    # 1. Barcode Scan
    barcodes = decode(img_np)
    if barcodes:
        results = [b.data.decode('utf-8') for b in barcodes]
        return ", ".join(results), "scan"  # <--- แก้เป็นคำว่า scan

    # 2. AI-OCR
    ocr_results = reader.readtext(img_np, detail=0, allowlist='0123456789')
    valid_numbers = [num for num in ocr_results if len(num) > 5]

    if valid_numbers:
        return ", ".join(valid_numbers), "ocr" # <--- แก้เป็นคำว่า ocr
    
    return "Not Found", "-"

def download_image(url):
    """ดาวน์โหลดรูปภาพ"""
    try:
        if not isinstance(url, str) or not url.lower().startswith(('http://', 'https://')):
            return None

        response = session.get(url, timeout=10)
        response.raise_for_status()
        
        image_bytes = io.BytesIO(response.content)
        file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# --- 3. MAIN PROCESS ---
def load_data(file_path):
    """โหลดข้อมูลจาก CSV/Excel"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    try:
        if file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
            except pd.errors.ParserError:
                df = pd.read_csv(file_path, header=None)
                cols = [COL_URL] + [f"Col_{i}" for i in range(1, len(df.columns))]
                df.columns = cols[:len(df.columns)]
        else:
            df = pd.read_excel(file_path)
            
        if COL_URL not in df.columns:
            print(f"⚠️ Column '{COL_URL}' not found. Using first column.")
            df.rename(columns={df.columns[0]: COL_URL}, inplace=True)
            
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def main():
    input_path = os.path.join(INPUT_FOLDER, INPUT_FILE_NAME)
    output_path = os.path.join(INPUT_FOLDER, OUTPUT_FILE_NAME)

    print(f"📂 Reading from: {input_path}")
    df = load_data(input_path)
    
    if df is None or df.empty:
        return

    print(f"✅ Loaded {len(df)} rows. Starting process on {'GPU' if USE_GPU else 'CPU'}...")

    results_data = []
    types_data = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        url = str(row[COL_URL]).strip()
        
        img = download_image(url)
        
        if img is not None:
            enhanced_img = preprocess_image(img)
            text, method = analyze_image(enhanced_img)
            results_data.append(text)
            types_data.append(method)
        else:
            results_data.append("Error")
            types_data.append("-")

    # เพิ่มคอลัมน์ผลลัพธ์
    df['Detected_Serial'] = results_data
    df['Method'] = types_data # <--- คอลัมน์นี้จะเก็บค่า "scan" หรือ "ocr"

    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"🎉 Success! Results saved to: {output_path}")
    except PermissionError:
        print(f"❌ Error: Could not save file. Please close {OUTPUT_FILE_NAME}.")

if __name__ == "__main__":
    main()