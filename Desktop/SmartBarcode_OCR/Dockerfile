# ใช้ Python Slim
FROM python:3.9-slim

# ตั้งค่าพื้นฐาน
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 1. ลง Driver ที่จำเป็น (ZBar สำหรับ Barcode, GL สำหรับ OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

# 2. ลง PyTorch แบบ CPU-Only (นี่คือจุดสำคัญที่ทำให้ไฟล์เล็ก!)
# เราเจาะจงโหลดจาก link ของ cpu โดยเฉพาะ
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. ลง Library อื่นๆ ตามมา
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. ก๊อปปี้โค้ด
COPY src/ ./src/

# สั่งรันโปรแกรม
CMD ["python", "src/main.py"]