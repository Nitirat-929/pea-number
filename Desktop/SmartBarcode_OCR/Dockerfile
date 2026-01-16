# ใช้ Base Image ที่มี CUDA (Driver การ์ดจอ) ในตัว
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ตั้งค่าพื้นฐาน Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ติดตั้ง Library พื้นฐานสำหรับ OpenCV (แก้ปัญหา cv2)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy ไฟล์ settings และติดตั้ง Library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy โค้ดทั้งหมดเข้ากล่อง
COPY . .

# สั่งรันโปรแกรม (ชี้ไปที่โฟลเดอร์ src)
CMD ["python", "src/main.py"]