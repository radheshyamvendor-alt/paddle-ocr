FROM python:3.10-slim

# Install system dependencies required by OpenCV and PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-based paddlepaddle first to ensure correctness on CPU-only container host
RUN pip install --no-cache-dir paddlepaddle==2.6.2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY . .

# Warm up / pre-download OCR weight models at build time to avoid cold start issues
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"

EXPOSE 8000
ENV PORT=8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
