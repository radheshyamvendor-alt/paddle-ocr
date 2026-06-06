FROM python:3.10-slim

# Install system dependencies required by OpenCV and PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir paddlepaddle==3.1.1
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY . .

EXPOSE 8000
ENV PORT=8000

CMD ["python", "main.py"]
