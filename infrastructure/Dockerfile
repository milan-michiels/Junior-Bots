FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential  \
    curl \
    poppler-utils \
    ffmpeg \
    tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /chatbot-service

COPY requirements.txt /chatbot-service/requirements.txt

RUN apt-get install -y poppler-utils tesseract-ocr

RUN pip install --upgrade pip && pip install --no-cache-dir --upgrade -r /chatbot-service/requirements.txt

COPY app /chatbot-service/app

COPY data /chatbot-service/data

EXPOSE 8000

CMD ["fastapi", "run", "/chatbot-service/app/main.py", "--port", "8000"]