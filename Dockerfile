# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY session_title_generator.py ./
COPY server.py ./

ENV PORT=10000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]
