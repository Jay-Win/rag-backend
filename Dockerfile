# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System packages needed by unstructured + sqlite + libmagic
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend
COPY . .

# Expose FastAPI port
EXPOSE 9000

# Env you might want to parametrize
# ENV OLLAMA_HOST=http://h01.m5.jay-win.de:11434

# Start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "9000"]
