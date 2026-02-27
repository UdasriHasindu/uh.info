ARG PYTHON_VERSION=3.13.7
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directory for ChromaDB and set ownership
RUN mkdir -p chroma_langchain_db && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# IMPORTANT: Use Railway's PORT variable
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]