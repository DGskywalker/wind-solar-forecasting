FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ultra-fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy pyproject.toml first for layer caching
COPY pyproject.toml README.md ./

# Install project dependencies
RUN uv pip install --system -e ".[dev]"

# Copy source code and configuration
COPY . .

EXPOSE 8000 8501 5000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
