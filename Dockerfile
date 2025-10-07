# syntax=docker/dockerfile:1.5
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md /app/
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install -e .[dev]

FROM base AS runtime
COPY . /app
RUN addgroup --system app && adduser --system --ingroup app app
USER app
ENV SERVICE=api
CMD ["uvicorn", "ufc_winprob.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
