FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /opt/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Copy poetry files
COPY pyproject.toml ./

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && export PATH="/root/.local/bin:$PATH" \
    && poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project true \
    && poetry install --no-interaction

# Add poetry environment to PATH
ENV PATH="/opt/app/.venv/bin:$PATH"
ENV PYTHONPATH="/opt/app:$PYTHONPATH"

# Copy the rest of the application
COPY . .

CMD ["python3"] 