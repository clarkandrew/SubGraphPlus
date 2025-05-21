# Use official Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsqlite3-dev \
    wget \
    curl \
    gnupg \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create required directories
RUN mkdir -p data/faiss cache/dde logs models config

# Copy project files
COPY . .

# Initialize SQLite database
RUN python -c "import sqlite3, os; \
    conn = sqlite3.connect('data/staging.db'); \
    conn.execute('CREATE TABLE IF NOT EXISTS staging_triples (head TEXT, relation TEXT, tail TEXT, head_name TEXT, relation_name TEXT, tail_name TEXT, status TEXT DEFAULT \"pending\", timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)'); \
    conn.execute('CREATE TABLE IF NOT EXISTS error_log (triple_id INTEGER, error TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)'); \
    conn.execute('CREATE TABLE IF NOT EXISTS auth_log (ip_hash TEXT, attempt_count INTEGER DEFAULT 1, last_attempt DATETIME DEFAULT CURRENT_TIMESTAMP)'); \
    conn.execute('CREATE INDEX IF NOT EXISTS idx_staging_status ON staging_triples(status)'); \
    conn.commit(); conn.close()"

# Make scripts executable
RUN chmod +x scripts/*.py *.sh || true

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Download models if they do not exist\n\
if [ ! -d "models/mlp" ]; then\n\
    echo "Downloading MLP models..."\n\
    python scripts/download_models.py || echo "Warning: Failed to download models"\n\
fi\n\
\n\
# Wait for Neo4j to be available\n\
echo "Waiting for Neo4j..."\n\
MAX_RETRIES=30\n\
RETRIES=0\n\
until wget -q -O - http://subgraphrag_neo4j:7474 > /dev/null 2>&1 || [ $RETRIES -eq $MAX_RETRIES ]; do\n\
    echo "Waiting for Neo4j to be ready... ($((++RETRIES))/$MAX_RETRIES)"\n\
    sleep 2\n\
done\n\
\n\
if [ $RETRIES -eq $MAX_RETRIES ]; then\n\
    echo "Warning: Neo4j may not be available. Continuing anyway..."\n\
else\n\
    echo "Neo4j is ready."\n\
    # Run schema migrations\n\
    python scripts/migrate_schema.py || echo "Warning: Schema migration failed"\n\
fi\n\
\n\
# Start the application\n\
exec "$@"\n\
' > /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
