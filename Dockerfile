FROM python:3.11-slim

# Install system dependencies required by spaCy and other NLP libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model
RUN python -m spacy download en_core_web_sm

# Download NLTK VADER lexicon
RUN python -c "import nltk; nltk.download('vader_lexicon')"

# Copy application code and assets
COPY api.py .
COPY models/ models/
COPY data/ data/

# Expose the port Render / other platforms expect
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
