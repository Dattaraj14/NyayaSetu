FROM python:3.11-slim

# Install Tesseract OCR (English, Hindi, Marathi) and PDF rendering libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-mar \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create UID 1000 user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TESSERACT_CMD=/usr/bin/tesseract \
    EMBEDDING_DEVICE=cpu

WORKDIR $HOME/app

# Install Python dependencies first for better Docker layer caching
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY --chown=user:user . .

# Ensure runtime directories exist and are owned by UID 1000
RUN mkdir -p uploads temp data/vector_indexes

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]