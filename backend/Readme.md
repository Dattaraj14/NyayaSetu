# NyayaSetu Backend

The backend is a FastAPI application that accepts legal PDF uploads, extracts and organizes their text, builds a semantic search index, and answers questions using retrieved passages from the selected document.

## Stack

- FastAPI and Uvicorn for the HTTP API.
- Pydantic Settings for environment-based configuration.
- SQLAlchemy async sessions for database access.
- PyMuPDF for PDF inspection, text extraction, and page rendering.
- Tesseract through `pytesseract` for OCR.
- BGE-M3 through Sentence Transformers for embeddings.
- FAISS for per-document vector search.
- Gemini through the Google Gen AI SDK for answer generation.

Dependencies are pinned in `requirements.txt`.

## Run locally

Run the following from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` and the PostgreSQL connection in `.env`, then start the API:

```powershell
uvicorn app.main:app --app-dir backend --reload
```

The API documentation is available at `http://localhost:8000/docs`.

The backend creates database tables at startup with SQLAlchemy `create_all`. The checked-in migration versions directory currently has no migration scripts.

## Environment configuration

The main settings are defined in `app/core/config.py`. `.env.example` documents:

- `DATABASE_URL`: async PostgreSQL connection string.
- `GEMINI_API_KEY` and `LLM_MODEL`: Gemini credentials and model selection.
- `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`, and `EMBEDDING_BATCH_SIZE`.
- `VECTOR_DB_PATH`: base directory for FAISS indexes.
- `MAX_FILE_SIZE_MB` and `MAX_PAGES`.
- `TESSERACT_CMD` and OCR confidence settings.
- `ALLOWED_ORIGINS` for browser requests.

The example file also has `DATABASE_SYNC_URL`, but the current backend database connection uses `DATABASE_URL`.

The settings include options for local, OpenAI, and Groq providers, as well as PaddleOCR, but the implemented chat service is Gemini-based and the OCR pipeline is Tesseract-based. Scanned-document support requires a working Tesseract installation and the `eng`, `hin`, and `mar` language packs.

The code defaults the embedding device to CUDA and falls back to CPU if CUDA is unavailable. BGE-M3 may need to be downloaded on first use.

## Request flow

### PDF upload and processing

`POST /api/documents/upload` validates the `.pdf` extension and file size, generates a safe unique filename, saves the uploaded file, creates a database record, and schedules background processing.

The processor:

1. Detects whether the PDF is text based, scanned, hybrid, or unknown.
2. Extracts text from text PDFs with PyMuPDF, or renders pages and runs OCR for scanned and hybrid PDFs.
3. Preprocesses page text.
4. Detects numbered paragraphs and legal sections.
5. Creates chunks that preserve page, section, paragraph, and OCR artifact metadata.
6. Saves pages, sections, and chunks in the database.

The chunker targets about 700 estimated tokens per chunk and caps chunks at about 1,000 estimated tokens. These token counts are approximations based on word counts.

### Semantic indexing

`POST /api/documents/{document_id}/index` embeds the document's chunks with BGE-M3 and saves a FAISS index and metadata under the configured vector index directory. Indexes are stored separately per document.

### Document question answering

`POST /api/chat/{document_id}` retrieves up to five relevant chunks, builds a prompt instructing Gemini to answer only from those passages, and returns the answer and source references. The request accepts a message, an optional session UUID, a response language (`en`, `hi`, or `mr`), and a mode.

`general_legal` mode currently returns HTTP 501. Use `document_qa` for the implemented flow.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Basic application status |
| `GET` | `/api` | Health check |
| `POST` | `/api/documents/upload` | Upload a PDF |
| `GET` | `/api/documents/` | List documents |
| `GET` | `/api/documents/{document_id}` | Get document details, pages, sections, and entities |
| `POST` | `/api/documents/{document_id}/index` | Build the FAISS index |
| `POST` | `/api/chat/{document_id}` | Ask a question about an indexed document |

Uploaded files must be PDF files and fit within the configured maximum size. The upload route checks the extension and size; a PDF magic-byte validation helper exists but is not currently called by the route.

## Database and local files

SQLAlchemy models cover uploaded documents, pages, sections, chunks, entities, summaries, translations, legal knowledge, chat sessions, and processing logs. The current upload pipeline saves document pages, sections, and chunks. Other model types do not imply that their associated feature has a working API.

The backend writes uploaded PDFs and FAISS indexes to local disk. Preserve the database and these directories if you need to retain uploaded documents and indexes.

## Current implementation limits

- Chat uses Gemini, despite configuration settings that list additional providers.
- OCR uses Tesseract, despite settings that mention PaddleOCR.
- General legal mode, summaries, simplification, and translation are not currently implemented as API routes.
- The health endpoint reports configured service flags; it does not verify every external service or model.
- The frontend and backend currently use different field names for some progress and source data.
- The default SQLite URL requires `aiosqlite`, which is not listed as a backend dependency. The provided environment example uses PostgreSQL.
