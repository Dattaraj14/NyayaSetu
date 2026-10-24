# NyayaSetu Frontend

The NyayaSetu frontend is a React and TypeScript single-page application built with Vite. It provides an interface for uploading a legal PDF, following its processing status, choosing a response language, and asking questions about the document.

## Features

- PDF selection and a 50 MB client-side size check.
- Upload and processing status display.
- Automatic request to build a semantic search index after document processing.
- Document question and answer chat.
- English, Hindi, and Marathi response-language selection.
- Markdown rendering for assistant answers.

The frontend supports document-grounded chat. The backend's general legal mode is not implemented.

## Requirements

- Node.js and npm.
- The NyayaSetu backend running at `http://127.0.0.1:8000`.

## Run locally

From the frontend directory:

```powershell
npm install
npm run dev
```

Vite prints the local development URL, usually `http://localhost:5173`.

## Available scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start the Vite development server |
| `npm run build` | Type-check and build the production frontend |
| `npm run lint` | Run Oxlint |
| `npm run preview` | Preview the production build |

## Backend connection

The API base URL is currently set directly in `src/App.tsx`:

```ts
const API_BASE = 'http://127.0.0.1:8000/api'
```

The backend must allow the frontend origin through its `ALLOWED_ORIGINS` configuration.

## Current integration notes

- The frontend reads processing progress from `progress`, while the backend document response uses `processing_progress`.
- The frontend source display expects fields such as `page_number` and `section_type`; the chat response currently returns `page` and `section`. Source entries may therefore show without page or section details.
- The chat response uses a `message` field. The frontend supports that field, with `answer` as a fallback.
