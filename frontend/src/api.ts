/** Strictly typed API client and SSE stream reader for NyayaSetu backend. */

export const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ||
  'http://127.0.0.1:8000/api'

export type DocumentStatus =
  | 'uploaded'
  | 'validating'
  | 'extracting'
  | 'ocr_processing'
  | 'preprocessing'
  | 'analyzing'
  | 'embedding'
  | 'summarizing'
  | 'completed'
  | 'failed'

export type LanguageCode = 'en' | 'hi' | 'mr'
export type OcrLanguageCode = 'eng' | 'eng+hin' | 'eng+mar' | 'eng+hin+mar'
export type SummaryLength = 'short' | 'medium' | 'detailed'

export interface DocumentListItem {
  id: string
  filename: string
  status: DocumentStatus
  progress: number
  message: string | null
  created_at: string
}

export interface DocumentDetail {
  id: string
  filename: string
  original_filename: string
  file_size: number
  page_count: number
  pdf_type: string | null
  status: DocumentStatus
  status_message: string | null
  processing_progress: number
  court: string | null
  case_name: string | null
  case_number: string | null
  document_type: string
  language: string
  created_at: string
  updated_at: string
}

export interface UploadResponse {
  id: string
  filename: string
  status: DocumentStatus
  message: string
}

export interface IndexResponse {
  document_id: string
  chunks_indexed: number
  status: string
  message: string
}

export interface SourceReference {
  document_name: string | null
  page: number | null
  section: string | null
  paragraph: string | null
  content: string | null
  knowledge_source: string | null
}

export interface VerifiedClaim {
  text: string
  source_ids: string[]
  chunk_ids: string[]
  pages: number[]
  supported: boolean
}

export interface ChatFinalPayload {
  type: 'final'
  session_id: string
  message: string
  sources: SourceReference[]
  was_refused: boolean
  refusal_reason: string | null
  mode: string
  intent: string | null
  verification_status: string | null
  claims: VerifiedClaim[]
  trace: string[]
}

export interface ChatStreamProgressEvent {
  type: 'progress'
  node: string
  status: string
}

export interface ChatStreamTokenEvent {
  type: 'token'
  text: string
}

export interface ChatStreamErrorEvent {
  type: 'error'
  detail: string
}

export type ChatStreamEvent =
  | ChatStreamProgressEvent
  | ChatStreamTokenEvent
  | ChatFinalPayload
  | ChatStreamErrorEvent

export interface UnusualClause {
  clause_reference: string
  risk_explanation: string
}

export interface StructuredSummaryData {
  // Judgment fields
  court?: string
  date?: string
  case_number?: string
  parties?: string[]
  facts?: string[]
  issues?: string[]
  petitioner_arguments?: string[]
  respondent_arguments?: string[]
  reasoning?: string[]
  holding?: string
  final_order?: string
  statutes_cited?: string[]
  precedents_cited?: string[]
  // Contract fields
  effective_date?: string
  term?: string
  obligations?: string[]
  payment_terms?: string
  termination?: string
  liability_and_indemnity?: string
  governing_law_and_jurisdiction?: string
  unusual_or_one_sided_clauses?: UnusualClause[]
  // Notice / general fields
  title?: string
  key_points?: string[]
  deadlines_or_actions?: string[]
  statutes_or_clauses_cited?: string[]
}

export interface SummaryClaim {
  text: string
  pages: number[]
}

export interface SummaryRecord {
  id: string
  document_id: string
  summary_type: SummaryLength
  language: LanguageCode
  document_type: string
  content: string
  structured_summary: StructuredSummaryData
  claims: SummaryClaim[]
  verification_status: string
  trace: string[]
  model_used: string | null
  created_at: string
}

async function parseError(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as {
      detail?: string | Array<{ msg?: string }>
    }
    if (typeof data.detail === 'string') return data.detail
    if (Array.isArray(data.detail)) {
      return data.detail.map((d) => d.msg || String(d)).join(', ')
    }
  } catch {
    // Fall back to status text
  }
  return `Request failed (${response.status} ${response.statusText})`
}

export async function listDocuments(): Promise<DocumentListItem[]> {
  const response = await fetch(`${API_BASE}/documents/`)
  if (!response.ok) throw new Error(await parseError(response))
  return (await response.json()) as DocumentListItem[]
}

export async function getDocument(documentId: string): Promise<DocumentDetail> {
  const response = await fetch(`${API_BASE}/documents/${documentId}`)
  if (!response.ok) throw new Error(await parseError(response))
  return (await response.json()) as DocumentDetail
}

export async function uploadDocument(
  file: File,
  ocrLanguage: OcrLanguageCode,
): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('ocr_language', ocrLanguage)

  const response = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) throw new Error(await parseError(response))
  return (await response.json()) as UploadResponse
}

export async function indexDocument(documentId: string): Promise<IndexResponse> {
  const response = await fetch(`${API_BASE}/documents/${documentId}/index`, {
    method: 'POST',
  })
  if (!response.ok) throw new Error(await parseError(response))
  return (await response.json()) as IndexResponse
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/documents/${documentId}`, {
    method: 'DELETE',
  })
  if (!response.ok) throw new Error(await parseError(response))
}

export async function streamDocumentChat(
  documentId: string,
  payload: {
    message: string
    session_id: string
    language: LanguageCode
  },
  onEvent: (event: ChatStreamEvent) => void,
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat/${documentId}/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: payload.message,
      session_id: payload.session_id,
      language: payload.language,
      mode: 'document_qa',
    }),
  })

  if (!response.ok) {
    throw new Error(await parseError(response))
  }

  if (!response.body) {
    throw new Error('Streaming response body is unavailable.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() || ''

    for (const part of parts) {
      const lines = part
        .split('\n')
        .map((l) => l.trim())
        .filter((l) => l.startsWith('data:'))

      for (const line of lines) {
        const jsonStr = line.slice(5).trim()
        if (!jsonStr) continue
        try {
          const parsed = JSON.parse(jsonStr) as ChatStreamEvent
          onEvent(parsed)
        } catch {
          // Ignore malformed keepalive chunks
        }
      }
    }
  }
}

export async function listSummaries(
  documentId: string,
): Promise<SummaryRecord[]> {
  const response = await fetch(`${API_BASE}/documents/${documentId}/summaries`)
  if (!response.ok) throw new Error(await parseError(response))
  return (await response.json()) as SummaryRecord[]
}

export async function generateSummary(
  documentId: string,
  options: {
    summary_type: SummaryLength
    language: LanguageCode
    force_refresh: boolean
  },
): Promise<SummaryRecord> {
  const response = await fetch(`${API_BASE}/documents/${documentId}/summarize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options),
  })
  if (!response.ok) throw new Error(await parseError(response))
  return (await response.json()) as SummaryRecord
}

export function getSummaryExportUrl(
  documentId: string,
  summaryType: SummaryLength,
  language: LanguageCode,
  exportFormat: 'md' | 'json',
): string {
  const params = new URLSearchParams({
    summary_type: summaryType,
    language,
    export_format: exportFormat,
  })
  return `${API_BASE}/documents/${documentId}/summaries/export?${params.toString()}`
}