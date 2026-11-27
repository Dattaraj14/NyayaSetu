import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  AlertCircle,
  BookOpen,
  CheckCircle2,
  Database,
  Download,
  FileText,
  Loader2,
  RefreshCw,
  Scale,
  Send,
  ShieldAlert,
  ShieldCheck,
  Trash2,
  Upload,
  X,
} from 'lucide-react'
import {
  deleteDocument,
  generateSummary,
  getDocument,
  getSummaryExportUrl,
  indexDocument,
  listDocuments,
  listSummaries,
  streamDocumentChat,
  uploadDocument,
} from './api'
import type {
  DocumentDetail,
  DocumentListItem,
  LanguageCode,
  OcrLanguageCode,
  SourceReference,
  SummaryLength,
  SummaryRecord,
  VerifiedClaim,
} from './api'
import './App.css'

type WorkspaceTab = 'qa' | 'brief'

interface ChatTurn {
  id: string
  role: 'user' | 'assistant'
  content: string
  intent?: string | null
  verificationStatus?: string | null
  wasRefused?: boolean
  sources?: SourceReference[]
  claims?: VerifiedClaim[]
  trace?: string[]
  isStreaming?: boolean
}

const QA_PIPELINE_NODES = [
  { key: 'router', label: 'routing' },
  { key: 'analyze_query', label: 'analyzing' },
  { key: 'retrieve', label: 'retrieving' },
  { key: 'rerank', label: 'reranking' },
  { key: 'grade_evidence', label: 'grading' },
  { key: 'generate', label: 'generating' },
  { key: 'verify', label: 'verifying' },
  { key: 'finalize', label: 'finalizing' },
] as const

const SUGGESTED_QUESTIONS = [
  'Who are the parties and what is the core dispute?',
  'What statutes or sections are cited in this document?',
  'What was the final order or operative decision?',
]

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const qaEndRef = useRef<HTMLDivElement>(null)

  const [activeTab, setActiveTab] = useState<WorkspaceTab>('qa')
  const [language, setLanguage] = useState<LanguageCode>('en')
  const [backendOnline, setBackendOnline] = useState(true)

  // Dossiers state
  const [documents, setDocuments] = useState<DocumentListItem[]>([])
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null)
  const [selectedDoc, setSelectedDoc] = useState<DocumentDetail | null>(null)

  // Upload & processing state
  const [stagedFile, setStagedFile] = useState<File | null>(null)
  const [ocrLanguage, setOcrLanguage] = useState<OcrLanguageCode>('eng')
  const [uploading, setUploading] = useState(false)
  const [indexing, setIndexing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [statusNote, setStatusNote] = useState('')
  const [globalError, setGlobalError] = useState('')
  const [needsIndexHint, setNeedsIndexHint] = useState(false)

  // Q&A + SSE LangGraph state
  const [sessionId, setSessionId] = useState<string>(() => crypto.randomUUID())
  const [question, setQuestion] = useState('')
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [chatLoading, setChatLoading] = useState(false)
  const [activeNode, setActiveNode] = useState<string | null>(null)
  const [completedNodes, setCompletedNodes] = useState<string[]>([])

  // Evidence Inspector state
  const [inspectedSources, setInspectedSources] = useState<SourceReference[]>([])
  const [inspectedClaims, setInspectedClaims] = useState<VerifiedClaim[]>([])
  const [activeSourceIndex, setActiveSourceIndex] = useState(0)

  // Structured Brief (LangGraph Summary) state
  const [summaryType, setSummaryType] = useState<SummaryLength>('medium')
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [activeSummary, setActiveSummary] = useState<SummaryRecord | null>(null)

  const refreshDocuments = useCallback(async (selectId?: string) => {
    try {
      const docs = await listDocuments()
      setBackendOnline(true)
      setDocuments(docs)
      if (selectId) {
        setSelectedDocId(selectId)
      } else {
        setSelectedDocId((prev) => {
          if (prev && docs.some((d) => d.id === prev)) return prev
          return docs[0]?.id ?? null
        })
      }
    } catch {
      setBackendOnline(false)
    }
  }, [])

  useEffect(() => {
    void refreshDocuments()
  }, [refreshDocuments])

  // Load selected document detail & existing summaries whenever selection changes
  useEffect(() => {
    if (!selectedDocId) {
      setSelectedDoc(null)
      setActiveSummary(null)
      return
    }

    let cancelled = false
    const loadDetail = async () => {
      try {
        const [detail, existingSummaries] = await Promise.all([
          getDocument(selectedDocId),
          listSummaries(selectedDocId).catch(() => [] as SummaryRecord[]),
        ])
        if (cancelled) return
        setSelectedDoc(detail)
        setProgress(Math.round(detail.processing_progress || 0))
        setStatusNote(detail.status_message || '')
        const matched =
          existingSummaries.find(
            (s) => s.summary_type === summaryType && s.language === language,
          ) || existingSummaries[0] || null
        setActiveSummary(matched)
      } catch (err) {
        if (!cancelled && err instanceof Error) {
          setGlobalError(err.message)
        }
      }
    }

    void loadDetail()
    return () => {
      cancelled = true
    }
  }, [selectedDocId, summaryType, language])

  useEffect(() => {
    qaEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns, chatLoading])

  const handleSelectDossier = (docId: string) => {
    if (docId === selectedDocId) return
    setSelectedDocId(docId)
    setTurns([])
    setInspectedSources([])
    setInspectedClaims([])
    setActiveSourceIndex(0)
    setActiveNode(null)
    setCompletedNodes([])
    setSessionId(crypto.randomUUID())
    setGlobalError('')
    setNeedsIndexHint(false)
  }

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    setGlobalError('')
    if (
      file.type !== 'application/pdf' &&
      !file.name.toLowerCase().endsWith('.pdf')
    ) {
      setGlobalError('Please select a PDF document.')
      return
    }
    if (file.size > 50 * 1024 * 1024) {
      setGlobalError('PDF file size must be 50 MB or less.')
      return
    }
    setStagedFile(file)
  }

  const handleBuildIndex = async (docId: string) => {
    setIndexing(true)
    setNeedsIndexHint(false)
    setGlobalError('')
    setProgress(95)
    setStatusNote('Building semantic + BM25 search index...')
    try {
      const res = await indexDocument(docId)
      setProgress(100)
      setStatusNote(res.message)
      const detail = await getDocument(docId)
      setSelectedDoc(detail)
      await refreshDocuments(docId)
    } catch (err) {
      setGlobalError(
        err instanceof Error ? err.message : 'Failed to build search index.',
      )
    } finally {
      setIndexing(false)
    }
  }

  const pollUntilComplete = async (docId: string) => {
    for (let attempt = 0; attempt < 180; attempt += 1) {
      const detail = await getDocument(docId)
      setSelectedDoc(detail)
      setProgress(Math.round(detail.processing_progress || 0))
      setStatusNote(detail.status_message || 'Processing document...')

      if (detail.status === 'completed') {
        await handleBuildIndex(docId)
        return
      }
      if (detail.status === 'failed') {
        setGlobalError(detail.status_message || 'Document processing failed.')
        await refreshDocuments(docId)
        return
      }

      await new Promise((resolve) => setTimeout(resolve, 1800))
    }
    setGlobalError('Document processing timed out.')
  }

  const handleUpload = async () => {
    if (!stagedFile) return
    setUploading(true)
    setGlobalError('')
    setNeedsIndexHint(false)
    setProgress(5)
    setStatusNote('Uploading PDF...')

    try {
      const uploaded = await uploadDocument(stagedFile, ocrLanguage)
      setStagedFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ''

      setTurns([])
      setInspectedSources([])
      setInspectedClaims([])
      setSessionId(crypto.randomUUID())
      await refreshDocuments(uploaded.id)
      await pollUntilComplete(uploaded.id)
    } catch (err) {
      setGlobalError(
        err instanceof Error ? err.message : 'Failed to upload document.',
      )
    } finally {
      setUploading(false)
    }
  }

  const handleDeleteDossier = async (
    docId: string,
    event: React.MouseEvent,
  ) => {
    event.stopPropagation()
    setGlobalError('')
    try {
      await deleteDocument(docId)
      if (selectedDocId === docId) {
        setSelectedDocId(null)
        setSelectedDoc(null)
        setTurns([])
        setInspectedSources([])
        setInspectedClaims([])
        setActiveSummary(null)
      }
      await refreshDocuments()
    } catch (err) {
      setGlobalError(
        err instanceof Error ? err.message : 'Failed to delete document.',
      )
    }
  }

  const handleAskQuestion = async (event: FormEvent) => {
    event.preventDefault()
    const trimmed = question.trim()
    if (!trimmed || !selectedDocId || chatLoading) return
    if (selectedDoc?.status !== 'completed') {
      setGlobalError('Wait until document processing completes before asking.')
      return
    }

    setQuestion('')
    setGlobalError('')
    setNeedsIndexHint(false)
    setChatLoading(true)
    setActiveNode('router')
    setCompletedNodes([])

    const userTurnId = crypto.randomUUID()
    const assistantTurnId = crypto.randomUUID()

    setTurns((prev) => [
      ...prev,
      { id: userTurnId, role: 'user', content: trimmed },
      { id: assistantTurnId, role: 'assistant', content: '', isStreaming: true },
    ])

    try {
      await streamDocumentChat(
        selectedDocId,
        {
          message: trimmed,
          session_id: sessionId,
          language,
        },
        (evt) => {
          if (evt.type === 'progress') {
            setActiveNode(evt.node)
            setCompletedNodes((prev) =>
              prev.includes(evt.node) ? prev : [...prev, evt.node],
            )
          } else if (evt.type === 'token') {
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantTurnId
                  ? { ...t, content: t.content + evt.text }
                  : t,
              ),
            )
          } else if (evt.type === 'final') {
            setSessionId(evt.session_id)
            setActiveNode(null)
            setCompletedNodes(evt.trace)
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantTurnId
                  ? {
                      ...t,
                      content: evt.message,
                      intent: evt.intent,
                      verificationStatus: evt.verification_status,
                      wasRefused: evt.was_refused,
                      sources: evt.sources,
                      claims: evt.claims,
                      trace: evt.trace,
                      isStreaming: false,
                    }
                  : t,
              ),
            )
            setInspectedSources(evt.sources)
            setInspectedClaims(evt.claims)
            setActiveSourceIndex(0)
          } else if (evt.type === 'error') {
            setGlobalError(evt.detail)
            if (evt.detail.toLowerCase().includes('vector index')) {
              setNeedsIndexHint(true)
            }
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantTurnId
                  ? {
                      ...t,
                      content: evt.detail,
                      wasRefused: true,
                      verificationStatus: 'refused',
                      isStreaming: false,
                    }
                  : t,
              ),
            )
          }
        },
      )
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : 'Failed to stream response.'
      setGlobalError(msg)
      if (msg.toLowerCase().includes('vector index')) {
        setNeedsIndexHint(true)
      }
      setTurns((prev) =>
        prev.map((t) =>
          t.id === assistantTurnId
            ? {
                ...t,
                content: 'Unable to complete the query. Please retry.',
                wasRefused: true,
                verificationStatus: 'refused',
                isStreaming: false,
              }
            : t,
        ),
      )
    } finally {
      setChatLoading(false)
      setActiveNode(null)
    }
  }

  const handleGenerateBrief = async (forceRefresh: boolean) => {
    if (!selectedDocId || summaryLoading) return
    setSummaryLoading(true)
    setGlobalError('')
    try {
      const summary = await generateSummary(selectedDocId, {
        summary_type: summaryType,
        language,
        force_refresh: forceRefresh,
      })
      setActiveSummary(summary)
      const detail = await getDocument(selectedDocId)
      setSelectedDoc(detail)
    } catch (err) {
      setGlobalError(
        err instanceof Error ? err.message : 'Failed to generate summary.',
      )
    } finally {
      setSummaryLoading(false)
    }
  }

  const isDocReady = selectedDoc?.status === 'completed'
  const isBusy =
    uploading ||
    indexing ||
    (selectedDoc !== null &&
      selectedDoc.status !== 'completed' &&
      selectedDoc.status !== 'failed')

  const currentSource = inspectedSources[activeSourceIndex] ?? null

  return (
    <div className="chamber-shell">
      {/* ── Masthead ── */}
      <header className="masthead">
        <div className="masthead-row">
          <div className="brand-lockup">
            <div className="brand-emblem">
              <Scale size={20} />
            </div>
            <div>
              <h1 className="brand-title">NYAYASETU</h1>
              <p className="brand-subtitle">
                Indian Legal Document Intelligence Workspace
              </p>
            </div>
          </div>

          <nav className="mode-switch" aria-label="Workspace mode">
            <button
              type="button"
              className={`mode-btn ${activeTab === 'qa' ? 'active' : ''}`}
              onClick={() => setActiveTab('qa')}
            >
              Document QA
            </button>
            <button
              type="button"
              className={`mode-btn ${activeTab === 'brief' ? 'active' : ''}`}
              onClick={() => setActiveTab('brief')}
            >
              Structured Brief
            </button>
          </nav>

          <div className="masthead-controls">
            <label className="lang-select-wrap">
              <span>Language</span>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value as LanguageCode)}
                disabled={chatLoading || summaryLoading}
              >
                <option value="en">English</option>
                <option value="hi">हिंदी (Hindi)</option>
                <option value="mr">मराठी (Marathi)</option>
              </select>
            </label>

            <div className="sys-pill">
              <span
                className={`sys-dot ${backendOnline ? '' : 'offline'}`}
              />
              {backendOnline ? 'LANGGRAPH READY' : 'BACKEND OFFLINE'}
            </div>
          </div>
        </div>
      </header>

      {/* ── 3-Pane Workspace ── */}
      <main className="chamber-grid">
        {/* ── LEFT PANE: LODGED DOSSIERS ── */}
        <aside className="pane dossier-pane">
          <div className="pane-title-bar">
            <h2 className="pane-kicker">Lodged Dossiers</h2>
            <button
              type="button"
              className="icon-btn"
              title="Refresh documents"
              onClick={() => void refreshDocuments()}
            >
              <RefreshCw size={14} />
            </button>
          </div>

          <div className="upload-box">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFileChange}
              hidden
            />

            {!stagedFile ? (
              <button
                type="button"
                className="dropzone-btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading || indexing}
              >
                <Upload size={20} />
                <strong>Lodge Legal PDF</strong>
                <span>Judgments, contracts, notices (max 50 MB)</span>
              </button>
            ) : (
              <div className="staged-file">
                <FileText size={18} />
                <div className="staged-file-meta">
                  <strong>{stagedFile.name}</strong>
                  <span>{(stagedFile.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
                <button
                  type="button"
                  className="icon-btn"
                  onClick={() => {
                    setStagedFile(null)
                    if (fileInputRef.current) fileInputRef.current.value = ''
                  }}
                >
                  <X size={15} />
                </button>
              </div>
            )}

            <div className="ocr-row">
              <label htmlFor="ocr-lang">OCR Language</label>
              <select
                id="ocr-lang"
                value={ocrLanguage}
                onChange={(e) =>
                  setOcrLanguage(e.target.value as OcrLanguageCode)
                }
                disabled={uploading}
              >
                <option value="eng">eng (Fast)</option>
                <option value="eng+hin">eng+hin</option>
                <option value="eng+mar">eng+mar</option>
                <option value="eng+hin+mar">eng+hin+mar</option>
              </select>
            </div>

            {stagedFile && (
              <button
                type="button"
                className="btn-primary"
                onClick={() => void handleUpload()}
                disabled={uploading || indexing}
              >
                {uploading ? (
                  <Loader2 size={15} className="spin" />
                ) : (
                  <Upload size={15} />
                )}
                {uploading ? 'Processing...' : 'Upload, Extract & Index'}
              </button>
            )}

            {isBusy && (
              <div className="progress-card">
                <div className="progress-head">
                  <span>
                    <Loader2 size={13} className="spin" />
                    {indexing ? 'Indexing vectors' : 'Processing PDF'}
                  </span>
                  <span>{progress}%</span>
                </div>
                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{ width: `${Math.min(100, Math.max(5, progress))}%` }}
                  />
                </div>
                {statusNote && <p className="progress-note">{statusNote}</p>}
              </div>
            )}
          </div>

          <div className="dossier-list">
            {documents.length === 0 ? (
              <div className="dossier-empty">
                No documents lodged yet. Upload a PDF above to begin.
              </div>
            ) : (
              documents.map((doc) => {
                const badgeClass =
                  doc.status === 'completed'
                    ? 'completed'
                    : doc.status === 'failed'
                      ? 'failed'
                      : 'processing'
                return (
                  <div
                    key={doc.id}
                    className={`dossier-item ${selectedDocId === doc.id ? 'selected' : ''}`}
                    onClick={() => handleSelectDossier(doc.id)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        handleSelectDossier(doc.id)
                      }
                    }}
                  >
                    <div className="dossier-item-top">
                      <div className="dossier-name">
                        <FileText size={15} />
                        <strong>{doc.filename}</strong>
                      </div>
                      <button
                        type="button"
                        className="icon-btn danger"
                        title="Delete document"
                        onClick={(e) => void handleDeleteDossier(doc.id, e)}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                    <div className="dossier-item-bottom">
                      <span className={`status-badge ${badgeClass}`}>
                        {doc.status === 'completed' && (
                          <CheckCircle2 size={11} />
                        )}
                        {doc.status}
                      </span>
                      <span
                        style={{
                          fontFamily: 'var(--font-mono)',
                          fontSize: '10px',
                          color: 'var(--ink-muted)',
                        }}
                      >
                        {Math.round(doc.progress || 0)}%
                      </span>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </aside>

        {/* ── CENTER PANE: COUNSEL Q&A OR STRUCTURED BRIEF ── */}
        <section className="pane center-pane">
          <div className="pane-title-bar">
            <h2 className="pane-kicker">
              {activeTab === 'qa'
                ? 'Counsel Q&A Workspace'
                : 'Structured Legal Brief'}
            </h2>
            {selectedDoc && (
              <div className="case-strip-actions">
                <button
                  type="button"
                  className="btn-secondary"
                  disabled={!isDocReady || indexing}
                  onClick={() => void handleBuildIndex(selectedDoc.id)}
                  title="Build or rebuild FAISS + BM25 search index"
                >
                  {indexing ? (
                    <Loader2 size={13} className="spin" />
                  ) : (
                    <Database size={13} />
                  )}
                  Index
                </button>
                {activeTab === 'qa' && (
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={() => {
                      setTurns([])
                      setInspectedSources([])
                      setInspectedClaims([])
                      setCompletedNodes([])
                      setSessionId(crypto.randomUUID())
                    }}
                  >
                    New Thread
                  </button>
                )}
              </div>
            )}
          </div>

          {selectedDoc && (
            <div className="case-strip">
              <div className="case-strip-title">
                <BookOpen size={16} />
                <strong>{selectedDoc.original_filename}</strong>
              </div>
              <div className="case-strip-actions">
                <span className="meta-chip">{selectedDoc.page_count} pages</span>
                <span className="meta-chip">
                  {selectedDoc.document_type || 'legal'}
                </span>
                {selectedDoc.pdf_type && (
                  <span className="meta-chip">{selectedDoc.pdf_type}</span>
                )}
              </div>
            </div>
          )}

          {globalError && (
            <div className="error-banner">
              <div className="error-banner-text">
                <AlertCircle size={16} />
                <span>{globalError}</span>
              </div>
              <div className="case-strip-actions">
                {needsIndexHint && selectedDocId && (
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={() => void handleBuildIndex(selectedDocId)}
                  >
                    Index Now
                  </button>
                )}
                <button
                  type="button"
                  className="icon-btn"
                  onClick={() => setGlobalError('')}
                >
                  <X size={14} />
                </button>
              </div>
            </div>
          )}

          {activeTab === 'qa' ? (
            <>
              {/* Live LangGraph Pipeline Stepper */}
              <div className="pipeline-bar">
                <div className="pipeline-label">
                  LangGraph Verification Pipeline
                </div>
                <div className="pipeline-steps">
                  {QA_PIPELINE_NODES.map((node, idx) => {
                    const isActive = activeNode === node.key
                    const isDone =
                      !isActive && completedNodes.includes(node.key)
                    return (
                      <span
                        key={node.key}
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '6px',
                        }}
                      >
                        <span
                          className={`pipeline-node ${
                            isActive ? 'active' : isDone ? 'done' : ''
                          }`}
                        >
                          {isActive && <Loader2 size={10} className="spin" />}
                          {node.label}
                        </span>
                        {idx < QA_PIPELINE_NODES.length - 1 && (
                          <span className="pipeline-arrow">→</span>
                        )}
                      </span>
                    )
                  })}
                </div>
              </div>

              {/* Conversation Feed */}
              <div className="qa-scroll">
                {!selectedDoc ? (
                  <div className="empty-state">
                    <Scale size={30} color="var(--accent-vermilion)" />
                    <h2>Select or Lodge a Legal Dossier</h2>
                    <p>
                      Upload an Indian court judgment, contract, or legal notice
                      in English, Hindi, or Marathi. Every answer is retrieved
                      via hybrid search and verified against cited passages.
                    </p>
                  </div>
                ) : turns.length === 0 ? (
                  <div className="empty-state">
                    <h2>Examine This Document</h2>
                    <p>
                      Ask factual questions, look up specific sections, or ask
                      for plain-language explanations. Follow-up questions keep
                      context across turns.
                    </p>
                    <div className="prompt-chips">
                      {SUGGESTED_QUESTIONS.map((sample) => (
                        <button
                          key={sample}
                          type="button"
                          className="prompt-chip"
                          onClick={() => setQuestion(sample)}
                        >
                          {sample}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  turns.map((turn) =>
                    turn.role === 'user' ? (
                      <article key={turn.id} className="qa-card user-q">
                        <div className="qa-card-head">
                          <span className="qa-role-tag">Question</span>
                        </div>
                        <p className="qa-question-text">{turn.content}</p>
                      </article>
                    ) : (
                      <article key={turn.id} className="qa-card assistant-a">
                        <div className="qa-card-head">
                          <span className="qa-role-tag">
                            Verified Response
                            {turn.intent ? ` · ${turn.intent}` : ''}
                          </span>
                          {turn.verificationStatus && (
                            <span
                              className={`verify-stamp ${
                                turn.verificationStatus === 'verified'
                                  ? 'verified'
                                  : turn.verificationStatus === 'refused'
                                    ? 'refused'
                                    : 'partial'
                              }`}
                            >
                              {turn.verificationStatus === 'verified' ? (
                                <ShieldCheck size={12} />
                              ) : (
                                <ShieldAlert size={12} />
                              )}
                              {turn.verificationStatus.replace('_', ' ')}
                            </span>
                          )}
                        </div>

                        {turn.isStreaming && !turn.content ? (
                          <div
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '8px',
                              color: 'var(--ink-muted)',
                              fontSize: '13px',
                            }}
                          >
                            <Loader2 size={15} className="spin" />
                            <span>
                              Running LangGraph node:{' '}
                              <strong>{activeNode || 'starting'}</strong>...
                            </span>
                          </div>
                        ) : (
                          <div className="markdown-body">
                            <ReactMarkdown>{turn.content}</ReactMarkdown>
                          </div>
                        )}

                        {turn.sources && turn.sources.length > 0 && (
                          <div className="citation-row">
                            <span className="citation-label">
                              Cited Passages:
                            </span>
                            {turn.sources.map((src, sIdx) => {
                              const isSelected =
                                inspectedSources === turn.sources &&
                                activeSourceIndex === sIdx
                              return (
                                <button
                                  key={sIdx}
                                  type="button"
                                  className={`citation-chip ${isSelected ? 'active' : ''}`}
                                  onClick={() => {
                                    setInspectedSources(turn.sources || [])
                                    setInspectedClaims(turn.claims || [])
                                    setActiveSourceIndex(sIdx)
                                  }}
                                >
                                  S{sIdx + 1}
                                  {src.page ? ` · p.${src.page}` : ''}
                                </button>
                              )
                            })}
                          </div>
                        )}

                        {turn.trace && turn.trace.length > 0 && (
                          <div className="trace-footnote">
                            Trace: {turn.trace.join(' → ')}
                          </div>
                        )}
                      </article>
                    ),
                  )
                )}
                <div ref={qaEndRef} />
              </div>

              {/* Question Input */}
              <form className="composer" onSubmit={(e) => void handleAskQuestion(e)}>
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder={
                    isDocReady
                      ? 'Ask a grounded question about this legal document...'
                      : 'Select or upload a completed document to begin...'
                  }
                  disabled={!isDocReady || chatLoading || indexing}
                />
                <button
                  type="submit"
                  className="composer-send"
                  disabled={!question.trim() || !isDocReady || chatLoading}
                >
                  {chatLoading ? (
                    <Loader2 size={15} className="spin" />
                  ) : (
                    <Send size={15} />
                  )}
                  Ask
                </button>
              </form>
            </>
          ) : (
            /* ── STRUCTURED BRIEF TAB ── */
            <>
              <div className="brief-toolbar">
                <div className="brief-controls">
                  <select
                    value={summaryType}
                    onChange={(e) =>
                      setSummaryType(e.target.value as SummaryLength)
                    }
                    disabled={summaryLoading}
                  >
                    <option value="short">Short Brief</option>
                    <option value="medium">Medium Brief</option>
                    <option value="detailed">Detailed Brief</option>
                  </select>

                  <button
                    type="button"
                    className="btn-primary"
                    style={{ width: 'auto', marginTop: 0 }}
                    disabled={!isDocReady || summaryLoading}
                    onClick={() => void handleGenerateBrief(Boolean(activeSummary))}
                  >
                    {summaryLoading ? (
                      <Loader2 size={14} className="spin" />
                    ) : (
                      <BookOpen size={14} />
                    )}
                    {summaryLoading
                      ? 'Generating Verified Brief...'
                      : activeSummary
                        ? 'Regenerate Brief'
                        : 'Generate Structured Brief'}
                  </button>
                </div>

                {activeSummary && selectedDocId && (
                  <div className="brief-controls">
                    <a
                      className="btn-secondary"
                      href={getSummaryExportUrl(
                        selectedDocId,
                        activeSummary.summary_type,
                        activeSummary.language,
                        'md',
                      )}
                    >
                      <Download size={13} />
                      Export .MD
                    </a>
                    <a
                      className="btn-secondary"
                      href={getSummaryExportUrl(
                        selectedDocId,
                        activeSummary.summary_type,
                        activeSummary.language,
                        'json',
                      )}
                    >
                      <Download size={13} />
                      Export .JSON
                    </a>
                  </div>
                )}
              </div>

              <div className="brief-scroll">
                {!selectedDoc ? (
                  <div className="empty-state">
                    <h2>No Document Selected</h2>
                    <p>Select a document from the left pane to generate its structured legal brief.</p>
                  </div>
                ) : summaryLoading ? (
                  <div className="empty-state">
                    <Loader2 size={28} className="spin" color="var(--accent-vermilion)" />
                    <h2>Drafting & Verifying Brief</h2>
                    <p>
                      Detecting document type, summarizing sections hierarchically,
                      and verifying claims against document pages...
                    </p>
                  </div>
                ) : !activeSummary ? (
                  <div className="empty-state">
                    <h2>Structured Legal Digest</h2>
                    <p>
                      Generate an Indian Law Report–style judgment brief or a
                      clause-by-clause contract risk brief with page references
                      and claim verification.
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="brief-meta-grid">
                      <div className="brief-meta-card">
                        <span>Document Type</span>
                        <strong>{activeSummary.document_type.toUpperCase()}</strong>
                      </div>
                      <div className="brief-meta-card">
                        <span>Verification</span>
                        <strong>
                          {activeSummary.verification_status.replace('_', ' ').toUpperCase()}
                        </strong>
                      </div>
                      {activeSummary.structured_summary.court && (
                        <div className="brief-meta-card">
                          <span>Court / Forum</span>
                          <strong>{activeSummary.structured_summary.court}</strong>
                        </div>
                      )}
                      {activeSummary.structured_summary.case_number && (
                        <div className="brief-meta-card">
                          <span>Case Number</span>
                          <strong>
                            {activeSummary.structured_summary.case_number}
                          </strong>
                        </div>
                      )}
                      {(activeSummary.structured_summary.date ||
                        activeSummary.structured_summary.effective_date) && (
                        <div className="brief-meta-card">
                          <span>Date</span>
                          <strong>
                            {activeSummary.structured_summary.date ||
                              activeSummary.structured_summary.effective_date}
                          </strong>
                        </div>
                      )}
                    </div>

                    <div className="brief-section-card">
                      <h4>Executive Digest</h4>
                      <div className="markdown-body">
                        <ReactMarkdown>{activeSummary.content}</ReactMarkdown>
                      </div>
                    </div>

                    {activeSummary.structured_summary.holding && (
                      <div className="brief-section-card">
                        <h4>Holding & Final Order</h4>
                        <p style={{ margin: '0 0 8px', fontSize: '13.5px', lineHeight: 1.65 }}>
                          <strong>Holding: </strong>
                          {activeSummary.structured_summary.holding}
                        </p>
                        {activeSummary.structured_summary.final_order && (
                          <p style={{ margin: 0, fontSize: '13.5px', lineHeight: 1.65 }}>
                            <strong>Final Order: </strong>
                            {activeSummary.structured_summary.final_order}
                          </p>
                        )}
                      </div>
                    )}

                    {activeSummary.structured_summary.parties &&
                      activeSummary.structured_summary.parties.length > 0 && (
                        <div className="brief-section-card">
                          <h4>Parties</h4>
                          <ul className="brief-list">
                            {activeSummary.structured_summary.parties.map((p, i) => (
                              <li key={i}>{p}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                    {activeSummary.structured_summary.issues &&
                      activeSummary.structured_summary.issues.length > 0 && (
                        <div className="brief-section-card">
                          <h4>Issues & Reasoning</h4>
                          <ul className="brief-list">
                            {activeSummary.structured_summary.issues.map((item, i) => (
                              <li key={`iss-${i}`}>{item}</li>
                            ))}
                            {(activeSummary.structured_summary.reasoning || []).map(
                              (item, i) => (
                                <li key={`rsn-${i}`}>{item}</li>
                              ),
                            )}
                          </ul>
                        </div>
                      )}

                    {activeSummary.structured_summary.unusual_or_one_sided_clauses &&
                      activeSummary.structured_summary.unusual_or_one_sided_clauses
                        .length > 0 && (
                        <div className="brief-section-card">
                          <h4>Unusual or One-Sided Clauses</h4>
                          {activeSummary.structured_summary.unusual_or_one_sided_clauses.map(
                            (clause, i) => (
                              <div key={i} className="risk-clause-box">
                                <strong>{clause.clause_reference}</strong>
                                <span>{clause.risk_explanation}</span>
                              </div>
                            ),
                          )}
                        </div>
                      )}

                    {activeSummary.trace.length > 0 && (
                      <div className="trace-footnote">
                        LangGraph Trace: {activeSummary.trace.join(' → ')}
                      </div>
                    )}
                  </>
                )}
              </div>
            </>
          )}
        </section>

        {/* ── RIGHT PANE: EVIDENCE INSPECTOR ── */}
        <aside className="pane evidence-pane">
          <div className="pane-title-bar">
            <h2 className="pane-kicker">Evidence Inspector</h2>
            {activeTab === 'qa' && inspectedSources.length > 0 && (
              <span className="meta-chip">
                {inspectedSources.length} sources
              </span>
            )}
          </div>

          <div className="evidence-body">
            {activeTab === 'brief' ? (
              activeSummary && activeSummary.claims.length > 0 ? (
                <div className="claims-box" style={{ borderTop: 0, paddingTop: 0 }}>
                  <h5>Verified Summary Claims ({activeSummary.claims.length})</h5>
                  {activeSummary.claims.map((claim, idx) => (
                    <div key={idx} className="claim-item">
                      <div>{claim.text}</div>
                      {claim.pages.length > 0 && (
                        <div className="claim-item-meta">
                          <ShieldCheck size={12} />
                          Pages: {claim.pages.join(', ')}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="dossier-empty">
                  Verified summary claims and page mappings will appear here
                  after generating a brief.
                </div>
              )
            ) : currentSource ? (
              <>
                <div className="evidence-tabs">
                  {inspectedSources.map((src, idx) => (
                    <button
                      key={idx}
                      type="button"
                      className={`citation-chip ${activeSourceIndex === idx ? 'active' : ''}`}
                      onClick={() => setActiveSourceIndex(idx)}
                    >
                      S{idx + 1}
                      {src.page ? ` (p.${src.page})` : ''}
                    </button>
                  ))}
                </div>

                <div className="evidence-card">
                  <div className="evidence-meta">
                    <span className="evidence-badge">
                      SOURCE S{activeSourceIndex + 1}
                    </span>
                    {currentSource.page !== null && (
                      <span className="meta-chip">Page {currentSource.page}</span>
                    )}
                    {currentSource.section && (
                      <span className="meta-chip">{currentSource.section}</span>
                    )}
                    {currentSource.paragraph && (
                      <span className="meta-chip">
                        Para {currentSource.paragraph}
                      </span>
                    )}
                  </div>
                  <p className="evidence-excerpt">
                    {currentSource.content || 'No excerpt text available.'}
                  </p>
                </div>

                {inspectedClaims.length > 0 && (
                  <div className="claims-box">
                    <h5>Verified Claims ({inspectedClaims.length})</h5>
                    {inspectedClaims.map((claim, idx) => (
                      <div key={idx} className="claim-item">
                        <div>{claim.text}</div>
                        <div className="claim-item-meta">
                          <ShieldCheck size={12} />
                          <span>{claim.source_ids.join(', ')}</span>
                          {claim.pages.length > 0 && (
                            <span>· Page {claim.pages.join(', ')}</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="dossier-empty">
                Ask a question to inspect the exact supporting document passages,
                page numbers, and verified claims here.
              </div>
            )}
          </div>
        </aside>
      </main>

      <footer className="chamber-footer">
        <span>
          <strong>NyayaSetu</strong> — Document-Grounded Indian Legal AI
        </span>
        <span>
          Informational assistance only · Not a substitute for professional legal advice
        </span>
      </footer>
    </div>
  )
}

export default App