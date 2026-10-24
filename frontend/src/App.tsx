import { useRef, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import {
  FileText,
  Upload,
  Send,
  Loader2,
  CheckCircle2,
  AlertCircle,
  BookOpen,
  MessageSquare,
  Languages,
  X,
} from 'lucide-react'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000/api'

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

type Source = {
  chunk_id?: string
  page_number?: number
  section_type?: string
  score?: number
}

type DocumentStatus =
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

const languageCodeMap: Record<string, string> = {
  English: 'en',
  Hindi: 'hi',
  Marathi: 'mr',
}

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [documentId, setDocumentId] = useState<string | null>(null)

  // One chat session for the current browser session.
  const [sessionId, setSessionId] = useState<string>(() =>
    crypto.randomUUID(),
  )

  const [status, setStatus] = useState<DocumentStatus | null>(null)
  const [statusMessage, setStatusMessage] = useState('')
  const [progress, setProgress] = useState(0)

  const [uploading, setUploading] = useState(false)
  const [indexing, setIndexing] = useState(false)

  const [question, setQuestion] = useState('')
  const [language, setLanguage] = useState('English')

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sources, setSources] = useState<Source[]>([])

  const [chatLoading, setChatLoading] = useState(false)
  const [error, setError] = useState('')

  const isProcessing =
    uploading ||
    indexing ||
    (status !== null &&
      status !== 'completed' &&
      status !== 'failed')

  const handleFileSelect = (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0]

    if (!file) return

    setError('')
    setSelectedFile(null)

    if (
      file.type !== 'application/pdf' &&
      !file.name.toLowerCase().endsWith('.pdf')
    ) {
      setError('Please select a PDF file.')
      return
    }

    const maxSize = 50 * 1024 * 1024

    if (file.size > maxSize) {
      setError('PDF size must be 50 MB or less.')
      return
    }

    setSelectedFile(file)
  }

  const resetDocument = () => {
    setSelectedFile(null)
    setDocumentId(null)
    setStatus(null)
    setStatusMessage('')
    setProgress(0)
    setMessages([])
    setSources([])
    setError('')
    setSessionId(crypto.randomUUID())

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const uploadDocument = async () => {
    if (!selectedFile) {
      setError('Please select a PDF first.')
      return
    }

    setUploading(true)
    setError('')
    setMessages([])
    setSources([])
    setDocumentId(null)
    setStatus('uploaded')
    setProgress(5)
    setStatusMessage('Uploading document...')
    setSessionId(crypto.randomUUID())

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await axios.post(
        `${API_BASE}/documents/upload`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        },
      )

      // Backend returns "id", not "document_id".
      const id = response.data.id

      if (!id) {
        throw new Error(
          'Backend did not return a document ID.',
        )
      }

      setDocumentId(id)
      setStatus(response.data.status || 'uploaded')
      setStatusMessage(
        response.data.message ||
          'Document uploaded successfully.',
      )
      setProgress(10)

      await pollDocumentStatus(id)
    } catch (err) {
      console.error(err)

      if (axios.isAxiosError(err)) {
        setError(
          err.response?.data?.detail ||
            'Failed to upload the document. Make sure the backend is running.',
        )
      } else if (err instanceof Error) {
        setError(err.message)
      } else {
        setError('An unexpected error occurred during upload.')
      }

      setStatus('failed')
    } finally {
      setUploading(false)
    }
  }

  const pollDocumentStatus = async (id: string) => {
    const maxAttempts = 180

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const response = await axios.get(
          `${API_BASE}/documents/${id}`,
        )

        const document = response.data

        setStatus(document.status)
        setStatusMessage(
          document.status_message ||
            'Processing document...',
        )

        if (typeof document.progress === 'number') {
          setProgress(document.progress)
        }

        if (document.status === 'completed') {
          setProgress(90)
          setStatusMessage(
            'Document processing completed. Building semantic search index...',
          )

          await buildIndex(id)
          return
        }

        if (document.status === 'failed') {
          setError(
            document.status_message ||
              'Document processing failed.',
          )
          return
        }
      } catch (err) {
        console.error(
          'Document status polling error:',
          err,
        )
      }

      await new Promise((resolve) =>
        setTimeout(resolve, 2000),
      )
    }

    setError(
      'Document processing is taking longer than expected.',
    )
    setStatus('failed')
  }

  const buildIndex = async (id: string) => {
    setIndexing(true)
    setStatus('embedding')
    setProgress(95)
    setStatusMessage(
      'Building semantic search index...',
    )

    try {
      const response = await axios.post(
        `${API_BASE}/documents/${id}/index`,
      )

      setStatus('completed')
      setProgress(100)
      setStatusMessage(
        response.data.message ||
          'Document is ready for questions.',
      )
    } catch (err) {
      console.error(err)

      if (axios.isAxiosError(err)) {
        setError(
          err.response?.data?.detail ||
            'Failed to build the semantic search index.',
        )
      } else {
        setError(
          'Failed to build the semantic search index.',
        )
      }

      setStatus('failed')
    } finally {
      setIndexing(false)
    }
  }

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()

    const userQuestion = question.trim()

    if (!userQuestion) return

    if (!documentId) {
      setError(
        'Please upload and process a legal document first.',
      )
      return
    }

    if (status !== 'completed') {
      setError(
        'Please wait until the document is ready.',
      )
      return
    }

    setQuestion('')
    setError('')

    setMessages((previous) => [
      ...previous,
      {
        role: 'user',
        content: userQuestion,
      },
    ])

    setChatLoading(true)

    try {
      const response = await axios.post(
        `${API_BASE}/chat/${documentId}`,
        {
          // IMPORTANT:
          // Backend expects "message", not "question".
          message: userQuestion,

          // Backend expects a session UUID.
          session_id: sessionId,

          // Backend expects language codes.
          language:
            languageCodeMap[language] || 'en',
        },
      )

      const answer =
        response.data?.answer ||
        response.data?.message ||
        'No answer returned.'

      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          content: String(answer),
        },
      ])

      const returnedSources =
        Array.isArray(response.data?.sources)
          ? response.data.sources
          : []

      setSources(returnedSources)
    } catch (err) {
      console.error('Chat request failed:', err)

      let errorMessage =
        'Failed to get an answer. Please try again.'

      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail

        if (typeof detail === 'string') {
          errorMessage = detail
        } else if (Array.isArray(detail)) {
          errorMessage = detail
            .map((item) => item?.msg || String(item))
            .join(', ')
        }
      } else if (err instanceof Error) {
        errorMessage = err.message
      }

      setError(errorMessage)

      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          content:
            'I could not process that question. Please try again.',
        },
      ])
    } finally {
      setChatLoading(false)
    }
  }

  const getStatusLabel = () => {
    if (uploading) return 'Uploading PDF'
    if (indexing) return 'Building search index'

    switch (status) {
      case 'uploaded':
        return 'Document uploaded'
      case 'validating':
        return 'Validating PDF'
      case 'extracting':
        return 'Extracting text'
      case 'ocr_processing':
        return 'Running OCR'
      case 'preprocessing':
        return 'Preprocessing document'
      case 'analyzing':
        return 'Analyzing legal structure'
      case 'embedding':
        return 'Creating embeddings'
      case 'summarizing':
        return 'Generating summary'
      case 'completed':
        return 'Document ready'
      case 'failed':
        return 'Processing failed'
      default:
        return 'Waiting for document'
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <BookOpen size={22} />
          </div>

          <div>
            <h1>NyayaSetu</h1>
            <p>Legal documents, explained simply.</p>
          </div>
        </div>

        <div className="header-badge">
          <span className="status-dot" />
          Legal Document Assistant
        </div>
      </header>

      <main className="main-container">
        <section className="hero">
          <h2>Understand your legal document</h2>

          <p>
            Upload an Indian legal document and ask questions
            about its contents in simple language.
          </p>
        </section>

        <div className="workspace">
          <aside className="sidebar">
            <div className="card upload-card">
              <div className="card-header">
                <h3>
                  <FileText size={18} />
                  Legal Document
                </h3>

                <p>
                  Upload a PDF judgment or legal document.
                </p>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                onChange={handleFileSelect}
                hidden
              />

              {!selectedFile ? (
                <button
                  className="upload-zone"
                  onClick={() =>
                    fileInputRef.current?.click()
                  }
                  type="button"
                >
                  <Upload size={28} />
                  <strong>Choose PDF</strong>
                  <span>
                    Click to select a legal document
                  </span>
                  <small>Maximum size: 50 MB</small>
                </button>
              ) : (
                <div className="selected-file">
                  <div className="file-icon">
                    <FileText size={22} />
                  </div>

                  <div className="file-details">
                    <strong>{selectedFile.name}</strong>
                    <span>
                      {(
                        selectedFile.size /
                        1024 /
                        1024
                      ).toFixed(2)}{' '}
                      MB
                    </span>
                  </div>

                  {!isProcessing && (
                    <button
                      className="icon-button"
                      onClick={resetDocument}
                      title="Remove document"
                      type="button"
                    >
                      <X size={18} />
                    </button>
                  )}
                </div>
              )}

              {selectedFile &&
                !documentId &&
                !uploading && (
                  <button
                    className="primary-button"
                    onClick={uploadDocument}
                    type="button"
                  >
                    <Upload size={17} />
                    Upload & Process
                  </button>
                )}

              {documentId &&
                (isProcessing ||
                  status === 'completed') && (
                  <div className="processing-panel">
                    <div className="processing-header">
                      <span>
                        {status === 'completed' ? (
                          <CheckCircle2 size={17} />
                        ) : (
                          <Loader2
                            size={17}
                            className="spin"
                          />
                        )}

                        {getStatusLabel()}
                      </span>

                      <strong>{progress}%</strong>
                    </div>

                    <div className="progress-track">
                      <div
                        className="progress-bar"
                        style={{
                          width: `${progress}%`,
                        }}
                      />
                    </div>

                    <p>{statusMessage}</p>
                  </div>
                )}

              {status === 'completed' &&
                documentId && (
                  <div className="ready-message">
                    <CheckCircle2 size={18} />

                    <div>
                      <strong>Document ready</strong>
                      <span>
                        You can now ask questions about this
                        document.
                      </span>
                    </div>
                  </div>
                )}

              {status === 'failed' && (
                <div className="error-box">
                  <AlertCircle size={18} />
                  <span>
                    {error ||
                      'Document processing failed.'}
                  </span>
                </div>
              )}

              {error &&
                status !== 'failed' && (
                  <div className="error-box">
                    <AlertCircle size={18} />
                    <span>{error}</span>
                  </div>
                )}
            </div>

            <div className="card language-card">
              <h3>
                <Languages size={18} />
                Response Language
              </h3>

              <select
                value={language}
                onChange={(event) =>
                  setLanguage(event.target.value)
                }
                disabled={chatLoading}
              >
                <option value="English">
                  English
                </option>
                <option value="Hindi">
                  Hindi
                </option>
                <option value="Marathi">
                  Marathi
                </option>
              </select>
            </div>

            <div className="info-card">
              <strong>Important</strong>

              <p>
                NyayaSetu provides informational assistance and
                does not replace advice from a qualified lawyer.
              </p>
            </div>
          </aside>

          <section className="chat-panel">
            <div className="chat-header">
              <div>
                <h3>
                  <MessageSquare size={19} />
                  Document Chat
                </h3>

                <p>
                  Ask questions about the uploaded document.
                </p>
              </div>

              {documentId &&
                status === 'completed' && (
                  <span className="ready-pill">
                    <CheckCircle2 size={14} />
                    Ready
                  </span>
                )}
            </div>

            <div className="messages">
              {messages.length === 0 ? (
                <div className="empty-chat">
                  <div className="empty-icon">
                    <MessageSquare size={30} />
                  </div>

                  <h3>Ask about your document</h3>

                  <p>
                    Upload a legal PDF and ask questions such
                    as:
                  </p>

                  <div className="question-examples">
                    <button
                      type="button"
                      onClick={() =>
                        setQuestion(
                          'What is this case mainly about?',
                        )
                      }
                    >
                      What is this case mainly about?
                    </button>

                    <button
                      type="button"
                      onClick={() =>
                        setQuestion(
                          'What were the main arguments of the parties?',
                        )
                      }
                    >
                      What were the main arguments?
                    </button>

                    <button
                      type="button"
                      onClick={() =>
                        setQuestion(
                          'What was the final decision?',
                        )
                      }
                    >
                      What was the final decision?
                    </button>
                  </div>
                </div>
              ) : (
                messages.map((message, index) => (
                  <div
                    className={`message-row ${message.role}`}
                    key={index}
                  >
                    <div className="message-bubble">
                      {message.role === 'assistant' ? (
                        <ReactMarkdown>
                          {message.content}
                        </ReactMarkdown>
                      ) : (
                        <p>{message.content}</p>
                      )}
                    </div>
                  </div>
                ))
              )}

              {chatLoading && (
                <div className="message-row assistant">
                  <div className="message-bubble loading-message">
                    <Loader2
                      size={18}
                      className="spin"
                    />
                    <span>
                      Analyzing the document...
                    </span>
                  </div>
                </div>
              )}
            </div>

            {sources.length > 0 && (
              <div className="sources">
                <strong>
                  Sources from document
                </strong>

                <div className="source-list">
                  {sources.map(
                    (source, index) => (
                      <div
                        className="source-item"
                        key={index}
                      >
                        <span>
                          Source {index + 1}
                        </span>

                        {source.page_number && (
                          <span>
                            Page{' '}
                            {source.page_number}
                          </span>
                        )}

                        {source.section_type && (
                          <span>
                            {source.section_type}
                          </span>
                        )}

                        {typeof source.score ===
                          'number' && (
                          <span>
                            Relevance:{' '}
                            {(
                              source.score * 100
                            ).toFixed(1)}
                            %
                          </span>
                        )}
                      </div>
                    ),
                  )}
                </div>
              </div>
            )}

            <form
              className="chat-input-area"
              onSubmit={handleSubmit}
            >
              <input
                value={question}
                onChange={(event) =>
                  setQuestion(event.target.value)
                }
                placeholder={
                  status === 'completed'
                    ? 'Ask a question about this document...'
                    : 'Upload a document to start chatting...'
                }
                disabled={
                  status !== 'completed' ||
                  chatLoading ||
                  indexing
                }
              />

              <button
                type="submit"
                disabled={
                  !question.trim() ||
                  status !== 'completed' ||
                  chatLoading
                }
                title="Send question"
              >
                {chatLoading ? (
                  <Loader2
                    size={19}
                    className="spin"
                  />
                ) : (
                  <Send size={19} />
                )}
              </button>
            </form>
          </section>
        </div>
      </main>

      <footer>
        <span>NyayaSetu</span>
        <span>
          Informational assistance only • Not legal advice
        </span>
      </footer>
    </div>
  )
}

export default App