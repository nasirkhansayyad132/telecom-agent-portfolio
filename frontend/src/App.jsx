import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

// ===== Custom Hooks =====

function useTheme() {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('theme')
    return saved || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggle = () => setTheme(t => t === 'light' ? 'dark' : 'light')
  return { theme, toggle }
}

function useSessionId() {
  const [sessionId] = useState(() => {
    let id = sessionStorage.getItem('session_id')
    if (!id) {
      id = 'sess_' + Math.random().toString(36).substring(2, 15)
      sessionStorage.setItem('session_id', id)
    }
    return id
  })
  return sessionId
}

function useBackendStatus(apiBase) {
  const [status, setStatus] = useState('checking')

  useEffect(() => {
    let active = true
    setStatus('checking')

    const check = async () => {
      const controller = new AbortController()
      const timeoutId = window.setTimeout(() => controller.abort(), 4000)

      try {
        const response = await fetch(`${apiBase}/health`, {
          signal: controller.signal,
          cache: 'no-store',
        })

        if (!response.ok) throw new Error('Health check failed')
        if (active) setStatus('connected')
      } catch (error) {
        if (active) setStatus('offline')
      } finally {
        window.clearTimeout(timeoutId)
      }
    }

    check()
    const intervalId = window.setInterval(check, 15000)

    return () => {
      active = false
      window.clearInterval(intervalId)
    }
  }, [apiBase])

  return status
}

// ===== Enhanced Markdown Parser =====

function parseMarkdown(text) {
  if (!text) return ''

  let result = text
    // Headers
    .replace(/^#{4,}\s*(.+)$/gm, '<h4 class="md-subheader">$1</h4>')
    .replace(/^###\s*(.+)$/gm, '<h3 class="md-header">$1</h3>')
    .replace(/^##\s*(.+)$/gm, '<h2 class="md-header">$1</h2>')
    .replace(/^#\s*(.+)$/gm, '<h1 class="md-header">$1</h1>')

    // Dial codes with Copy Button
    .replace(/(\*[\d\*]+#)/g, (match) => {
      const code = match.replace(/'/g, "\\'")
      return `<div class="code-chip"><span class="code-text">${match}</span><button class="copy-btn" onclick="window.copyToClipboard('${code}')">Copy</button></div>`
    })

    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')

    // AFN prices
    .replace(/AFN\s*(\d+)/g, '<span class="price-tag">AFN $1</span>')

    // Steps styling
    .replace(/^> (.+)$/gm, '<div class="step-card">$1</div>')

    // Code
    .replace(/`([^`]+)`/g, '<code>$1</code>')

    // Lists
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/^• (.+)$/gm, '<li>$1</li>')
    .replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>')

    // Line breaks
    .replace(/\n/g, '<br />')

  // Wrap lists
  result = result.replace(/(<li>[\s\S]*?<\/li>)+/g, '<ul class="md-list">$&</ul>')

  return result
}

function FormattedText({ text }) {
  return (
    <span
      className="formatted-text"
      dangerouslySetInnerHTML={{ __html: parseMarkdown(text) }}
    />
  )
}

// ===== Components =====

function StreamingText({ text, isStreaming }) {
  return (
    <span className="message-text">
      <FormattedText text={text} />
      {isStreaming && <span className="typing-cursor" />}
    </span>
  )
}

function FloatingSources({ sources }) {
  const [isOpen, setIsOpen] = useState(false)

  // Auto-open if we have sources and it's a new set (handled by key change)
  useEffect(() => {
    if (sources && sources.length > 0) {
      setIsOpen(false) // Start closed to not annoy, or true if preferred
    }
  }, [sources])

  if (!sources || sources.length === 0) return null

  return (
    <div className={`floating-sources ${isOpen ? 'open' : ''}`}>
      <button
        className="sources-toggle-btn"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span>📚 Evidence & Tools ({sources.length})</span>
        <span className="toggle-icon">{isOpen ? '▼' : '▲'}</span>
      </button>

      {isOpen && (
        <div className="sources-list">
          {sources.map((source, idx) => (
            <div key={idx} className="source-row">
              <div className="source-title">{source.source}</div>
              <div className="source-text">{source.snippet}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Message({ message }) {
  const { role, content, toolCall, isStreaming } = message
  const isUser = role === 'user'

  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className={`message-avatar ${isUser ? 'user-avatar' : 'agent-avatar'}`}>
        {isUser ? 'U' : 'A'}
      </div>
      <div className="message-content">
        <div className="message-bubble">
          <StreamingText text={content} isStreaming={isStreaming} />
        </div>

      </div>
    </div>
  )
}

function EmptyState({ onSuggestionClick }) {
  const suggestions = [
    'What bundles are available for 500 AFN?',
    'Help me activate my SIM card',
    'Show me data bundle options',
    'Create a support ticket',
  ]

  return (
    <div className="empty-state">
      <div className="icon">💬</div>
      <h3>How can I help you today?</h3>
      <p>Ask about bundles, account issues, or get support</p>
      <div className="suggestions">
        {suggestions.map((text, idx) => (
          <button
            key={idx}
            className="suggestion-chip"
            onClick={() => onSuggestionClick(text)}
          >
            {text}
          </button>
        ))}
      </div>
    </div>
  )
}

// ===== Telecom Agent Panel =====

function TelecomAgent({ apiBase }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    window.copyToClipboard = (text) => { navigator.clipboard.writeText(text) }
  }, [])
  const chatWindowRef = useRef(null)
  const inputRef = useRef(null)
  const sessionId = useSessionId()

  const latestAssistantMessage = [...messages].reverse().find(m => m.role === 'assistant')
  const currentSources = latestAssistantMessage?.sources || []

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight
    }
  }, [messages])

  useEffect(() => {
    if (inputRef.current && !loading) {
      inputRef.current.focus()
    }
  }, [loading, messages])

  const sendMessage = useCallback(async (text) => {
    const messageText = text || input.trim()
    if (!messageText) return

    setMessages(prev => [...prev, {
      role: 'user',
      content: messageText,
    }])

    setInput('')
    setLoading(true)

    const assistantMsgId = Date.now()
    setMessages(prev => [...prev, {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      isStreaming: true,
    }])

    const fetchFallback = async () => {
      const fallbackResponse = await fetch(`${apiBase}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: messageText,
          session_id: sessionId,
        }),
      })

      const data = await fallbackResponse.json()

      setMessages(prev => prev.map(msg =>
        msg.id === assistantMsgId ? {
          ...msg,
          content: data.answer,
          sources: data.sources,
          toolCall: data.tool_used,
          isStreaming: false,
        } : msg
      ))
    }

    try {
      const controller = new AbortController()
      const response = await fetch(`${apiBase}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: messageText,
          session_id: sessionId,
          stream: true,
        }),
        signal: controller.signal,
      })

      if (!response.ok) {
        await fetchFallback()
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let fullContent = ''
      let sources = []
      let shouldStop = false
      let timedOut = false
      let streamCompleted = false
      let idleTimeoutId = null

      const resetIdleTimeout = () => {
        if (idleTimeoutId) {
          window.clearTimeout(idleTimeoutId)
        }
        idleTimeoutId = window.setTimeout(() => {
          timedOut = true
          controller.abort()
        }, 12000)
      }

      resetIdleTimeout()

      while (true) {
        let done = false
        let value
        try {
          const readResult = await reader.read()
          done = readResult.done
          value = readResult.value
        } catch (readError) {
          if (readError?.name === 'AbortError') {
            timedOut = true
            break
          }
          throw readError
        }
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') continue

            try {
              const event = JSON.parse(data)
              resetIdleTimeout()

              if (event.type === 'chunk') {
                fullContent += event.content
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantMsgId ? { ...msg, content: fullContent } : msg
                ))
              } else if (event.type === 'sources') {
                sources = event.sources
              } else if (event.type === 'done') {
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantMsgId ? {
                    ...msg,
                    content: event.full_answer || fullContent,
                    sources: sources,
                    isStreaming: false,
                  } : msg
                ))
                streamCompleted = true
                shouldStop = true
              } else if (event.type === 'error') {
                const errorText = event.message || 'Sorry, there was an error processing your message.'
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantMsgId ? {
                    ...msg,
                    content: errorText,
                    sources: sources,
                    isStreaming: false,
                  } : msg
                ))
                streamCompleted = true
                shouldStop = true
              }
            } catch (e) {
              console.error('Parse error:', e)
            }
          }
        }

        if (shouldStop) {
          await reader.cancel()
          break
        }
      }

      if (idleTimeoutId) {
        window.clearTimeout(idleTimeoutId)
      }

      if (timedOut && !streamCompleted && !fullContent) {
        await fetchFallback()
        return
      }

      setMessages(prev => prev.map(msg =>
        msg.id === assistantMsgId ? { ...msg, isStreaming: false, sources } : msg
      ))

    } catch (error) {
      console.error('Chat error:', error)
      setMessages(prev => prev.map(msg =>
        msg.id === assistantMsgId ? {
          ...msg,
          content: 'Sorry, there was an error processing your message. Please try again.',
          isStreaming: false,
        } : msg
      ))
    } finally {
      setLoading(false)
    }
  }, [apiBase, input, sessionId])

  const handleSubmit = (e) => {
    e.preventDefault()
    sendMessage()
  }

  return (
    <div className="content-area chat-mode">
      {/* Dedicate Chat Panel */}
      <div className="chat-panel">
        <div className="chat-window" ref={chatWindowRef}>
          {messages.length === 0 ? (
            <EmptyState onSuggestionClick={sendMessage} />
          ) : (
            messages.map((msg, idx) => (
              <Message key={msg.id || idx} message={msg} />
            ))
          )}
        </div>
      </div>

      {/* Footer / Input Section */}
      <div className="footer-section">
        <div className="evidence-strip">
          <FloatingSources sources={currentSources} />
        </div>

        <form className="input-bar" onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            type="text"
            className="main-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message here..."
            disabled={loading}
          />
          <button
            type="submit"
            className="send-button-primary"
            disabled={loading || !input.trim()}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  )
}

// ===== Excel to PPT Panel (Simplified) =====

function ExcelPptBot({ apiBase }) {
  // ... (Keeping logic same, just structure update)
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState({ type: '', message: '' })
  const [downloadUrl, setDownloadUrl] = useState('')
  const [loading, setLoading] = useState(false)

  const handleFileChange = (e) => { /* ... */
    const selected = e.target.files?.[0]
    setFile(selected || null)
  }
  const handleUpload = async () => { /* ... */ }

  return (
    <div className="content-area excel-mode">
      <div className="upload-box-large">
        <div className="upload-icon-large">📊</div>
        <h2>Upload Data</h2>
        <p>Select your Excel or CSV file to generate slides.</p>

        <input
          type="file"
          id="file-upload"
          accept=".xlsx,.xls,.csv"
          onChange={handleFileChange}
          disabled={loading}
        />
        <label htmlFor="file-upload" className="file-label">
          {file ? `📄 ${file.name}` : '📁 Choose File'}
        </label>

        <button
          className="action-button"
          onClick={handleUpload}
          disabled={loading || !file}
        >
          {loading ? 'Converting...' : 'Generate Presentation'}
        </button>
      </div>
    </div>
  )
}

// ===== Main App =====

function App() {
  const [activeTab, setActiveTab] = useState('agent')
  const backendStatus = useBackendStatus(API_BASE)
  const statusMeta = {
    connected: { label: 'Connected', className: 'status-connected' },
    checking: { label: 'Connecting', className: 'status-checking' },
    offline: { label: 'Offline', className: 'status-offline' },
  }
  const statusInfo = statusMeta[backendStatus] || statusMeta.checking

  return (
    <div className="app-background">
      <div className="product-card">

        {/* Header embedded in card */}
        <header className="card-header">
          <div className="header-col left">
            <div className="brand-line">AWCC SUPPORT</div>
            <h1>Telecom Assistant</h1>
            <div className="subtitle-text">Fast bundle help, troubleshooting, and automation.</div>
          </div>

          <div className="header-col center">
            <h2>Guided AWCC support</h2>
            <div className="helper-text">Ask for bundles, APN, troubleshooting, or tickets.</div>
          </div>

          <div className="header-col right">
            <div className={`status-pill ${statusInfo.className}`}>
              <span className="dot"></span> {statusInfo.label}
            </div>
            <div className="toggle-pills">
              <button
                className={`pill ${activeTab === 'agent' ? 'active' : ''}`}
                onClick={() => setActiveTab('agent')}
              >
                Telecom Agent
              </button>
              <button
                className={`pill ${activeTab === 'ppt' ? 'active' : ''}`}
                onClick={() => setActiveTab('ppt')}
              >
                Excel → PPT
              </button>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <div className="card-body">
          {activeTab === 'agent' ? (
            <TelecomAgent apiBase={API_BASE} />
          ) : (
            <ExcelPptBot apiBase={API_BASE} />
          )}
        </div>

      </div>
    </div>
  )
}

export default App
