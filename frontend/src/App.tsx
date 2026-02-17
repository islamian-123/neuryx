import { useState, useRef, useEffect } from 'react'
import { Mic, Square, Settings, X, Sparkles, History, Clock, FileText } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { ModelManager } from './components/ModelManager'

function App() {
  const [isRecording, setIsRecording] = useState(false)
  const [status, setStatus] = useState('Tap to speak')
  const [liveText, setLiveText] = useState('')
  const [detectedLang, setDetectedLang] = useState<string | null>(null)
  const [showSettings, setShowSettings] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [historyItems, setHistoryItems] = useState<any[]>([])

  // Refs for batch recording
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const startRecording = async () => {
    try {
      setLiveText('')
      setDetectedLang(null)
      chunksRef.current = []

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = async () => {
        setStatus('Processing...')

        // Create Audio Blob
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' })

        // Prepare Upload
        const formData = new FormData()
        formData.append('file', audioBlob, 'recording.webm')
        formData.append('language', 'auto') // Default to Auto

        try {
          // Use the optimized backend endpoint
          const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData,
          })

          const result = await response.json()

          if (result.status === 'success') {
            setLiveText(result.full_text)
            const langName = result.language === 'ur' ? 'Urdu (Romanized)' : result.language
            setDetectedLang(langName)
            setStatus(`Done (${result.language}, ${result.duration.toFixed(1)}s)`)
          } else {
            setStatus('Error processing')
            setLiveText(`Error: ${result.message}`)
          }
        } catch (error) {
          console.error('Upload failed:', error)
          setStatus('Upload failed (Check connection)')
        } finally {
          setIsRecording(false)
          // Stop all tracks
          stream.getTracks().forEach(track => track.stop())
        }
      }

      mediaRecorder.start()
      setIsRecording(true)
      setStatus('Recording...')

    } catch (err) {
      console.error('Microphone error:', err)
      setStatus('Microphone blocked')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
  }

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }

  const fetchHistory = async () => {
    try {
      const res = await fetch('/history')
      const data = await res.json()
      setHistoryItems(data.history || [])
    } catch (e) {
      console.error("Failed to load history", e)
    }
  }

  // Load history when modal opens
  useEffect(() => {
    if (showHistory) {
      fetchHistory()
    }
  }, [showHistory])

  const loadHistoryItem = async (id: string) => {
    try {
      const res = await fetch(`/history/${id}`)
      const data = await res.json()

      setLiveText(data.full_text)
      const langName = data.language === 'ur' ? 'Urdu (Romanized)' : data.language
      setDetectedLang(langName)
      setStatus(`Loaded from History (${new Date(data.timestamp * 1000).toLocaleTimeString()})`)
      setShowHistory(false)
    } catch (e) {
      console.error("Failed to load history detail", e)
    }
  }

  return (
    <div className="app-container">
      {/* Background glow */}
      <div className={`bg-glow ${isRecording ? 'recording' : ''}`} />

      {/* Header */}
      <header className="app-header">
        <div className="header-left" />
        <div className="header-center">
          <h1 className="app-title">NEURYX</h1>
          <p className="app-subtitle">Local Neural Speech Engine</p>
        </div>
        <div className="header-right flex items-center gap-2">
          <button
            onClick={() => setShowHistory(true)}
            className="settings-btn"
            aria-label="History"
            title="History"
          >
            <History size={20} />
          </button>
          <button
            onClick={() => setShowSettings(true)}
            className="settings-btn"
            aria-label="Settings"
            title="Settings"
          >
            <Settings size={20} />
          </button>
        </div>
      </header>

      {/* Settings Modal */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="modal-overlay"
            onClick={() => setShowSettings(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={e => e.stopPropagation()}
              className="modal-content"
            >
              <div className="modal-header">
                <h2><Settings size={18} /> Settings</h2>
                <button onClick={() => setShowSettings(false)} className="modal-close"><X size={18} /></button>
              </div>
              <div className="modal-body">
                <ModelManager />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* History Modal */}
      <AnimatePresence>
        {showHistory && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="modal-overlay"
            onClick={() => setShowHistory(false)}
          >
            <motion.div
              initial={{ scale: 0.9, translateY: 20, opacity: 0 }}
              animate={{ scale: 1, translateY: 0, opacity: 1 }}
              exit={{ scale: 0.9, translateY: 20, opacity: 0 }}
              onClick={e => e.stopPropagation()}
              className="modal-content"
              style={{ maxWidth: '600px', height: '80vh', display: 'flex', flexDirection: 'column' }}
            >
              <div className="modal-header">
                <h2><History size={18} /> Memory</h2>
                <button onClick={() => setShowHistory(false)} className="modal-close"><X size={18} /></button>
              </div>
              <div className="modal-body custom-scrollbar" style={{ overflowY: 'auto', flex: 1, padding: '1rem' }}>
                {historyItems.length === 0 ? (
                  <div className="text-center opacity-50 py-10">No history found.</div>
                ) : (
                  <div className="space-y-3">
                    {historyItems.map((item) => (
                      <div
                        key={item.id}
                        onClick={() => loadHistoryItem(item.id)}
                        className="p-4 rounded-lg bg-white/5 hover:bg-white/10 cursor-pointer transition-colors border border-white/5"
                      >
                        <div className="flex justify-between items-start mb-2">
                          <div className="flex items-center gap-2 text-sm text-primary/80">
                            <Clock size={14} />
                            <span>{new Date(item.timestamp * 1000).toLocaleString()}</span>
                          </div>
                          <span className="text-xs px-2 py-1 rounded bg-black/20 text-white/50 border border-white/5">
                            {item.duration ? `${item.duration.toFixed(1)}s` : 'N/A'}
                          </span>
                        </div>
                        <div className="flex items-start gap-3">
                          <FileText size={16} className="mt-1 opacity-50 shrink-0" />
                          <p className="text-sm opacity-80 line-clamp-2 leading-relaxed">
                            {item.full_text || "No text content"}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Card */}
      <main className="main-card">

        {/* Status Indicator (previously Lang Selector area) */}
        <div className="h-12 flex items-center justify-center">
          {isRecording && (
            <div className="flex items-center space-x-2 text-primary/80 animate-pulse">
              <div className="w-2 h-2 rounded-full bg-red-500" />
              <span className="text-sm font-light tracking-wide uppercase">Listening...</span>
            </div>
          )}
        </div>

        {/* Big Mic Button */}
        <div className="mic-area">
          <motion.button
            whileHover={{ scale: 1.06 }}
            whileTap={{ scale: 0.94 }}
            onClick={toggleRecording}
            className={`mic-btn ${isRecording ? 'recording' : ''}`}
          >
            {isRecording ? (
              <>
                <div className="pulse-ring" />
                <div className="pulse-ring delay" />
                <Square size={36} fill="currentColor" />
              </>
            ) : (
              <Mic size={36} />
            )}
          </motion.button>

          <div className="status-row">
            <div className={`status-dot ${isRecording ? 'live' : ''}`} />
            <span>{status}</span>
          </div>
        </div>

        {/* Transcript Area */}
        <div className="transcript-area">
          <div className="transcript-header">
            <span className="transcript-label">Transcript</span>
            {detectedLang && (
              <span className="lang-badge flex items-center gap-1">
                <Sparkles size={12} /> {detectedLang}
              </span>
            )}
          </div>
          {/* Result Transcript */}
          <div className={`flex-1 p-6 overflow-y-auto custom-scrollbar`}>
            {liveText ? (
              <p className={`text-2xl leading-relaxed ${isRecording ? 'opacity-50' : ''} text-left`}>
                {liveText}
              </p>
            ) : (
              <p className="transcript-placeholder">
                {isRecording ? 'Recording linked to Batch Engine...' : 'Press microphone to record'}
              </p>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
