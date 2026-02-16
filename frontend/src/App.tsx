import { useState, useRef } from 'react'
import { Mic, Square, Settings, X } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { ModelManager } from './components/ModelManager'

const LANGUAGES = [
  { code: 'en', name: 'English', prompt: 'Hello. Technical terms. Aerospace. Engineering.' },
  { code: 'ur', name: 'اردو', prompt: 'اردو زبان۔ پاکستان۔ انسٹیٹیوٹ آف اسپیس ٹیکنالوجی۔ ایرو اسپیس انجینئرنگ۔' },
  { code: '', name: 'Roman Urdu', prompt: 'Roman Urdu. English letters only. Hindi Urdu mix. Kya haal hai. Aerospace Engineering.' },
]

function App() {
  const [isRecording, setIsRecording] = useState(false)
  const [status, setStatus] = useState('Tap to speak')
  const [liveText, setLiveText] = useState('')
  const [selectedLang, setSelectedLang] = useState(LANGUAGES[0])
  const [showSettings, setShowSettings] = useState(false)

  // Refs for batch recording
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const startRecording = async () => {
    try {
      setLiveText('')
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
        formData.append('language', selectedLang.code)

        try {
          const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData,
          })

          const result = await response.json()

          if (result.status === 'success') {
            setLiveText(result.full_text)
            setStatus(`Done (${result.language}, ${result.duration.toFixed(1)}s)`)
          } else {
            setStatus('Error processing')
            setLiveText(`Error: ${result.message}`)
          }
        } catch (error) {
          console.error('Upload failed:', error)
          setStatus('Upload failed')
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
      // UI update happens in onstop
    }
  }

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
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
        <div className="header-right">
          <button
            onClick={() => setShowSettings(true)}
            className="settings-btn"
            aria-label="Settings"
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

      {/* Main Card */}
      <main className="main-card">

        {/* Language Selector */}
        <div className="lang-selector">
          {LANGUAGES.map(lang => (
            <button
              key={lang.name}
              onClick={() => { if (!isRecording) setSelectedLang(lang) }}
              className={`lang-btn ${selectedLang.name === lang.name ? 'active' : ''}`}
              disabled={isRecording}
            >
              {lang.name}
            </button>
          ))}
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
            <span className="lang-badge">{selectedLang.name}</span>
          </div>
          {/* Result Transcript */}
          <div className={`flex-1 p-6 overflow-y-auto custom-scrollbar ${selectedLang.code === 'ur' ? 'font-urdu' : ''}`}>
            {liveText ? (
              <p className={`text-2xl leading-relaxed ${isRecording ? 'opacity-50' : ''} ${selectedLang.code === 'ur' ? 'text-right' : 'text-left'}`}>
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
