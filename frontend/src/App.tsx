import { useState, useRef, useCallback } from 'react'
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

  // Refs for streaming
  const socketRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)

  const cleanupAudio = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current = null
    }
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop())
      mediaStreamRef.current = null
    }
  }, [])

  const startRecording = async () => {
    try {
      setStatus('Connecting...')
      setLiveText('')

      // Build WebSocket URL using Vite proxy (relative path)
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const params = new URLSearchParams({
        language: selectedLang.code,
        ...(selectedLang.prompt ? { prompt: selectedLang.prompt } : {})
      })
      const wsUrl = `${wsProtocol}//${window.location.host}/stream?${params.toString()}`

      const socket = new WebSocket(wsUrl)
      socket.binaryType = 'arraybuffer'
      socketRef.current = socket

      socket.onopen = async () => {
        setStatus('Listening...')
        setIsRecording(true)

        try {
          // Create AudioContext at 16kHz for Whisper
          const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 })
          audioContextRef.current = audioCtx

          const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 1,
              echoCancellation: true,
              noiseSuppression: true,
            }
          })
          mediaStreamRef.current = stream

          const source = audioCtx.createMediaStreamSource(stream)

          // 4096 frames @ 16kHz ≈ 0.256s per chunk
          const processor = audioCtx.createScriptProcessor(4096, 1, 1)
          processorRef.current = processor

          processor.onaudioprocess = (e) => {
            if (socket.readyState === WebSocket.OPEN) {
              const inputData = e.inputBuffer.getChannelData(0)
              // Send raw float32 PCM bytes
              socket.send(inputData.buffer)
            }
          }

          source.connect(processor)
          processor.connect(audioCtx.destination)
        } catch (err) {
          console.error('Microphone error:', err)
          setStatus('Microphone blocked')
          socket.close()
        }
      }

      socket.onmessage = (event) => {
        setLiveText(event.data)
      }

      socket.onerror = () => {
        setStatus('Connection failed')
        setIsRecording(false)
        cleanupAudio()
      }

      socket.onclose = () => {
        setIsRecording(false)
        setStatus('Tap to speak')
        cleanupAudio()
      }

    } catch (e) {
      console.error(e)
      setStatus('Error')
    }
  }

  const stopRecording = () => {
    if (socketRef.current) {
      socketRef.current.close()
      socketRef.current = null
    }
    cleanupAudio()
    setIsRecording(false)
    setStatus('Tap to speak')
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
          {/* Live Transcript */}
          <div className={`flex-1 p-6 overflow-y-auto custom-scrollbar ${selectedLang.code === 'ur' ? 'font-urdu' : ''}`}>
            {liveText ? (
              <p className={`text-2xl leading-relaxed ${isRecording ? 'animate-pulse-subtle' : ''} ${selectedLang.code === 'ur' ? 'text-right' : 'text-left'}`}>
                {liveText}
              </p>
            ) : (
              <p className="transcript-placeholder">
                {isRecording ? 'Listening...' : 'Press the microphone to start'}
              </p>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
