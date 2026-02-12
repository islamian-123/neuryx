import { useState, useRef, useEffect } from 'react'
import { Mic, Square, Radio, FileText, Activity, Settings, X, Download, Languages } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { ModelManager } from './components/ModelManager'

type Mode = 'batch' | 'stream'

interface TranscriptionResult {
  file: string;
  transcript: string;
  timestamp: string;
}

const LANGUAGES = [
  { code: 'en', name: 'English', prompt: '' },
  { code: 'ur', name: 'Urdu', prompt: '' },
  { code: 'ur', name: 'Roman Urdu', prompt: 'Hum roman urdu mein likh rahe hain. Transcribe in Roman Urdu.' },
]

function App() {
  const [mode, setMode] = useState<Mode>('batch')
  const [isRecording, setIsRecording] = useState(false)
  const [status, setStatus] = useState('Ready')
  const [liveText, setLiveText] = useState('')
  const [history, setHistory] = useState<TranscriptionResult[]>([])
  const [showSettings, setShowSettings] = useState(false)
  const [selectedLang, setSelectedLang] = useState(LANGUAGES[0])

  // Refs for Streaming
  const socketRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)

  const startBatchRecording = async () => {
    try {
      setStatus('Recording on Server...')
      const res = await fetch('/record/start', { method: 'POST' })
      const data = await res.json()
      if (data.status === 'started') {
        setIsRecording(true)
      } else {
        setStatus(`Error: ${data.status}`)
      }
    } catch (e) {
      console.error(e)
      setStatus('Connection Failed')
    }
  }

  const stopBatchRecording = async () => {
    try {
      setStatus('Processing...')
      const res = await fetch('/record/stop', { method: 'POST' })
      const data = await res.json()
      if (data.status === 'success') {
        setHistory(prev => [{
          file: data.file,
          transcript: data.transcript,
          timestamp: new Date().toLocaleTimeString()
        }, ...prev])
        setStatus('Ready')
      } else {
        setStatus(`Error: ${data.message || 'Unknown'}`)
      }
      setIsRecording(false)
    } catch (e) {
      console.error(e)
      setStatus('Error stopping')
      setIsRecording(false)
    }
  }

  const startStreaming = async () => {
    try {
      setStatus('Connecting to Stream...')

      // Pass language and prompt in URL
      const params = new URLSearchParams({
        language: selectedLang.code,
        prompt: selectedLang.prompt
      })
      const socket = new WebSocket(`ws://localhost:8000/stream?${params.toString()}`)
      socket.binaryType = 'arraybuffer'

      socket.onopen = async () => {
        setStatus('Listening...')
        setIsRecording(true)
        setLiveText('')

        try {
          // Setup Audio with explicit downsampling
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
          mediaStreamRef.current = stream

          // Create Context - browser will allow this sample rate
          const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)()
          audioContextRef.current = audioCtx

          const source = audioCtx.createMediaStreamSource(stream)

          // Create a second context strictly for 16kHz if needed, but easier is to just resample manually 
          // or rely on the backend. But since we identified sample rate as issue:
          // We will use a ScriptProcessor and downsample if needed, 
          // OR usually WebSocket expects raw PCM. 
          // Simple fix: OfflineAudioContext or just simple decimation if ratio is integer.

          // BETTER FIX: use AudioContext with sampleRate: 16000 options if browser supports it
          // OR let backend handle ANY sample rate? Backend relies on faster-whisper which expects 16k usually.

          // Let's try creating a 16kHz context specifically. 
          // Some browsers might not support it, but most modern ones do or resample automatically.
          // BUT `getUserMedia` stream tracks conform to hardware.
          // So we need: Source(48k) -> AudioCtx(48k) -> ScriptProcessor -> Downsample -> Send.

          // Actually, let's try a simpler approach first:
          // Just set the AudioContext to 16000. Chrome/Firefox usually handle the resampling from Mic(48k) to Ctx(16k) automatically.
          // If the previous attempt failed, maybe it was because we sent FLOAT32 but didn't verify backend handling?
          // Backend expects float32 in numpy.

          // Let's stick to 16k context property.

        } catch (err) {
          console.error("Error setting up audio", err);
          setStatus("Audio Error");
          return;
        }

        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 })
        audioContextRef.current = audioCtx
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        mediaStreamRef.current = stream
        const source = audioCtx.createMediaStreamSource(stream)

        // 4096 frames @ 16kHz = ~0.25s
        const processor = audioCtx.createScriptProcessor(4096, 1, 1)

        processor.onaudioprocess = (e) => {
          if (socket.readyState === WebSocket.OPEN) {
            const inputData = e.inputBuffer.getChannelData(0)
            // Send as float32 bytes
            socket.send(inputData.buffer)
          }
        }

        source.connect(processor)
        processor.connect(audioCtx.destination)
        processorRef.current = processor
      }

      socket.onmessage = (event) => {
        setLiveText(event.data)
      }

      socket.onclose = () => {
        setIsRecording(false)
        setStatus('Ready')
        cleanupAudio()
      }

      socketRef.current = socket

    } catch (e) {
      console.error(e)
      setStatus('Microphone Error')
    }
  }

  const stopStreaming = () => {
    if (socketRef.current) {
      socketRef.current.close()
    }
    cleanupAudio()
    setIsRecording(false)
    setStatus('Ready')
    if (liveText) {
      setHistory(prev => [{
        file: 'Stream Session',
        transcript: liveText,
        timestamp: new Date().toLocaleTimeString()
      }, ...prev])
    }
  }

  const cleanupAudio = () => {
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
  }

  const toggleRecording = () => {
    if (isRecording) {
      if (mode === 'batch') stopBatchRecording()
      else stopStreaming()
    } else {
      if (mode === 'batch') startBatchRecording()
      else startStreaming()
    }
  }

  return (
    <div className="min-h-screen p-8 flex flex-col items-center relative">
      <header className="mb-12 text-center relative z-10 w-full max-w-2xl flex justify-between items-end">
        <div className="flex-1"></div>
        <div className="text-center">
          <h1 className="text-5xl font-bold tracking-tighter mb-2">
            <span className="title-gradient">NEURYX</span>
          </h1>
          <p className="text-text-secondary">Local Neural Speech Engine</p>
        </div>
        <div className="flex-1 flex justify-end">
          <button
            onClick={() => setShowSettings(true)}
            className="p-2 bg-slate-800 rounded-full hover:bg-slate-700 hover:text-white transition text-slate-400"
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
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setShowSettings(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={e => e.stopPropagation()}
              className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-lg shadow-2xl overflow-hidden"
            >
              <div className="p-4 border-b border-slate-800 flex justify-between items-center">
                <h2 className="text-xl font-bold flex items-center gap-2"><Settings size={20} /> Settings</h2>
                <button onClick={() => setShowSettings(false)} className="hover:bg-slate-800 p-1 rounded-full"><X size={20} /></button>
              </div>
              <div className="p-6">
                <ModelManager />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="glass-panel w-full max-w-2xl flex flex-col gap-6">
        {/* Controls */}
        <div className="flex flex-col md:flex-row justify-between items-center p-2 bg-slate-800/50 rounded-lg gap-4">
          <div className="flex gap-2">
            <button
              onClick={() => setMode('batch')}
              className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors ${mode === 'batch' ? 'bg-blue-600 text-white' : 'hover:bg-slate-700'}`}
            >
              <FileText size={18} /> Batch
            </button>
            <button
              onClick={() => setMode('stream')}
              className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors ${mode === 'stream' ? 'bg-purple-600 text-white' : 'hover:bg-slate-700'}`}
            >
              <Activity size={18} /> Stream
            </button>
          </div>

          {/* Language Selector */}
          <div className="flex items-center gap-2 bg-slate-900/50 p-1 rounded">
            {LANGUAGES.map(lang => (
              <button
                key={lang.name}
                onClick={() => setSelectedLang(lang)}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${selectedLang.name === lang.name ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-slate-200'}`}
              >
                {lang.name}
              </button>
            ))}
          </div>
        </div>

        {/* Main Action */}
        <div className="flex flex-col items-center justify-center py-6">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleRecording}
            className={`
              w-24 h-24 rounded-full flex items-center justify-center shadow-lg transition-all mb-4
              ${isRecording
                ? 'bg-red-500/20 text-red-500 border-2 border-red-500 shadow-red-500/20'
                : 'bg-blue-500 text-white shadow-blue-500/30 hover:bg-blue-600'}
            `}
          >
            {isRecording ? <Square size={32} fill="currentColor" /> : <Mic size={32} />}
          </motion.button>

          <div className="flex items-center gap-2 text-sm text-text-secondary h-6">
            <div className={`w-2 h-2 rounded-full ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-slate-500'}`} />
            {status}
          </div>
        </div>

        {/* Live Transcript Area (Stream Mode) */}
        <AnimatePresence>
          {mode === 'stream' && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-slate-900/50 p-4 rounded-xl min-h-[100px] border border-slate-700/50"
            >
              <h3 className="text-xs font-semibold text-purple-400 mb-2 uppercase tracking-wider flex justify-between">
                Live Transcript
                <span className="text-slate-500 normal-case bg-slate-800 px-2 rounded">{selectedLang.name}</span>
              </h3>
              <p className="text-lg leading-relaxed text-slate-200 mt-2">
                {liveText || <span className="text-slate-600 italic">Listening...</span>}
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* History */}
      <div className="w-full max-w-2xl mt-12 pb-12">
        <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
          History <span className="text-xs bg-slate-800 px-2 py-0.5 rounded-full text-slate-400">{history.length}</span>
        </h2>

        <div className="space-y-4">
          <AnimatePresence>
            {history.map((item, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-slate-800/40 border border-slate-700/50 p-4 rounded-lg flex flex-col gap-2 hover:bg-slate-800/60 transition-colors"
              >
                <div className="flex justify-between items-start">
                  <span className="text-xs font-mono text-slate-500">{item.timestamp}</span>
                  <span className="text-xs bg-slate-700 text-slate-300 px-2 rounded">{item.file.includes('recording') ? 'Batch' : 'Stream'}</span>
                </div>
                <p className="text-slate-300">{item.transcript}</p>
              </motion.div>
            ))}
          </AnimatePresence>

          {history.length === 0 && (
            <div className="text-center py-12 text-slate-600 border border-dashed border-slate-800 rounded-xl">
              No recordings yet
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
