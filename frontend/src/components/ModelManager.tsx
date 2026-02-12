import { useState, useEffect } from 'react'
import { Download, Trash2, Check, RefreshCw, AlertCircle } from 'lucide-react'

interface ModelStatus {
    [key: string]: boolean;
}

export function ModelManager() {
    const [models, setModels] = useState<ModelStatus>({})
    const [loading, setLoading] = useState<string | null>(null)
    const [error, setError] = useState<string | null>(null)

    const fetchModels = async () => {
        try {
            const res = await fetch('/models')
            const data = await res.json()
            setModels(data)
        } catch (e) {
            setError('Failed to load models')
        }
    }

    useEffect(() => {
        fetchModels()
    }, [])

    const handleDownload = async (model: string) => {
        if (loading) return
        setLoading(model)
        setError(null)
        try {
            const res = await fetch(`/models/${model}/download`, { method: 'POST' })
            if (!res.ok) throw new Error('Download failed')
            // Poll or just wait? It's background task.
            // For now, let's just simulate success or wait a bit then refresh
            // ideally we'd have a status endpoint or WebSocket
            setTimeout(fetchModels, 2000)
        } catch (e) {
            setError(`Failed to download ${model}`)
        } finally {
            setLoading(null)
        }
    }

    const handleDelete = async (model: string) => {
        if (!confirm(`Delete ${model} model?`)) return
        try {
            const res = await fetch(`/models/${model}`, { method: 'DELETE' })
            if (!res.ok) throw new Error('Delete failed')
            fetchModels()
        } catch (e) {
            setError(`Failed to delete ${model}`)
        }
    }

    return (
        <div className="p-4 bg-slate-800 rounded-lg border border-slate-700">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                Offline Models
                <button onClick={fetchModels} className="text-slate-400 hover:text-white transition-colors">
                    <RefreshCw size={16} />
                </button>
            </h3>

            {error && (
                <div className="bg-red-500/10 text-red-500 p-3 rounded mb-4 flex items-center gap-2 text-sm">
                    <AlertCircle size={16} /> {error}
                </div>
            )}

            <div className="space-y-3">
                {Object.entries(models).map(([name, installed]) => (
                    <div key={name} className="flex items-center justify-between p-3 bg-slate-900/50 rounded hover:bg-slate-900 transition-colors">
                        <div className="flex items-center gap-3">
                            <span className="font-medium capitalize text-slate-200">{name}</span>
                            {installed && <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded flex items-center gap-1"><Check size={12} /> Installed</span>}
                        </div>

                        <div className="flex items-center gap-2">
                            {installed ? (
                                <button
                                    onClick={() => handleDelete(name)}
                                    className="p-2 text-slate-500 hover:text-red-400 transition-colors rounded-full hover:bg-slate-800"
                                    title="Delete Model"
                                >
                                    <Trash2 size={18} />
                                </button>
                            ) : (
                                <button
                                    onClick={() => handleDownload(name)}
                                    disabled={loading === name}
                                    className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm font-medium transition-colors ${loading === name ? 'bg-slate-700 text-slate-400' : 'bg-blue-600 text-white hover:bg-blue-500'}`}
                                >
                                    {loading === name ? <RefreshCw size={14} className="animate-spin" /> : <Download size={14} />}
                                    Download
                                </button>
                            )}
                        </div>
                    </div>
                ))}
            </div>
            <p className="text-xs text-slate-500 mt-4 leading-relaxed">
                Downloaded models are stored locally (`backend/models`). <br />
                Larger models (medium/large) require more RAM and may be slower.
            </p>
        </div>
    )
}
