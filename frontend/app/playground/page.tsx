'use client'

import { useState } from 'react'
import { Upload, Mic, Play, ArrowLeft, FileAudio, Check, AlertCircle, Loader2, Zap, Activity } from 'lucide-react'
import Link from 'next/link'
import { cn } from '@/lib/utils'
import { api } from '@/lib/api'

export default function PlaygroundPage() {
    const [file, setFile] = useState<File | null>(null)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState<{ text: string; confidence: number; latency_ms: number; model_id: string } | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [dragActive, setDragActive] = useState(false)

    // Mock models for now, could fetch from API
    const models = [
        { id: 'whisper-base-v1', name: 'Whisper Base (Production)' },
        { id: 'whisper-tiny-demo', name: 'Whisper Tiny (Fast)' },
        { id: 'custom-ft-v2', name: 'Custom Fine-Tuned V2' }
    ]
    const [selectedModel, setSelectedModel] = useState(models[0].id)

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true)
        } else if (e.type === 'dragleave') {
            setDragActive(false)
        }
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            validateAndSetFile(e.dataTransfer.files[0])
        }
    }

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        e.preventDefault()
        if (e.target.files && e.target.files[0]) {
            validateAndSetFile(e.target.files[0])
        }
    }

    const validateAndSetFile = (f: File) => {
        if (!f.type.startsWith('audio/')) {
            setError('Please upload an audio file (MP3, WAV, etc.)')
            return
        }
        setFile(f)
        setError(null)
        setResult(null)
    }

    const handleTranscribe = async () => {
        if (!file) return

        setLoading(true)
        setError(null)
        try {
            const res = await api.transcribe(file, selectedModel)
            setResult(res)
        } catch (err: any) {
            setError(err.message || 'Transcription failed')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen p-6">
            <header className="mb-8 flex items-center gap-4">
                <Link href="/" className="p-2 rounded-lg bg-background-secondary hover:bg-background-tertiary transition-colors">
                    <ArrowLeft className="w-5 h-5" />
                </Link>
                <div>
                    <h1 className="text-2xl font-bold gradient-text">Inference Playground</h1>
                    <p className="text-foreground-muted">Test your models interactively</p>
                </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
                {/* Input Section */}
                <div className="space-y-6">
                    <div className="glass rounded-xl p-6">
                        <h2 className="text-lg font-semibold mb-4">1. Select Model</h2>
                        <select
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            className="w-full p-3 rounded-lg bg-background-tertiary border border-border focus:ring-2 ring-primary outline-none"
                        >
                            {models.map(m => (
                                <option key={m.id} value={m.id}>{m.name}</option>
                            ))}
                        </select>
                    </div>

                    <div className="glass rounded-xl p-6">
                        <h2 className="text-lg font-semibold mb-4">2. Upload Audio</h2>
                        <div
                            className={cn(
                                "border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer",
                                dragActive ? "border-primary bg-primary/5" : "border-background-tertiary hover:border-primary/50",
                                file ? "border-green-500/50 bg-green-500/5" : ""
                            )}
                            onDragEnter={handleDrag}
                            onDragLeave={handleDrag}
                            onDragOver={handleDrag}
                            onDrop={handleDrop}
                            onClick={() => document.getElementById('audio-upload')?.click()}
                        >
                            <input
                                id="audio-upload"
                                type="file"
                                accept="audio/*"
                                className="hidden"
                                onChange={handleChange}
                            />

                            {file ? (
                                <div className="flex flex-col items-center gap-2">
                                    <FileAudio className="w-10 h-10 text-green-500" />
                                    <p className="font-medium">{file.name}</p>
                                    <p className="text-sm text-foreground-muted">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation()
                                            setFile(null)
                                            setResult(null)
                                        }}
                                        className="mt-2 text-xs text-red-400 hover:text-red-300"
                                    >
                                        Remove
                                    </button>
                                </div>
                            ) : (
                                <div className="flex flex-col items-center gap-2">
                                    <Upload className="w-10 h-10 text-foreground-muted" />
                                    <p className="font-medium">Click to upload or drag & drop</p>
                                    <p className="text-sm text-foreground-muted">MP3, WAV, M4A up to 10MB</p>
                                </div>
                            )}
                        </div>
                    </div>

                    <button
                        onClick={handleTranscribe}
                        disabled={!file || loading}
                        className={cn(
                            "w-full py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all",
                            !file
                                ? "bg-background-tertiary text-foreground-muted cursor-not-allowed"
                                : loading
                                    ? "bg-primary/80 cursor-wait"
                                    : "bg-primary hover:bg-primary/90 hover:scale-[1.02]"
                        )}
                    >
                        {loading ? (
                            <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                Transcribing...
                            </>
                        ) : (
                            <>
                                <Zap className="w-5 h-5" />
                                Run Inference
                            </>
                        )}
                    </button>

                    {error && (
                        <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center gap-2 text-red-400">
                            <AlertCircle className="w-4 h-4" />
                            {error}
                        </div>
                    )}
                </div>

                {/* Output Section */}
                <div className="space-y-6">
                    <div className={cn(
                        "glass rounded-xl p-6 h-full min-h-[400px] flex flex-col",
                        !result && "justify-center items-center text-foreground-muted opacity-50"
                    )}>
                        {!result ? (
                            <div className="text-center">
                                <Activity className="w-12 h-12 mx-auto mb-4 opacity-20" />
                                <p>Results will appear here</p>
                            </div>
                        ) : (
                            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-6">
                                <div>
                                    <h3 className="text-sm font-medium text-foreground-muted mb-2 uppercase tracking-wider">Transcription</h3>
                                    <div className="p-4 rounded-lg bg-background-tertiary border border-border text-lg leading-relaxed">
                                        {result.text}
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                                        <p className="text-xs text-blue-400 mb-1">Confidence</p>
                                        <p className="text-2xl font-mono text-blue-300">{(result.confidence * 100).toFixed(1)}%</p>
                                    </div>
                                    <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
                                        <p className="text-xs text-purple-400 mb-1">Latency</p>
                                        <p className="text-2xl font-mono text-purple-300">{result.latency_ms} ms</p>
                                    </div>
                                </div>

                                <div>
                                    <h3 className="text-sm font-medium text-foreground-muted mb-2 uppercase tracking-wider">Debug Info</h3>
                                    <div className="p-4 rounded-lg bg-black/30 font-mono text-xs text-green-400 overflow-x-auto">
                                        {JSON.stringify({
                                            model: result.model_id,
                                            file: file?.name,
                                            size: file?.size,
                                            timestamp: new Date().toISOString()
                                        }, null, 2)}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
