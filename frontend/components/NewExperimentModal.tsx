'use client'

import { useState } from 'react'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'

interface NewExperimentModalProps {
    isOpen: boolean
    onClose: () => void
    onCreate: (config: ExperimentFormData) => void
}

interface ExperimentFormData {
    name: string
    model_architecture: string
    model_variant: string
    batch_size: number
    learning_rate: number
    max_epochs: number
    mixed_precision: boolean
    gradient_accumulation: number
    tags: string[]
}

const MODEL_OPTIONS = {
    whisper: {
        variants: ['tiny', 'base', 'small', 'medium'],
        description: 'OpenAI Whisper - Multilingual ASR',
    },
    wav2vec2: {
        variants: ['base', 'large', 'large-lv60k'],
        description: 'Facebook Wav2Vec 2.0 - Self-supervised learning',
    },
}

export function NewExperimentModal({ isOpen, onClose, onCreate }: NewExperimentModalProps) {
    const [formData, setFormData] = useState<ExperimentFormData>({
        name: '',
        model_architecture: 'whisper',
        model_variant: 'tiny',
        batch_size: 8,
        learning_rate: 0.0001,
        max_epochs: 5,
        mixed_precision: true,
        gradient_accumulation: 4,
        tags: [],
    })

    const [tagInput, setTagInput] = useState('')

    if (!isOpen) return null

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        if (!formData.name.trim()) return

        onCreate(formData)
        onClose()
        // Reset form
        setFormData({
            name: '',
            model_architecture: 'whisper',
            model_variant: 'tiny',
            batch_size: 8,
            learning_rate: 0.0001,
            max_epochs: 5,
            mixed_precision: true,
            gradient_accumulation: 4,
            tags: [],
        })
    }

    const addTag = () => {
        if (tagInput.trim() && !formData.tags.includes(tagInput.trim())) {
            setFormData({ ...formData, tags: [...formData.tags, tagInput.trim()] })
            setTagInput('')
        }
    }

    const removeTag = (tag: string) => {
        setFormData({ ...formData, tags: formData.tags.filter((t) => t !== tag) })
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="relative z-10 w-full max-w-lg glass rounded-xl p-6 m-4 max-h-[90vh] overflow-y-auto">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-bold">New Experiment</h2>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
                    >
                        <X className="w-5 h-5 text-foreground-muted" />
                    </button>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="space-y-5">
                    {/* Name */}
                    <div>
                        <label className="block text-sm font-medium mb-2">
                            Experiment Name
                        </label>
                        <input
                            type="text"
                            value={formData.name}
                            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                            placeholder="whisper_tiny_librispeech"
                            className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent focus:ring-1 focus:ring-accent outline-none transition-colors"
                            required
                        />
                    </div>

                    {/* Model Selection */}
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium mb-2">
                                Architecture
                            </label>
                            <select
                                value={formData.model_architecture}
                                onChange={(e) =>
                                    setFormData({
                                        ...formData,
                                        model_architecture: e.target.value,
                                        model_variant: MODEL_OPTIONS[e.target.value as keyof typeof MODEL_OPTIONS].variants[0],
                                    })
                                }
                                className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none"
                            >
                                <option value="whisper">Whisper</option>
                                <option value="wav2vec2">Wav2Vec2</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-2">Variant</label>
                            <select
                                value={formData.model_variant}
                                onChange={(e) =>
                                    setFormData({ ...formData, model_variant: e.target.value })
                                }
                                className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none"
                            >
                                {MODEL_OPTIONS[
                                    formData.model_architecture as keyof typeof MODEL_OPTIONS
                                ].variants.map((v) => (
                                    <option key={v} value={v}>
                                        {v}
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>

                    {/* Training Params */}
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium mb-2">
                                Batch Size
                            </label>
                            <input
                                type="number"
                                value={formData.batch_size}
                                onChange={(e) =>
                                    setFormData({ ...formData, batch_size: parseInt(e.target.value) })
                                }
                                min={1}
                                max={64}
                                className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-2">
                                Learning Rate
                            </label>
                            <input
                                type="number"
                                value={formData.learning_rate}
                                onChange={(e) =>
                                    setFormData({
                                        ...formData,
                                        learning_rate: parseFloat(e.target.value),
                                    })
                                }
                                step={0.0001}
                                min={0.00001}
                                className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none"
                            />
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium mb-2">
                                Max Epochs
                            </label>
                            <input
                                type="number"
                                value={formData.max_epochs}
                                onChange={(e) =>
                                    setFormData({ ...formData, max_epochs: parseInt(e.target.value) })
                                }
                                min={1}
                                max={100}
                                className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-2">
                                Grad Accumulation
                            </label>
                            <input
                                type="number"
                                value={formData.gradient_accumulation}
                                onChange={(e) =>
                                    setFormData({
                                        ...formData,
                                        gradient_accumulation: parseInt(e.target.value),
                                    })
                                }
                                min={1}
                                max={16}
                                className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none"
                            />
                        </div>
                    </div>

                    {/* Mixed Precision */}
                    <div className="flex items-center gap-3">
                        <input
                            type="checkbox"
                            id="mixed_precision"
                            checked={formData.mixed_precision}
                            onChange={(e) =>
                                setFormData({ ...formData, mixed_precision: e.target.checked })
                            }
                            className="w-4 h-4 rounded bg-background-tertiary border-background-tertiary"
                        />
                        <label htmlFor="mixed_precision" className="text-sm">
                            Enable Mixed Precision (FP16)
                        </label>
                    </div>

                    {/* Tags */}
                    <div>
                        <label className="block text-sm font-medium mb-2">Tags</label>
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={tagInput}
                                onChange={(e) => setTagInput(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
                                placeholder="Add tag..."
                                className="flex-1 px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none"
                            />
                            <button
                                type="button"
                                onClick={addTag}
                                className="px-4 py-2.5 rounded-lg bg-background-tertiary hover:bg-background text-foreground-muted hover:text-foreground transition-colors"
                            >
                                Add
                            </button>
                        </div>
                        {formData.tags.length > 0 && (
                            <div className="flex flex-wrap gap-2 mt-3">
                                {formData.tags.map((tag) => (
                                    <span
                                        key={tag}
                                        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-accent/20 text-accent text-xs"
                                    >
                                        {tag}
                                        <button
                                            type="button"
                                            onClick={() => removeTag(tag)}
                                            className="hover:text-white"
                                        >
                                            ×
                                        </button>
                                    </span>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Actions */}
                    <div className="flex gap-3 pt-4 border-t border-background-tertiary">
                        <button
                            type="button"
                            onClick={onClose}
                            className="flex-1 px-4 py-2.5 rounded-lg bg-background-tertiary hover:bg-background text-foreground-muted hover:text-foreground transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            className="flex-1 px-4 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white font-medium transition-colors"
                        >
                            Create Experiment
                        </button>
                    </div>
                </form>
            </div>
        </div>
    )
}
