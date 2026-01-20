'use client'

import { useState } from 'react'
import { Play, Pause, Square, MoreVertical, Clock, CheckCircle, XCircle, Loader2, Trash2, Copy, FileText } from 'lucide-react'
import { cn } from '@/lib/utils'
import { type Experiment } from '@/lib/api'

interface ExperimentListProps {
    experiments: Experiment[]
    loading: boolean
    onStart: (id: string) => void
    onStop: (id: string) => void
    onDelete: (id: string) => void
    onNewExperiment: () => void
}

const statusConfig = {
    running: {
        icon: Loader2,
        color: 'text-blue-400',
        bg: 'bg-blue-400/10',
        label: 'Running',
    },
    completed: {
        icon: CheckCircle,
        color: 'text-green-400',
        bg: 'bg-green-400/10',
        label: 'Completed',
    },
    failed: {
        icon: XCircle,
        color: 'text-red-400',
        bg: 'bg-red-400/10',
        label: 'Failed',
    },
    pending: {
        icon: Clock,
        color: 'text-yellow-400',
        bg: 'bg-yellow-400/10',
        label: 'Pending',
    },
    stopped: {
        icon: Square,
        color: 'text-gray-400',
        bg: 'bg-gray-400/10',
        label: 'Stopped',
    },
}

function formatDate(dateString: string): string {
    const date = new Date(dateString)
    const now = new Date()
    const diff = now.getTime() - date.getTime()

    const hours = Math.floor(diff / (1000 * 60 * 60))
    const days = Math.floor(hours / 24)

    if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`
    if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`
    return 'Just now'
}

function ExperimentRow({
    experiment,
    onStart,
    onStop,
    onDelete,
}: {
    experiment: Experiment
    onStart: (id: string) => void
    onStop: (id: string) => void
    onDelete: (id: string) => void
}) {
    const [menuOpen, setMenuOpen] = useState(false)
    const config = statusConfig[experiment.status] || statusConfig.pending
    const StatusIcon = config.icon

    // Model name from experiment name
    const modelName = experiment.name.includes('whisper')
        ? `Whisper ${experiment.name.includes('tiny') ? 'Tiny' : experiment.name.includes('base') ? 'Base' : 'Model'}`
        : experiment.name.includes('wav2vec')
            ? 'Wav2Vec2 Base'
            : 'Unknown Model'

    return (
        <tr className="border-b border-background-tertiary hover:bg-background-secondary/50 transition-colors">
            <td className="py-4 px-4">
                <div className="flex items-center gap-3">
                    <div className={cn('p-2 rounded-lg', config.bg)}>
                        <StatusIcon
                            className={cn(
                                'w-4 h-4',
                                config.color,
                                experiment.status === 'running' && 'animate-spin'
                            )}
                        />
                    </div>
                    <div>
                        <p className="font-medium">{experiment.name}</p>
                        <p className="text-sm text-foreground-muted">{modelName}</p>
                    </div>
                </div>
            </td>
            <td className="py-4 px-4">
                <span
                    className={cn(
                        'inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium',
                        config.bg,
                        config.color
                    )}
                >
                    {config.label}
                </span>
            </td>
            <td className="py-4 px-4">
                <div className="w-full max-w-[120px]">
                    <div className="flex items-center justify-between text-sm mb-1">
                        <span className="text-foreground-muted">Progress</span>
                        <span className="font-medium">{experiment.progress}%</span>
                    </div>
                    <div className="h-1.5 bg-background-tertiary rounded-full overflow-hidden">
                        <div
                            className={cn(
                                'h-full rounded-full transition-all duration-500',
                                experiment.status === 'failed' ? 'bg-red-500' :
                                    experiment.status === 'completed' ? 'bg-green-500' : 'bg-blue-500'
                            )}
                            style={{ width: `${experiment.progress}%` }}
                        />
                    </div>
                </div>
            </td>
            <td className="py-4 px-4">
                {experiment.wer ? (
                    <span className="font-mono font-medium">{experiment.wer}%</span>
                ) : (
                    <span className="text-foreground-muted">-</span>
                )}
            </td>
            <td className="py-4 px-4 text-foreground-muted">{formatDate(experiment.created_at)}</td>
            <td className="py-4 px-4 text-foreground-muted font-mono">
                {experiment.current_epoch}/{experiment.total_epochs}
            </td>
            <td className="py-4 px-4">
                <div className="flex items-center gap-2 relative">
                    {experiment.status === 'running' ? (
                        <button
                            onClick={() => onStop(experiment.id)}
                            className="p-1.5 rounded-lg hover:bg-red-500/20 hover:text-red-400 transition-colors"
                            title="Stop"
                        >
                            <Square className="w-4 h-4 text-foreground-muted hover:text-red-400" />
                        </button>
                    ) : (experiment.status === 'pending' || experiment.status === 'stopped') ? (
                        <button
                            onClick={() => onStart(experiment.id)}
                            className="p-1.5 rounded-lg hover:bg-green-500/20 hover:text-green-400 transition-colors"
                            title="Start"
                        >
                            <Play className="w-4 h-4 text-foreground-muted hover:text-green-400" />
                        </button>
                    ) : null}

                    <div className="relative">
                        <button
                            onClick={() => setMenuOpen(!menuOpen)}
                            className="p-1.5 rounded-lg hover:bg-background-tertiary transition-colors"
                        >
                            <MoreVertical className="w-4 h-4 text-foreground-muted" />
                        </button>

                        {menuOpen && (
                            <>
                                <div
                                    className="fixed inset-0 z-10"
                                    onClick={() => setMenuOpen(false)}
                                />
                                <div className="absolute right-0 bottom-full mb-2 z-20 w-40 glass rounded-lg py-1 shadow-xl border border-background-tertiary">
                                    <button
                                        onClick={() => {
                                            setMenuOpen(false)
                                            // TODO: View details
                                        }}
                                        className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-background-tertiary transition-colors"
                                    >
                                        <FileText className="w-4 h-4" />
                                        View Details
                                    </button>
                                    <button
                                        onClick={() => {
                                            setMenuOpen(false)
                                            // TODO: Clone experiment
                                        }}
                                        className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-background-tertiary transition-colors"
                                    >
                                        <Copy className="w-4 h-4" />
                                        Clone
                                    </button>
                                    <hr className="my-1 border-background-tertiary" />
                                    <button
                                        onClick={() => {
                                            setMenuOpen(false)
                                            onDelete(experiment.id)
                                        }}
                                        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 transition-colors"
                                    >
                                        <Trash2 className="w-4 h-4" />
                                        Delete
                                    </button>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </td>
        </tr>
    )
}

export function ExperimentList({
    experiments,
    loading,
    onStart,
    onStop,
    onDelete,
    onNewExperiment,
}: ExperimentListProps) {
    return (
        <div className="glass rounded-xl overflow-hidden">
            <div className="p-6 border-b border-background-tertiary">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <h2 className="text-lg font-semibold">Experiments</h2>
                        {loading && (
                            <Loader2 className="w-4 h-4 text-foreground-muted animate-spin" />
                        )}
                    </div>
                    <button
                        onClick={onNewExperiment}
                        className="px-4 py-2 bg-accent hover:bg-accent-hover text-white rounded-lg text-sm font-medium transition-colors"
                    >
                        New Experiment
                    </button>
                </div>
            </div>
            <div className="overflow-x-auto">
                <table className="w-full">
                    <thead>
                        <tr className="border-b border-background-tertiary text-left text-sm text-foreground-muted">
                            <th className="py-3 px-4 font-medium">Experiment</th>
                            <th className="py-3 px-4 font-medium">Status</th>
                            <th className="py-3 px-4 font-medium">Progress</th>
                            <th className="py-3 px-4 font-medium">WER</th>
                            <th className="py-3 px-4 font-medium">Started</th>
                            <th className="py-3 px-4 font-medium">Epochs</th>
                            <th className="py-3 px-4 font-medium">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {experiments.length === 0 ? (
                            <tr>
                                <td colSpan={7} className="py-12 text-center text-foreground-muted">
                                    {loading ? 'Loading experiments...' : 'No experiments yet. Create one to get started!'}
                                </td>
                            </tr>
                        ) : (
                            experiments.map((experiment) => (
                                <ExperimentRow
                                    key={experiment.id}
                                    experiment={experiment}
                                    onStart={onStart}
                                    onStop={onStop}
                                    onDelete={onDelete}
                                />
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
