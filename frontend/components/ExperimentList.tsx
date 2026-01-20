'use client'

import { Play, Pause, MoreVertical, Clock, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Experiment {
    id: string
    name: string
    status: 'running' | 'completed' | 'failed' | 'pending'
    model: string
    progress: number
    wer?: number
    startedAt: string
    duration: string
}

const demoExperiments: Experiment[] = [
    {
        id: 'exp_001',
        name: 'whisper_tiny_librispeech',
        status: 'running',
        model: 'Whisper Tiny',
        progress: 67,
        wer: 5.2,
        startedAt: '2 hours ago',
        duration: '2h 15m',
    },
    {
        id: 'exp_002',
        name: 'whisper_base_noisy',
        status: 'completed',
        model: 'Whisper Base',
        progress: 100,
        wer: 4.1,
        startedAt: 'Yesterday',
        duration: '8h 32m',
    },
    {
        id: 'exp_003',
        name: 'wav2vec2_baseline',
        status: 'failed',
        model: 'Wav2Vec2 Base',
        progress: 34,
        startedAt: '3 days ago',
        duration: '45m',
    },
    {
        id: 'exp_004',
        name: 'whisper_tiny_augmented',
        status: 'pending',
        model: 'Whisper Tiny',
        progress: 0,
        startedAt: 'Queued',
        duration: '-',
    },
]

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
}

function ExperimentRow({ experiment }: { experiment: Experiment }) {
    const config = statusConfig[experiment.status]
    const StatusIcon = config.icon

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
                        <p className="text-sm text-foreground-muted">{experiment.model}</p>
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
                                experiment.status === 'failed' ? 'bg-red-500' : 'bg-blue-500'
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
            <td className="py-4 px-4 text-foreground-muted">{experiment.startedAt}</td>
            <td className="py-4 px-4 text-foreground-muted font-mono">{experiment.duration}</td>
            <td className="py-4 px-4">
                <div className="flex items-center gap-2">
                    {experiment.status === 'running' ? (
                        <button className="p-1.5 rounded-lg hover:bg-background-tertiary transition-colors">
                            <Pause className="w-4 h-4 text-foreground-muted" />
                        </button>
                    ) : experiment.status === 'pending' ? (
                        <button className="p-1.5 rounded-lg hover:bg-background-tertiary transition-colors">
                            <Play className="w-4 h-4 text-foreground-muted" />
                        </button>
                    ) : null}
                    <button className="p-1.5 rounded-lg hover:bg-background-tertiary transition-colors">
                        <MoreVertical className="w-4 h-4 text-foreground-muted" />
                    </button>
                </div>
            </td>
        </tr>
    )
}

export function ExperimentList() {
    return (
        <div className="glass rounded-xl overflow-hidden">
            <div className="p-6 border-b border-background-tertiary">
                <div className="flex items-center justify-between">
                    <h2 className="text-lg font-semibold">Experiments</h2>
                    <button className="px-4 py-2 bg-accent hover:bg-accent-hover text-white rounded-lg text-sm font-medium transition-colors">
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
                            <th className="py-3 px-4 font-medium">Duration</th>
                            <th className="py-3 px-4 font-medium">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {demoExperiments.map((experiment) => (
                            <ExperimentRow key={experiment.id} experiment={experiment} />
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
