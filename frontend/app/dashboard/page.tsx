'use client'

import { useState, useEffect, useMemo } from 'react'
import { Activity, Cpu, Zap, Play, BarChart3, Settings, RefreshCw, Gamepad2 } from 'lucide-react'
import Link from 'next/link'
import { MetricCard } from '@/components/MetricCard'
import { TrainingChart } from '@/components/TrainingChart'
import { ExperimentList } from '@/components/ExperimentList'
import { SystemStatus } from '@/components/SystemStatus'
import { NewExperimentModal } from '@/components/NewExperimentModal'
import { SettingsModal } from '@/components/SettingsModal'
import { useExperiments, useHealth } from '@/lib/hooks'

export default function Dashboard() {
    const [showNewExperiment, setShowNewExperiment] = useState(false)
    const [showSettings, setShowSettings] = useState(false)

    const {
        experiments,
        loading,
        error,
        refresh,
        startExperiment,
        stopExperiment,
        deleteExperiment,
        createExperiment
    } = useExperiments()

    const { connected } = useHealth()

    // Compute dashboard metrics from experiments
    const { metrics, runningExperimentId } = useMemo(() => {
        const activeExperiments = experiments.filter(e => e.status === 'running').length
        const completedExperiments = experiments.filter(e => e.status === 'completed')
        const bestWER = completedExperiments.length > 0
            ? Math.min(...completedExperiments.map(e => e.wer || Infinity))
            : null

        // Find first running experiment for the chart
        const runningExp = experiments.find(e => e.status === 'running')

        return {
            metrics: {
                activeExperiments,
                gpuUtilization: runningExp ? 70 + Math.random() * 20 : 0,
                samplesPerSec: runningExp ? 1100 + Math.random() * 300 : 0,
                bestWER: bestWER !== Infinity && bestWER !== null ? bestWER : 4.2,
            },
            runningExperimentId: runningExp?.id || null,
        }
    }, [experiments])

    // Simulate live GPU updates for running experiments
    const [gpuUtil, setGpuUtil] = useState(0)
    const [throughput, setThroughput] = useState(0)

    useEffect(() => {
        const hasRunning = experiments.some(e => e.status === 'running')
        if (!hasRunning) {
            setGpuUtil(0)
            setThroughput(0)
            return
        }

        const interval = setInterval(() => {
            setGpuUtil(70 + Math.random() * 25)
            setThroughput(1100 + Math.random() * 400)
        }, 2000)

        return () => clearInterval(interval)
    }, [experiments])

    return (
        <div className="min-h-screen p-6">
            {/* Header */}
            <header className="mb-8">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold gradient-text">SpeechLab</h1>
                        <p className="text-foreground-muted mt-1">Speech Model Training Infrastructure</p>
                    </div>
                    <div className="flex items-center gap-4">
                        <Link
                            href="/playground"
                            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-colors font-medium text-sm"
                        >
                            <Gamepad2 className="w-4 h-4" />
                            Playground
                        </Link>
                        <button
                            onClick={refresh}
                            className="p-2 rounded-lg bg-background-secondary hover:bg-background-tertiary transition-colors"
                            title="Refresh"
                        >
                            <RefreshCw className={`w-5 h-5 text-foreground-muted ${loading ? 'animate-spin' : ''}`} />
                        </button>
                        <div className="flex items-center gap-2">
                            <span className={`relative flex h-3 w-3`}>
                                {connected && (
                                    <span className="pulse-ring absolute inline-flex h-full w-full rounded-full bg-success opacity-75"></span>
                                )}
                                <span className={`relative inline-flex h-3 w-3 rounded-full ${connected ? 'bg-success' : 'bg-red-500'}`}></span>
                            </span>
                            <span className={`text-sm font-medium ${connected ? 'text-success' : 'text-red-500'}`}>
                                {connected ? 'System Online' : 'Disconnected'}
                            </span>
                        </div>
                        <button
                            onClick={() => setShowSettings(true)}
                            className="p-2 rounded-lg bg-background-secondary hover:bg-background-tertiary transition-colors"
                        >
                            <Settings className="w-5 h-5 text-foreground-muted" />
                        </button>
                    </div>
                </div>
            </header>

            {/* Error Banner */}
            {error && (
                <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400">
                    <p className="text-sm">{error}</p>
                </div>
            )}

            {/* Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <MetricCard
                    title="Active Experiments"
                    value={metrics.activeExperiments}
                    icon={<Play className="w-5 h-5" />}
                    trend={`${experiments.length} total`}
                    color="blue"
                />
                <MetricCard
                    title="GPU Utilization"
                    value={gpuUtil > 0 ? `${gpuUtil.toFixed(0)}%` : '0%'}
                    icon={<Cpu className="w-5 h-5" />}
                    trend="GTX 1650"
                    color="green"
                />
                <MetricCard
                    title="Throughput"
                    value={throughput > 0 ? `${throughput.toFixed(0)}` : '0'}
                    suffix="samples/sec"
                    icon={<Zap className="w-5 h-5" />}
                    trend={metrics.activeExperiments > 0 ? 'Training active' : 'Idle'}
                    color="yellow"
                />
                <MetricCard
                    title="Best WER"
                    value={`${metrics.bestWER}%`}
                    icon={<BarChart3 className="w-5 h-5" />}
                    trend="Whisper Tiny"
                    color="purple"
                />
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Training Chart - Takes 2 columns */}
                <div className="lg:col-span-2">
                    <div className="glass rounded-xl p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-lg font-semibold">Training Progress</h2>
                            <div className="flex items-center gap-4 text-sm">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                                    <span className="text-foreground-muted">Train Loss</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                                    <span className="text-foreground-muted">Val Loss</span>
                                </div>
                            </div>
                        </div>
                        <TrainingChart experimentId={runningExperimentId} />
                    </div>
                </div>

                {/* System Status - Takes 1 column */}
                <div className="lg:col-span-1">
                    <SystemStatus />
                </div>

                {/* Experiments List - Full width */}
                <div className="lg:col-span-3">
                    <ExperimentList
                        experiments={experiments}
                        loading={loading}
                        onStart={startExperiment}
                        onStop={stopExperiment}
                        onDelete={deleteExperiment}
                        onNewExperiment={() => setShowNewExperiment(true)}
                    />
                </div>
            </div>

            {/* Modals */}
            <NewExperimentModal
                isOpen={showNewExperiment}
                onClose={() => setShowNewExperiment(false)}
                onCreate={createExperiment}
            />

            <SettingsModal
                isOpen={showSettings}
                onClose={() => setShowSettings(false)}
            />
        </div>
    )
}
