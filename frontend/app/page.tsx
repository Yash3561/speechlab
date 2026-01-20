'use client'

import { useState, useEffect } from 'react'
import { Activity, Cpu, Database, Zap, Play, BarChart3, Layers, Settings } from 'lucide-react'
import { MetricCard } from '@/components/MetricCard'
import { TrainingChart } from '@/components/TrainingChart'
import { ExperimentList } from '@/components/ExperimentList'
import { SystemStatus } from '@/components/SystemStatus'

export default function Dashboard() {
    const [metrics, setMetrics] = useState({
        activeExperiments: 2,
        gpuUtilization: 87,
        samplesPerSec: 1250,
        avgWER: 4.2,
    })

    // Simulate live updates
    useEffect(() => {
        const interval = setInterval(() => {
            setMetrics(prev => ({
                ...prev,
                gpuUtilization: Math.min(100, Math.max(60, prev.gpuUtilization + (Math.random() - 0.5) * 10)),
                samplesPerSec: Math.max(800, prev.samplesPerSec + (Math.random() - 0.5) * 100),
            }))
        }, 2000)

        return () => clearInterval(interval)
    }, [])

    return (
        <div className="min-h-screen p-6">
            {/* Header */}
            <header className="mb-8">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold gradient-text">SpeechLab</h1>
                        <p className="text-foreground-muted mt-1">Production-Grade Speech ML Pipeline</p>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2 text-success">
                            <span className="relative flex h-3 w-3">
                                <span className="pulse-ring absolute inline-flex h-full w-full rounded-full bg-success opacity-75"></span>
                                <span className="relative inline-flex h-3 w-3 rounded-full bg-success"></span>
                            </span>
                            <span className="text-sm font-medium">System Online</span>
                        </div>
                        <button className="p-2 rounded-lg bg-background-secondary hover:bg-background-tertiary transition-colors">
                            <Settings className="w-5 h-5 text-foreground-muted" />
                        </button>
                    </div>
                </div>
            </header>

            {/* Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <MetricCard
                    title="Active Experiments"
                    value={metrics.activeExperiments}
                    icon={<Play className="w-5 h-5" />}
                    trend="+1 from yesterday"
                    color="blue"
                />
                <MetricCard
                    title="GPU Utilization"
                    value={`${metrics.gpuUtilization.toFixed(0)}%`}
                    icon={<Cpu className="w-5 h-5" />}
                    trend="GTX 1650"
                    color="green"
                />
                <MetricCard
                    title="Throughput"
                    value={`${metrics.samplesPerSec.toFixed(0)}`}
                    suffix="samples/sec"
                    icon={<Zap className="w-5 h-5" />}
                    trend="+12% from baseline"
                    color="yellow"
                />
                <MetricCard
                    title="Best WER"
                    value={`${metrics.avgWER}%`}
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
                        <TrainingChart />
                    </div>
                </div>

                {/* System Status - Takes 1 column */}
                <div className="lg:col-span-1">
                    <SystemStatus />
                </div>

                {/* Experiments List - Full width */}
                <div className="lg:col-span-3">
                    <ExperimentList />
                </div>
            </div>
        </div>
    )
}
