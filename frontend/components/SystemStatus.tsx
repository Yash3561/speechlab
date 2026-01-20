'use client'

import { useEffect, useState } from 'react'
import { Database, Server, Cpu, HardDrive, Wifi, CheckCircle, AlertCircle, Activity, Box } from 'lucide-react'
import { cn } from '@/lib/utils'
import { api, type HealthStatus } from '@/lib/api'

interface ServiceStatus {
    name: string
    key: string
    icon: typeof Database
    details?: string
}

const serviceConfig: ServiceStatus[] = [
    { name: 'API Server', key: 'api', icon: Wifi, details: 'FastAPI' },
    { name: 'Database', key: 'database', icon: Database, details: 'Supabase' },
    { name: 'Redis Cache', key: 'redis', icon: Server, details: 'Upstash' },
    { name: 'MLflow', key: 'mlflow', icon: Activity, details: 'Tracking' },
    { name: 'Ray Cluster', key: 'ray_cluster', icon: Box, details: 'Compute' },
]

const statusStyles: Record<string, any> = {
    online: {
        dot: 'bg-green-400',
        text: 'text-green-400',
        label: 'Online',
    },
    offline: {
        dot: 'bg-red-400',
        text: 'text-red-400',
        label: 'Offline',
    },
    degraded: {
        dot: 'bg-yellow-400',
        text: 'text-yellow-400',
        label: 'Degraded',
    },
}

function ServiceRow({ service, status }: { service: ServiceStatus, status: string }) {
    const styles = statusStyles[status] || statusStyles.offline
    const Icon = service.icon

    return (
        <div className="flex items-center justify-between py-3 border-b border-background-tertiary last:border-0">
            <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-background-tertiary">
                    <Icon className="w-4 h-4 text-foreground-muted" />
                </div>
                <div>
                    <p className="font-medium text-sm">{service.name}</p>
                    {service.details && (
                        <p className="text-xs text-foreground-muted">{service.details}</p>
                    )}
                </div>
            </div>
            <div className="flex items-center gap-2">
                <span className={cn('relative flex h-2 w-2')}>
                    <span
                        className={cn(
                            'absolute inline-flex h-full w-full rounded-full opacity-75',
                            status === 'online' && 'animate-ping',
                            styles.dot
                        )}
                    />
                    <span className={cn('relative inline-flex h-2 w-2 rounded-full', styles.dot)} />
                </span>
                <span className={cn('text-xs font-medium', styles.text)}>{styles.label}</span>
            </div>
        </div>
    )
}

function MetricBar({ label, value, total, unit, colorClass }: { label: string, value: number, total?: number, unit: string, colorClass: string }) {
    const percent = total ? (value / total) * 100 : value
    const displayValue = total ? `${value} / ${total} ${unit}` : `${value}${unit}`

    return (
        <div className="mt-4 first:mt-0">
            <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-foreground-muted">{label}</span>
                <span className="text-sm font-medium">{displayValue}</span>
            </div>
            <div className="h-2 bg-background rounded-full overflow-hidden">
                <div
                    className={cn("h-full rounded-full transition-all duration-500", colorClass)}
                    style={{ width: `${Math.min(100, percent)}%` }}
                />
            </div>
        </div>
    )
}

export function SystemStatus() {
    const [health, setHealth] = useState<HealthStatus | null>(null)
    const [loading, setLoading] = useState(true)

    const fetchHealth = async () => {
        try {
            const data = await api.getHealth()
            setHealth(data)
        } catch (error) {
            console.error("Failed to fetch health:", error)
            setHealth(null) // Reset to show offline? Or keep last known?
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchHealth()
        const interval = setInterval(fetchHealth, 5000)
        return () => clearInterval(interval)
    }, [])

    const onlineCount = health ? Object.values(health.services).filter(s => s === 'online').length : 0
    const totalServices = serviceConfig.length
    const allOnline = onlineCount === totalServices

    return (
        <div className="glass rounded-xl p-6 h-full flex flex-col">
            <div className="flex items-center justify-between mb-6 shrink-0">
                <h2 className="text-lg font-semibold">System Status</h2>
                <div
                    className={cn(
                        'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium',
                        allOnline && health ? 'bg-green-400/10 text-green-400' : 'bg-yellow-400/10 text-yellow-400'
                    )}
                >
                    {allOnline && health ? (
                        <CheckCircle className="w-3 h-3" />
                    ) : (
                        <AlertCircle className="w-3 h-3" />
                    )}
                    {onlineCount}/{totalServices} Services
                </div>
            </div>

            <div className="space-y-0 shrink-0">
                {serviceConfig.map((service) => (
                    <ServiceRow
                        key={service.name}
                        service={service}
                        status={health?.services[service.key] || 'offline'}
                    />
                ))}
            </div>

            <div className="mt-6 p-4 rounded-lg bg-background-tertiary/50 grow flex flex-col justify-center">
                {health?.system_metrics ? (
                    <>
                        <MetricBar
                            label="CPU Usage"
                            value={health.system_metrics.cpu_percent}
                            unit="%"
                            colorClass="bg-blue-500"
                        />
                        <MetricBar
                            label="Memory"
                            value={health.system_metrics.memory_used_gb}
                            total={health.system_metrics.memory_total_gb}
                            unit="GB"
                            colorClass="bg-purple-500"
                        />
                        <MetricBar
                            label="GPU Memory"
                            value={health.system_metrics.gpu_memory_used}
                            total={health.system_metrics.gpu_memory_total}
                            unit="GB"
                            colorClass="bg-orange-500"
                        />
                    </>
                ) : (
                    <div className="text-center text-sm text-foreground-muted">
                        Loading metrics...
                    </div>
                )}
            </div>
        </div>
    )
}
