'use client'

import { Database, Server, Cpu, HardDrive, Wifi, CheckCircle, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ServiceStatus {
    name: string
    status: 'online' | 'offline' | 'degraded'
    icon: typeof Database
    details?: string
}

const services: ServiceStatus[] = [
    { name: 'PostgreSQL', status: 'online', icon: Database, details: 'Supabase' },
    { name: 'Redis', status: 'online', icon: Server, details: 'Upstash' },
    { name: 'MLflow', status: 'online', icon: HardDrive, details: 'localhost:5000' },
    { name: 'Ray Cluster', status: 'online', icon: Cpu, details: '1 GPU node' },
    { name: 'API Server', status: 'online', icon: Wifi, details: 'localhost:8000' },
]

const statusStyles = {
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

function ServiceRow({ service }: { service: ServiceStatus }) {
    const styles = statusStyles[service.status]
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
                            service.status === 'online' && 'animate-ping',
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

export function SystemStatus() {
    const onlineCount = services.filter((s) => s.status === 'online').length
    const allOnline = onlineCount === services.length

    return (
        <div className="glass rounded-xl p-6 h-full">
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold">System Status</h2>
                <div
                    className={cn(
                        'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium',
                        allOnline ? 'bg-green-400/10 text-green-400' : 'bg-yellow-400/10 text-yellow-400'
                    )}
                >
                    {allOnline ? (
                        <CheckCircle className="w-3 h-3" />
                    ) : (
                        <AlertCircle className="w-3 h-3" />
                    )}
                    {onlineCount}/{services.length} Services
                </div>
            </div>

            <div className="space-y-0">
                {services.map((service) => (
                    <ServiceRow key={service.name} service={service} />
                ))}
            </div>

            <div className="mt-6 p-4 rounded-lg bg-background-tertiary/50">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-foreground-muted">GPU Memory</span>
                    <span className="text-sm font-medium">2.8 / 4.0 GB</span>
                </div>
                <div className="h-2 bg-background rounded-full overflow-hidden">
                    <div
                        className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                        style={{ width: '70%' }}
                    />
                </div>
            </div>
        </div>
    )
}
