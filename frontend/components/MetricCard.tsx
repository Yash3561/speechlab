import { cn } from '@/lib/utils'
import { ReactNode } from 'react'

interface MetricCardProps {
    title: string
    value: string | number
    suffix?: string
    icon: ReactNode
    trend?: string
    color?: 'blue' | 'green' | 'yellow' | 'purple' | 'red'
}

const colorStyles = {
    blue: 'from-blue-500/20 to-blue-600/5 border-blue-500/30',
    green: 'from-green-500/20 to-green-600/5 border-green-500/30',
    yellow: 'from-yellow-500/20 to-yellow-600/5 border-yellow-500/30',
    purple: 'from-purple-500/20 to-purple-600/5 border-purple-500/30',
    red: 'from-red-500/20 to-red-600/5 border-red-500/30',
}

const iconColors = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    yellow: 'text-yellow-400',
    purple: 'text-purple-400',
    red: 'text-red-400',
}

export function MetricCard({
    title,
    value,
    suffix,
    icon,
    trend,
    color = 'blue',
}: MetricCardProps) {
    return (
        <div
            className={cn(
                'relative overflow-hidden rounded-xl border p-5',
                'bg-gradient-to-br',
                colorStyles[color],
                'transition-all duration-300 hover:scale-[1.02] hover:shadow-lg'
            )}
        >
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-sm text-foreground-muted font-medium">{title}</p>
                    <div className="mt-2 flex items-baseline gap-1">
                        <span className="text-3xl font-bold tabular-nums">{value}</span>
                        {suffix && (
                            <span className="text-sm text-foreground-muted">{suffix}</span>
                        )}
                    </div>
                    {trend && (
                        <p className="mt-2 text-xs text-foreground-muted">{trend}</p>
                    )}
                </div>
                <div className={cn('p-2 rounded-lg bg-background/50', iconColors[color])}>
                    {icon}
                </div>
            </div>
        </div>
    )
}
