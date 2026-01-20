import { AlertTriangle, CheckCircle, TrendingDown, TrendingUp, Minus } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface RegressionReport {
    candidate_id: string
    baseline_id: string
    metric: string
    candidate_value: number
    baseline_value: number
    diff: number
    relative_diff: number
    is_regression: boolean
    is_improvement: boolean
    severity: 'critical' | 'minor' | 'none'
}

interface RegressionAlertProps {
    report: RegressionReport | null
    loading?: boolean
    onPromote?: () => void
}

export function RegressionAlert({ report, loading, onPromote }: RegressionAlertProps) {
    if (loading) return <div className="text-sm text-muted-foreground">Analyzing regression...</div>
    if (!report || report.baseline_id === 'none') return null

    const isRegression = report.is_regression
    const isImprovement = report.is_improvement

    // Determine color and icon
    let colorClass = "bg-secondary/50 border-secondary"
    let icon = <Minus className="h-5 w-5" />
    let title = "No Significant Change"

    if (isRegression) {
        colorClass = report.severity === 'critical'
            ? "bg-red-500/10 border-red-500/50 text-red-500"
            : "bg-orange-500/10 border-orange-500/50 text-orange-500"
        icon = <AlertTriangle className="h-5 w-5" />
        title = `Regression Detected (${report.relative_diff}% Worse)`
    } else if (isImprovement) {
        colorClass = "bg-green-500/10 border-green-500/50 text-green-500"
        icon = <CheckCircle className="h-5 w-5" />
        title = `Improvement Detected (${Math.abs(report.relative_diff)}% Better)`
    }

    return (
        <div className={`flex items-center justify-between p-4 rounded-lg border ${colorClass} mb-4`}>
            <div className="flex items-center gap-4">
                {icon}
                <div>
                    <h4 className="font-semibold">{title}</h4>
                    <p className="text-sm opacity-90">
                        {report.metric.toUpperCase()}: {report.candidate_value} vs Baseline {report.baseline_value}
                    </p>
                </div>
            </div>

            {isImprovement && onPromote && (
                <Button
                    variant="outline"
                    size="sm"
                    onClick={onPromote}
                    className="border-green-500/50 hover:bg-green-500/20 text-green-500"
                >
                    Promote to Baseline
                </Button>
            )}
        </div>
    )
}
