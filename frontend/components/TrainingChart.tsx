'use client'

import { useState, useEffect, useRef } from 'react'
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from 'recharts'

interface MetricPoint {
    step: number
    trainLoss: number
    valLoss: number
}

interface TrainingChartProps {
    experimentId?: string | null
}

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        return (
            <div className="glass rounded-lg p-3 text-sm">
                <p className="font-medium mb-2">Step {label}</p>
                {payload.map((entry: any, index: number) => (
                    <p key={index} style={{ color: entry.color }}>
                        {entry.name}: {entry.value.toFixed(4)}
                    </p>
                ))}
            </div>
        )
    }
    return null
}

// Generate initial demo data
function generateDemoData(numPoints: number = 30): MetricPoint[] {
    const data: MetricPoint[] = []
    let trainLoss = 2.5
    let valLoss = 2.8

    for (let i = 0; i < numPoints; i++) {
        trainLoss = Math.max(0.1, trainLoss - 0.04 + (Math.random() - 0.5) * 0.02)
        valLoss = Math.max(0.15, valLoss - 0.035 + (Math.random() - 0.5) * 0.03)

        data.push({
            step: i * 100,
            trainLoss: parseFloat(trainLoss.toFixed(4)),
            valLoss: parseFloat(valLoss.toFixed(4)),
        })
    }

    return data
}

export function TrainingChart({ experimentId }: TrainingChartProps) {
    const [data, setData] = useState<MetricPoint[]>(generateDemoData(30))
    const [connected, setConnected] = useState(false)
    const wsRef = useRef<WebSocket | null>(null)

    useEffect(() => {
        // If no experiment ID provided, show demo animation
        if (!experimentId) {
            const interval = setInterval(() => {
                setData((prev) => {
                    if (prev.length >= 100) {
                        // Reset when too many points
                        return generateDemoData(30)
                    }

                    const lastPoint = prev[prev.length - 1]
                    const newTrainLoss = Math.max(
                        0.1,
                        lastPoint.trainLoss - 0.03 + (Math.random() - 0.5) * 0.02
                    )
                    const newValLoss = Math.max(
                        0.15,
                        lastPoint.valLoss - 0.025 + (Math.random() - 0.5) * 0.03
                    )

                    return [
                        ...prev,
                        {
                            step: lastPoint.step + 100,
                            trainLoss: parseFloat(newTrainLoss.toFixed(4)),
                            valLoss: parseFloat(newValLoss.toFixed(4)),
                        },
                    ]
                })
            }, 3000)

            return () => clearInterval(interval)
        }

        // Connect to WebSocket for real experiment
        const wsUrl = `ws://localhost:8000/api/experiments/ws/${experimentId}`

        try {
            const ws = new WebSocket(wsUrl)
            wsRef.current = ws

            ws.onopen = () => {
                console.log(`WebSocket connected for experiment: ${experimentId}`)
                setConnected(true)
                // Reset data when connecting to new experiment
                setData([])
            }

            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data)

                    if (message.type === 'metrics' && message.data) {
                        const { step, train_loss, val_loss } = message.data

                        setData((prev) => {
                            // Limit to 100 points for performance
                            const newData = prev.length >= 100 ? prev.slice(-99) : prev

                            return [
                                ...newData,
                                {
                                    step,
                                    trainLoss: train_loss,
                                    valLoss: val_loss,
                                },
                            ]
                        })
                    }
                } catch (e) {
                    console.error('Error parsing WebSocket message:', e)
                }
            }

            ws.onclose = () => {
                console.log('WebSocket disconnected')
                setConnected(false)
            }

            ws.onerror = (error) => {
                console.error('WebSocket error:', error)
                setConnected(false)
            }

            return () => {
                ws.close()
            }
        } catch (e) {
            console.error('Failed to connect WebSocket:', e)
        }
    }, [experimentId])

    return (
        <div className="relative">
            {connected && (
                <div className="absolute top-0 right-0 flex items-center gap-2 text-xs text-green-400">
                    <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                        <span className="relative inline-flex h-2 w-2 rounded-full bg-green-400"></span>
                    </span>
                    Live
                </div>
            )}

            <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                        data={data}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                        <XAxis
                            dataKey="step"
                            stroke="#71717a"
                            fontSize={12}
                            tickFormatter={(value) => `${value}`}
                        />
                        <YAxis
                            stroke="#71717a"
                            fontSize={12}
                            tickFormatter={(value) => value.toFixed(2)}
                            domain={['auto', 'auto']}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Line
                            type="monotone"
                            dataKey="trainLoss"
                            name="Train Loss"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            dot={false}
                            animationDuration={300}
                            isAnimationActive={!connected}
                        />
                        <Line
                            type="monotone"
                            dataKey="valLoss"
                            name="Val Loss"
                            stroke="#22c55e"
                            strokeWidth={2}
                            dot={false}
                            animationDuration={300}
                            isAnimationActive={!connected}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
}
