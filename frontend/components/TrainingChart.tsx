'use client'

import { useState, useEffect } from 'react'
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from 'recharts'

// Generate demo training data
function generateDemoData(numPoints: number = 50) {
    const data = []
    let trainLoss = 2.5
    let valLoss = 2.8

    for (let i = 0; i < numPoints; i++) {
        // Simulate loss decrease with some noise
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

export function TrainingChart() {
    const [data, setData] = useState(generateDemoData(30))

    // Simulate live updates
    useEffect(() => {
        const interval = setInterval(() => {
            setData((prev) => {
                if (prev.length >= 100) return prev

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
    }, [])

    return (
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
                    />
                    <Line
                        type="monotone"
                        dataKey="valLoss"
                        name="Val Loss"
                        stroke="#22c55e"
                        strokeWidth={2}
                        dot={false}
                        animationDuration={300}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    )
}
