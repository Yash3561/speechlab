import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
    title: 'SpeechLab | Production-Grade Speech ML Pipeline',
    description: 'Distributed training and evaluation infrastructure for speech models',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en" className="dark">
            <body className="min-h-screen bg-background text-foreground antialiased">
                {children}
            </body>
        </html>
    )
}
