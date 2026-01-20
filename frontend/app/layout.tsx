import type { Metadata } from 'next'
import './globals.css'
import { Providers } from './providers'

export const metadata: Metadata = {
    title: 'SpeechLab | Speech Model Training Infrastructure',
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
                <Providers>
                    {children}
                </Providers>
            </body>
        </html>
    )
}
