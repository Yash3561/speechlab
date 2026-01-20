'use client'

import type { Metadata } from 'next'
import './globals.css'
import { AuthProvider } from '@/lib/auth'

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en" className="dark">
            <body className="min-h-screen bg-background text-foreground antialiased">
                <AuthProvider>
                    {children}
                </AuthProvider>
            </body>
        </html>
    )
}
