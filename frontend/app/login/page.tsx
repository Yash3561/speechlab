'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/lib/auth'
import { Mail, Lock, User, ArrowRight, Loader2 } from 'lucide-react'

export default function LoginPage() {
    const router = useRouter()
    const { login, signup } = useAuth()

    const [isLogin, setIsLogin] = useState(true)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [name, setName] = useState('')

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setError('')
        setLoading(true)

        try {
            let success: boolean

            if (isLogin) {
                success = await login(email, password)
            } else {
                if (!name.trim()) {
                    setError('Name is required')
                    setLoading(false)
                    return
                }
                success = await signup(email, password, name)
            }

            if (success) {
                router.push('/')
            } else {
                setError(isLogin ? 'Invalid email or password' : 'Failed to create account')
            }
        } catch (err) {
            setError('Something went wrong. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center p-4">
            <div className="w-full max-w-md">
                {/* Logo */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold gradient-text mb-2">SpeechLab</h1>
                    <p className="text-foreground-muted">Speech Model Training Infrastructure</p>
                </div>

                {/* Auth Card */}
                <div className="glass rounded-2xl p-8">
                    {/* Tabs */}
                    <div className="flex mb-6 bg-background-tertiary rounded-lg p-1">
                        <button
                            onClick={() => setIsLogin(true)}
                            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${isLogin
                                    ? 'bg-accent text-white'
                                    : 'text-foreground-muted hover:text-foreground'
                                }`}
                        >
                            Sign In
                        </button>
                        <button
                            onClick={() => setIsLogin(false)}
                            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${!isLogin
                                    ? 'bg-accent text-white'
                                    : 'text-foreground-muted hover:text-foreground'
                                }`}
                        >
                            Sign Up
                        </button>
                    </div>

                    {/* Form */}
                    <form onSubmit={handleSubmit} className="space-y-4">
                        {!isLogin && (
                            <div>
                                <label className="block text-sm font-medium mb-2">Name</label>
                                <div className="relative">
                                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-foreground-muted" />
                                    <input
                                        type="text"
                                        value={name}
                                        onChange={(e) => setName(e.target.value)}
                                        placeholder="Your name"
                                        className="w-full pl-10 pr-4 py-3 bg-background-secondary border border-background-tertiary rounded-lg focus:outline-none focus:border-accent transition-colors"
                                    />
                                </div>
                            </div>
                        )}

                        <div>
                            <label className="block text-sm font-medium mb-2">Email</label>
                            <div className="relative">
                                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-foreground-muted" />
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="you@example.com"
                                    required
                                    className="w-full pl-10 pr-4 py-3 bg-background-secondary border border-background-tertiary rounded-lg focus:outline-none focus:border-accent transition-colors"
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-2">Password</label>
                            <div className="relative">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-foreground-muted" />
                                <input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="••••••••"
                                    required
                                    minLength={6}
                                    className="w-full pl-10 pr-4 py-3 bg-background-secondary border border-background-tertiary rounded-lg focus:outline-none focus:border-accent transition-colors"
                                />
                            </div>
                        </div>

                        {error && (
                            <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                                {error}
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full py-3 bg-accent hover:bg-accent-hover text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                        >
                            {loading ? (
                                <Loader2 className="w-5 h-5 animate-spin" />
                            ) : (
                                <>
                                    {isLogin ? 'Sign In' : 'Create Account'}
                                    <ArrowRight className="w-5 h-5" />
                                </>
                            )}
                        </button>
                    </form>

                    {/* Demo Credentials */}
                    <div className="mt-6 pt-6 border-t border-background-tertiary">
                        <p className="text-sm text-foreground-muted mb-3">Demo Accounts:</p>
                        <div className="space-y-2 text-xs">
                            <div className="flex justify-between p-2 rounded bg-background-secondary">
                                <span className="text-blue-400">Admin</span>
                                <span className="font-mono">admin@speechlab.dev / admin123</span>
                            </div>
                            <div className="flex justify-between p-2 rounded bg-background-secondary">
                                <span className="text-green-400">User</span>
                                <span className="font-mono">user@speechlab.dev / user123</span>
                            </div>
                            <div className="flex justify-between p-2 rounded bg-background-secondary">
                                <span className="text-yellow-400">Viewer</span>
                                <span className="font-mono">viewer@speechlab.dev / viewer123</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
