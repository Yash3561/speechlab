'use client'

import { useState } from 'react'
import { X, Database, Server, HardDrive, Cloud, Check } from 'lucide-react'
import { cn } from '@/lib/utils'

interface SettingsModalProps {
    isOpen: boolean
    onClose: () => void
}

interface SettingsState {
    apiUrl: string
    supabaseUrl: string
    supabaseKey: string
    upstashUrl: string
    upstashToken: string
    r2Endpoint: string
    r2AccessKey: string
    r2SecretKey: string
    mlflowUri: string
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
    const [activeTab, setActiveTab] = useState<'general' | 'cloud' | 'mlflow'>('general')
    const [settings, setSettings] = useState<SettingsState>({
        apiUrl: 'http://localhost:8000',
        supabaseUrl: '',
        supabaseKey: '',
        upstashUrl: '',
        upstashToken: '',
        r2Endpoint: '',
        r2AccessKey: '',
        r2SecretKey: '',
        mlflowUri: 'http://localhost:5000',
    })
    const [saved, setSaved] = useState(false)

    if (!isOpen) return null

    const handleSave = () => {
        // In production, this would save to localStorage or backend
        localStorage.setItem('speechlab_settings', JSON.stringify(settings))
        setSaved(true)
        setTimeout(() => setSaved(false), 2000)
    }

    const tabs = [
        { id: 'general', label: 'General', icon: Server },
        { id: 'cloud', label: 'Cloud Services', icon: Cloud },
        { id: 'mlflow', label: 'MLflow', icon: Database },
    ] as const

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="relative z-10 w-full max-w-2xl glass rounded-xl m-4 max-h-[90vh] overflow-hidden flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-background-tertiary">
                    <h2 className="text-xl font-bold">Settings</h2>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
                    >
                        <X className="w-5 h-5 text-foreground-muted" />
                    </button>
                </div>

                <div className="flex flex-1 min-h-0">
                    {/* Sidebar Tabs */}
                    <div className="w-48 border-r border-background-tertiary p-4 space-y-1">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={cn(
                                    'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                                    activeTab === tab.id
                                        ? 'bg-accent/20 text-accent'
                                        : 'text-foreground-muted hover:bg-background-tertiary hover:text-foreground'
                                )}
                            >
                                <tab.icon className="w-4 h-4" />
                                {tab.label}
                            </button>
                        ))}
                    </div>

                    {/* Content */}
                    <div className="flex-1 p-6 overflow-y-auto">
                        {activeTab === 'general' && (
                            <div className="space-y-5">
                                <div>
                                    <label className="block text-sm font-medium mb-2">
                                        API Server URL
                                    </label>
                                    <input
                                        type="text"
                                        value={settings.apiUrl}
                                        onChange={(e) =>
                                            setSettings({ ...settings, apiUrl: e.target.value })
                                        }
                                        className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none"
                                    />
                                    <p className="text-xs text-foreground-muted mt-1">
                                        Backend server endpoint for API calls
                                    </p>
                                </div>

                                <div className="p-4 rounded-lg bg-background-tertiary/50">
                                    <h3 className="text-sm font-medium mb-3">Quick Setup</h3>
                                    <div className="space-y-2 text-sm text-foreground-muted">
                                        <p>• Start backend: <code className="text-accent">uvicorn backend.api.main:app --reload</code></p>
                                        <p>• Start frontend: <code className="text-accent">npm run dev</code></p>
                                        <p>• Start Docker services: <code className="text-accent">docker-compose up -d</code></p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeTab === 'cloud' && (
                            <div className="space-y-6">
                                {/* Supabase */}
                                <div className="space-y-4">
                                    <div className="flex items-center gap-2">
                                        <Database className="w-4 h-4 text-green-400" />
                                        <h3 className="font-medium">Supabase (PostgreSQL)</h3>
                                    </div>
                                    <div className="grid gap-4">
                                        <input
                                            type="text"
                                            placeholder="Supabase URL"
                                            value={settings.supabaseUrl}
                                            onChange={(e) =>
                                                setSettings({ ...settings, supabaseUrl: e.target.value })
                                            }
                                            className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none text-sm"
                                        />
                                        <input
                                            type="password"
                                            placeholder="Supabase Anon Key"
                                            value={settings.supabaseKey}
                                            onChange={(e) =>
                                                setSettings({ ...settings, supabaseKey: e.target.value })
                                            }
                                            className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none text-sm"
                                        />
                                    </div>
                                </div>

                                {/* Upstash */}
                                <div className="space-y-4">
                                    <div className="flex items-center gap-2">
                                        <Server className="w-4 h-4 text-red-400" />
                                        <h3 className="font-medium">Upstash (Redis)</h3>
                                    </div>
                                    <div className="grid gap-4">
                                        <input
                                            type="text"
                                            placeholder="Upstash Redis URL"
                                            value={settings.upstashUrl}
                                            onChange={(e) =>
                                                setSettings({ ...settings, upstashUrl: e.target.value })
                                            }
                                            className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none text-sm"
                                        />
                                        <input
                                            type="password"
                                            placeholder="Upstash Token"
                                            value={settings.upstashToken}
                                            onChange={(e) =>
                                                setSettings({ ...settings, upstashToken: e.target.value })
                                            }
                                            className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none text-sm"
                                        />
                                    </div>
                                </div>

                                {/* Cloudflare R2 */}
                                <div className="space-y-4">
                                    <div className="flex items-center gap-2">
                                        <HardDrive className="w-4 h-4 text-orange-400" />
                                        <h3 className="font-medium">Cloudflare R2 (Storage)</h3>
                                    </div>
                                    <div className="grid gap-4">
                                        <input
                                            type="text"
                                            placeholder="R2 Endpoint"
                                            value={settings.r2Endpoint}
                                            onChange={(e) =>
                                                setSettings({ ...settings, r2Endpoint: e.target.value })
                                            }
                                            className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none text-sm"
                                        />
                                        <div className="grid grid-cols-2 gap-4">
                                            <input
                                                type="text"
                                                placeholder="Access Key"
                                                value={settings.r2AccessKey}
                                                onChange={(e) =>
                                                    setSettings({ ...settings, r2AccessKey: e.target.value })
                                                }
                                                className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none text-sm"
                                            />
                                            <input
                                                type="password"
                                                placeholder="Secret Key"
                                                value={settings.r2SecretKey}
                                                onChange={(e) =>
                                                    setSettings({ ...settings, r2SecretKey: e.target.value })
                                                }
                                                className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none text-sm"
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeTab === 'mlflow' && (
                            <div className="space-y-5">
                                <div>
                                    <label className="block text-sm font-medium mb-2">
                                        MLflow Tracking URI
                                    </label>
                                    <input
                                        type="text"
                                        value={settings.mlflowUri}
                                        onChange={(e) =>
                                            setSettings({ ...settings, mlflowUri: e.target.value })
                                        }
                                        className="w-full px-4 py-2.5 rounded-lg bg-background-tertiary border border-background-tertiary focus:border-accent outline-none"
                                    />
                                    <p className="text-xs text-foreground-muted mt-1">
                                        Local: http://localhost:5000 | Cloud: mlflow.yourdomain.com
                                    </p>
                                </div>

                                <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                                    <h3 className="text-sm font-medium text-green-400 mb-2">
                                        Connection Status
                                    </h3>
                                    <p className="text-sm text-foreground-muted">
                                        Run `docker-compose up -d mlflow` to start MLflow locally
                                    </p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between p-6 border-t border-background-tertiary">
                    <p className="text-xs text-foreground-muted">
                        Settings are stored locally in your browser
                    </p>
                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="px-4 py-2 rounded-lg bg-background-tertiary hover:bg-background text-foreground-muted hover:text-foreground transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSave}
                            className={cn(
                                'px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2',
                                saved
                                    ? 'bg-green-500 text-white'
                                    : 'bg-accent hover:bg-accent-hover text-white'
                            )}
                        >
                            {saved && <Check className="w-4 h-4" />}
                            {saved ? 'Saved!' : 'Save Settings'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}
