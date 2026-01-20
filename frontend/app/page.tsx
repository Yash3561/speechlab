import Link from 'next/link'
import { ArrowRight, Activity, Zap, Shield, GitBranch, Cpu, Database } from 'lucide-react'

export default function LandingPage() {
    return (
        <div className="min-h-screen bg-background text-foreground flex flex-col">
            {/* Navbar */}
            <nav className="border-b border-border/50 backdrop-blur-md sticky top-0 z-50">
                <div className="container mx-auto px-6 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-2 font-bold text-xl">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                            <Activity className="w-5 h-5 text-white" />
                        </div>
                        <span>SpeechLab</span>
                    </div>
                    <div className="flex items-center gap-4">
                        <Link href="https://github.com/Yash3561/speechlab" className="text-foreground-muted hover:text-foreground transition-colors font-medium text-sm">
                            Documentation
                        </Link>
                        <Link href="/dashboard" className="px-5 py-2 rounded-full bg-foreground text-background font-medium text-sm hover:opacity-90 transition-opacity">
                            Login / Dashboard
                        </Link>
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            <main className="flex-1 flex flex-col">
                <section className="py-24 lg:py-32 relative overflow-hidden">
                    <div className="absolute inset-0 bg-gradient-to-b from-blue-500/10 via-purple-500/5 to-background pointer-events-none" />

                    <div className="container mx-auto px-6 text-center relative z-10">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 text-blue-400 text-xs font-medium mb-6 animate-fade-in-up">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                            </span>
                            v1.0 is now available
                        </div>

                        <h1 className="text-5xl lg:text-7xl font-bold tracking-tight mb-6 bg-gradient-to-b from-foreground to-foreground-muted bg-clip-text text-transparent pb-2">
                            Scale Your Speech <br />
                            <span className="bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">Research Infrastructure</span>
                        </h1>

                        <p className="text-xl text-foreground-muted max-w-2xl mx-auto mb-10 leading-relaxed">
                            The enterprise-grade platform for distributed training, reproducible evaluation, and real-time observability. Built for ML Engineers who ship.
                        </p>

                        <div className="flex items-center justify-center gap-4">
                            <Link href="/dashboard" className="px-8 py-4 rounded-full bg-blue-600 text-white font-bold hover:bg-blue-500 hover:scale-105 transition-all flex items-center gap-2 shadow-lg shadow-blue-500/25">
                                Get Started <ArrowRight className="w-5 h-5" />
                            </Link>
                            <Link href="https://github.com/Yash3561/speechlab" className="px-8 py-4 rounded-full bg-background-secondary text-foreground font-bold hover:bg-background-tertiary transition-all border border-border">
                                View on GitHub
                            </Link>
                        </div>
                    </div>
                </section>

                {/* Features Grid */}
                <section className="py-24 bg-background-secondary/30">
                    <div className="container mx-auto px-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                            <FeatureCard
                                icon={<Cpu className="w-6 h-6 text-blue-400" />}
                                title="Distributed Training"
                                description="Scale from 1 to 100 GPUs seamlessly with Ray Train orchestration. Zero code changes required."
                            />
                            <FeatureCard
                                icon={<Zap className="w-6 h-6 text-yellow-400" />}
                                title="Real-time Observability"
                                description="Watch loss curves, WER, and GPU metrics update live via WebSocket streaming pipeline."
                            />
                            <FeatureCard
                                icon={<Shield className="w-6 h-6 text-green-400" />}
                                title="Regression Protection"
                                description="Automated baseline comparison pipelines ensure you never ship a degraded model."
                            />
                            <FeatureCard
                                icon={<GitBranch className="w-6 h-6 text-purple-400" />}
                                title="100% Reproducibility"
                                description="Every run is versioned. Code, config, data, and environment snapshots via MLflow."
                            />
                            <FeatureCard
                                icon={<Database className="w-6 h-6 text-pink-400" />}
                                title="Data Streaming"
                                description="Train on petabyte-scale datasets without OOM errors using Ray Data lazy loading."
                            />
                            <FeatureCard
                                icon={<Activity className="w-6 h-6 text-cyan-400" />}
                                title="Interactive Playground"
                                description="Drag & drop audio files to test your trained models immediately in the browser."
                            />
                        </div>
                    </div>
                </section>
            </main>

            <footer className="py-12 border-t border-border/50">
                <div className="container mx-auto px-6 text-center text-foreground-muted">
                    <p>© 2026 SpeechLab Platform. Built for the ML Community.</p>
                </div>
            </footer>
        </div>
    )
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
    return (
        <div className="p-8 rounded-2xl glass border border-white/5 hover:border-white/10 transition-colors group">
            <div className="w-12 h-12 rounded-xl bg-background-tertiary flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                {icon}
            </div>
            <h3 className="text-xl font-bold mb-3">{title}</h3>
            <p className="text-foreground-muted leading-relaxed">
                {description}
            </p>
        </div>
    )
}
