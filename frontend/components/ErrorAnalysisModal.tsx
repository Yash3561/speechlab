import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog'
import { Button } from './ui/button'
import { Play, Pause } from 'lucide-react'
import { useState } from 'react'

interface ErrorSample {
    id: string
    reference: string
    hypothesis: string
    wer: number
    audio_url: string
}

interface ErrorAnalysisModalProps {
    isOpen: boolean
    onClose: () => void
    samples: ErrorSample[] | undefined
    experimentName: string
}

export function ErrorAnalysisModal({ isOpen, onClose, samples, experimentName }: ErrorAnalysisModalProps) {
    const [playing, setPlaying] = useState<string | null>(null)

    const togglePlay = (url: string) => {
        if (playing === url) {
            setPlaying(null)
            // Logic to stop audio would go here in a real app
        } else {
            setPlaying(url)
            // Logic to play audio
        }
    }

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="max-w-3xl">
                <DialogHeader>
                    <DialogTitle>Error Analysis: {experimentName}</DialogTitle>
                </DialogHeader>

                <div className="space-y-4 max-h-[60vh] overflow-y-auto">
                    {!samples || samples.length === 0 ? (
                        <div className="text-center text-muted-foreground py-8">
                            No error samples available for this experiment.
                        </div>
                    ) : (
                        samples.map((sample) => (
                            <div key={sample.id} className="p-4 rounded-lg border border-border bg-card">
                                <div className="flex justify-between items-start mb-2">
                                    <div className="flex items-center gap-2">
                                        <span className="font-mono text-xs bg-red-500/10 text-red-500 px-2 py-1 rounded">
                                            WER: {(sample.wer * 100).toFixed(1)}%
                                        </span>
                                        <span className="text-sm text-muted-foreground">ID: {sample.id}</span>
                                    </div>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        className="h-8 w-8 p-0"
                                        onClick={() => togglePlay(sample.audio_url)}
                                    >
                                        {playing === sample.audio_url ? (
                                            <Pause className="h-4 w-4" />
                                        ) : (
                                            <Play className="h-4 w-4" />
                                        )}
                                    </Button>
                                </div>

                                <div className="space-y-2 font-mono text-sm">
                                    <div className="grid grid-cols-[80px_1fr] gap-2">
                                        <span className="text-green-500 font-semibold text-right">REF:</span>
                                        <span className="text-foreground/90">{sample.reference}</span>
                                    </div>
                                    <div className="grid grid-cols-[80px_1fr] gap-2">
                                        <span className="text-red-500 font-semibold text-right">HYP:</span>
                                        <span className="text-foreground/90">{sample.hypothesis}</span>
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </DialogContent>
        </Dialog>
    )
}
