"""
SpeechLab Metrics

Speech recognition evaluation metrics.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class TranscriptionMetrics:
    """Metrics for a single transcription."""
    wer: float  # Word Error Rate
    cer: float  # Character Error Rate
    insertions: int
    deletions: int
    substitutions: int
    ref_words: int
    hyp_words: int


def compute_wer(reference: str, hypothesis: str) -> TranscriptionMetrics:
    """
    Compute Word Error Rate between reference and hypothesis.
    
    WER = (S + D + I) / N
    
    Where:
        S = Substitutions
        D = Deletions
        I = Insertions
        N = Number of words in reference
        
    Args:
        reference: Ground truth transcription
        hypothesis: Model prediction
        
    Returns:
        TranscriptionMetrics with WER and breakdown
    """
    try:
        from jiwer import wer, cer, compute_measures
        
        measures = compute_measures(reference, hypothesis)
        
        return TranscriptionMetrics(
            wer=measures["wer"],
            cer=cer(reference, hypothesis),
            insertions=measures["insertions"],
            deletions=measures["deletions"],
            substitutions=measures["substitutions"],
            ref_words=len(reference.split()),
            hyp_words=len(hypothesis.split()),
        )
        
    except ImportError:
        # Fallback implementation
        return _compute_wer_fallback(reference, hypothesis)


def _compute_wer_fallback(reference: str, hypothesis: str) -> TranscriptionMetrics:
    """Fallback WER computation without jiwer."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Levenshtein distance with backtrace
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # Backtrace to get error breakdown
    i, j = m, n
    substitutions = deletions = insertions = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    
    wer = dp[m][n] / max(m, 1)
    
    # Simple CER
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))
    cer = _edit_distance(ref_chars, hyp_chars) / max(len(ref_chars), 1)
    
    return TranscriptionMetrics(
        wer=wer,
        cer=cer,
        insertions=insertions,
        deletions=deletions,
        substitutions=substitutions,
        ref_words=m,
        hyp_words=n,
    )


def _edit_distance(a: List, b: List) -> int:
    """Compute edit distance between two sequences."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def compute_rtf(audio_duration: float, processing_time: float) -> float:
    """
    Compute Real-Time Factor.
    
    RTF < 1 means faster than real-time.
    RTF = 1 means real-time.
    RTF > 1 means slower than real-time.
    
    Args:
        audio_duration: Audio duration in seconds
        processing_time: Time to process in seconds
        
    Returns:
        Real-time factor
    """
    if audio_duration == 0:
        return 0.0
    return processing_time / audio_duration


class RTFMeter:
    """Helper to measure RTF during inference."""
    
    def __init__(self):
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0
        self._start_time: Optional[float] = None
    
    def start(self):
        """Start timing."""
        self._start_time = time.perf_counter()
    
    def stop(self, audio_duration: float):
        """Stop timing and add measurement."""
        if self._start_time is None:
            return
        
        processing_time = time.perf_counter() - self._start_time
        self.total_audio_duration += audio_duration
        self.total_processing_time += processing_time
        self._start_time = None
    
    @property
    def rtf(self) -> float:
        """Get current RTF."""
        return compute_rtf(self.total_audio_duration, self.total_processing_time)
    
    def reset(self):
        """Reset measurements."""
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0
        self._start_time = None
