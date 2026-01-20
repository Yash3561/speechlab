"""
SpeechLab Evaluator

Comprehensive model evaluation with statistical testing.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.core.logging import logger
from backend.core.utils import generate_run_id
from backend.evaluation.metrics import compute_wer, compute_rtf, RTFMeter, TranscriptionMetrics


@dataclass
class TestSetResult:
    """Results for a single test set."""
    name: str
    wer: float
    cer: float
    rtf: float
    num_samples: int
    total_duration_hours: float
    errors: Dict[str, int] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    id: str
    model_id: str
    timestamp: str
    overall_wer: float
    overall_cer: float
    overall_rtf: float
    test_sets: List[TestSetResult]
    config: Dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """
    Speech model evaluator.
    
    Features:
    - Multi-test-set evaluation
    - WER/CER/RTF computation
    - Error breakdown (insertions, deletions, substitutions)
    - Statistical comparison
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run on
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def evaluate(
        self,
        test_sets: Dict[str, DataLoader],
        model_id: str = "unknown",
    ) -> EvaluationResult:
        """
        Evaluate model on multiple test sets.
        
        Args:
            test_sets: Dict mapping test set name to DataLoader
            model_id: Model identifier for logging
            
        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Evaluating model {model_id} on {len(test_sets)} test sets")
        
        all_results: List[TestSetResult] = []
        total_wer = 0.0
        total_cer = 0.0
        total_rtf = 0.0
        total_samples = 0
        
        for name, dataloader in test_sets.items():
            logger.info(f"  Evaluating on {name}...")
            result = self._evaluate_test_set(name, dataloader)
            all_results.append(result)
            
            total_wer += result.wer * result.num_samples
            total_cer += result.cer * result.num_samples
            total_rtf += result.rtf * result.num_samples
            total_samples += result.num_samples
        
        # Compute weighted averages
        overall_wer = total_wer / max(total_samples, 1)
        overall_cer = total_cer / max(total_samples, 1)
        overall_rtf = total_rtf / max(total_samples, 1)
        
        logger.info(f"Overall WER: {overall_wer:.2%} | CER: {overall_cer:.2%} | RTF: {overall_rtf:.3f}")
        
        return EvaluationResult(
            id=generate_run_id("eval"),
            model_id=model_id,
            timestamp=datetime.utcnow().isoformat(),
            overall_wer=overall_wer,
            overall_cer=overall_cer,
            overall_rtf=overall_rtf,
            test_sets=all_results,
        )
    
    def _evaluate_test_set(self, name: str, dataloader: DataLoader) -> TestSetResult:
        """Evaluate on a single test set."""
        rtf_meter = RTFMeter()
        
        all_metrics: List[TranscriptionMetrics] = []
        total_duration = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                waveforms = batch["waveform"].to(self.device)
                references = batch["text"]
                durations = batch["duration"]
                
                # Time inference
                rtf_meter.start()
                hypotheses = self._transcribe(waveforms)
                rtf_meter.stop(sum(durations).item())
                
                # Compute metrics per sample
                for ref, hyp in zip(references, hypotheses):
                    metrics = compute_wer(ref, hyp)
                    all_metrics.append(metrics)
                
                total_duration += sum(durations).item()
        
        # Aggregate
        avg_wer = sum(m.wer for m in all_metrics) / len(all_metrics)
        avg_cer = sum(m.cer for m in all_metrics) / len(all_metrics)
        
        errors = {
            "insertions": sum(m.insertions for m in all_metrics),
            "deletions": sum(m.deletions for m in all_metrics),
            "substitutions": sum(m.substitutions for m in all_metrics),
        }
        
        return TestSetResult(
            name=name,
            wer=avg_wer,
            cer=avg_cer,
            rtf=rtf_meter.rtf,
            num_samples=len(all_metrics),
            total_duration_hours=total_duration / 3600,
            errors=errors,
        )
    
    def _transcribe(self, waveforms: torch.Tensor) -> List[str]:
        """
        Transcribe batch of waveforms.
        
        Override for specific model types.
        """
        # Default: assume model returns logits, decode with argmax
        # This should be overridden for specific models
        outputs = self.model(waveforms)
        
        # Placeholder - actual decoding depends on model
        return ["" for _ in range(waveforms.shape[0])]
    
    def compare(
        self,
        baseline_result: EvaluationResult,
        candidate_result: EvaluationResult,
        threshold: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Compare two models and check for regression.
        
        Args:
            baseline_result: Baseline evaluation
            candidate_result: Candidate evaluation
            threshold: Maximum acceptable WER increase
            
        Returns:
            Comparison results with regression flag
        """
        wer_delta = candidate_result.overall_wer - baseline_result.overall_wer
        cer_delta = candidate_result.overall_cer - baseline_result.overall_cer
        
        regression = wer_delta > threshold
        
        return {
            "baseline_wer": baseline_result.overall_wer,
            "candidate_wer": candidate_result.overall_wer,
            "wer_delta": wer_delta,
            "wer_delta_pct": wer_delta / max(baseline_result.overall_wer, 0.001) * 100,
            "regression_detected": regression,
            "threshold": threshold,
            "cer_delta": cer_delta,
            "recommendation": "REJECT" if regression else "ACCEPT",
        }
