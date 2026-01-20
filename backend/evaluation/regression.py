"""
Regression Detection Logic for Speech Models.

This module handles:
1. Comparing candidate model metrcis against baseline
2. Detecting significant regressions vs noise
3. Formatting regression reports
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class RegressionReport(BaseModel):
    """Report comparing candidate run against baseline."""
    candidate_id: str
    baseline_id: str
    metric: str = "wer"
    
    candidate_value: float
    baseline_value: float
    
    diff: float          # absolute difference
    relative_diff: float # percentage difference
    
    is_regression: bool  # True if candidate is worse
    is_improvement: bool # True if candidate is better
    severity: str        # "critical", "minor", "none"
    
    timestamp: datetime = datetime.utcnow()


def detect_regression(
    candidate_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    primary_metric: str = "wer",
    lower_is_better: bool = True,
    threshold_percent: float = 1.0 # Significant change threshold
) -> Optional[RegressionReport]:
    """
    Compare candidate metrics to baseline to detect regressions.
    Returns None if metric is missing.
    """
    
    c_val = candidate_metrics.get(primary_metric)
    b_val = baseline_metrics.get(primary_metric)
    
    if c_val is None or b_val is None:
        return None
        
    # Calculate difference
    diff = c_val - b_val
    
    # Avoid division by zero
    if b_val == 0:
        relative_diff = 0.0 if c_val == 0 else float('inf')
    else:
        relative_diff = (diff / b_val) * 100.0
        
    # Determine status
    is_worse = False
    is_better = False
    
    if lower_is_better:
        # For WER: Higher is worse +, Lower is better -
        if relative_diff > threshold_percent: # e.g. +5% WER
            is_worse = True
        elif relative_diff < -threshold_percent: # e.g. -5% WER
            is_better = True
    else:
        # For Accuracy: Lower is worse -, Higher is better +
        if relative_diff < -threshold_percent:
            is_worse = True
        elif relative_diff > threshold_percent:
            is_better = True
            
    # Determine severity
    severity = "none"
    if is_worse:
        if abs(relative_diff) > 10.0:
            severity = "critical"
        elif abs(relative_diff) > 1.0:
            severity = "minor"
            
    return RegressionReport(
        candidate_id="unknown", # caller fills this
        baseline_id="unknown",  # caller fills this
        metric=primary_metric,
        candidate_value=round(c_val, 4),
        baseline_value=round(b_val, 4),
        diff=round(diff, 4),
        relative_diff=round(relative_diff, 2),
        is_regression=is_worse,
        is_improvement=is_better,
        severity=severity
    )
