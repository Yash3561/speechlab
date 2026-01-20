
import pytest
from backend.evaluation.regression import detect_regression

def test_detect_regression_worse():
    # WER increased (Worse)
    candidate = {"wer": 15.0}
    baseline = {"wer": 10.0}
    
    report = detect_regression(candidate, baseline, primary_metric="wer", lower_is_better=True)
    
    assert report.is_regression is True
    assert report.is_improvement is False
    assert report.relative_diff == 50.0 # (15-10)/10 * 100
    assert report.severity == "critical"

def test_detect_regression_better():
    # WER decreased (Better)
    candidate = {"wer": 5.0}
    baseline = {"wer": 10.0}
    
    report = detect_regression(candidate, baseline, primary_metric="wer", lower_is_better=True)
    
    assert report.is_regression is False
    assert report.is_improvement is True
    assert report.relative_diff == -50.0

def test_detect_regression_noise():
    # Tiny change (Noise)
    candidate = {"wer": 10.05}
    baseline = {"wer": 10.0}
    
    report = detect_regression(candidate, baseline, primary_metric="wer", lower_is_better=True, threshold_percent=1.0)
    
    assert report.is_regression is False
    assert report.is_improvement is False
    assert report.severity == "none"

def test_missing_metrics():
    candidate = {"cer": 5.0}
    baseline = {"wer": 10.0}
    
    report = detect_regression(candidate, baseline, primary_metric="wer")
    assert report is None
