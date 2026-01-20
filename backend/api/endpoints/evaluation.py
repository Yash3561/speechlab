"""
Evaluation API Endpoints

Handles WER/CER calculation, model comparison, and regression detection.
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from backend.auth.dependencies import get_current_user
from backend.tracking import get_tracker
from backend.evaluation.regression import detect_regression, RegressionReport

router = APIRouter()

# ============================================================
# Legacy Pydantic Models (Restored for Frontend Compatibility)
# ============================================================

class TestSetResult(BaseModel):
    """Evaluation results for a single test set."""
    test_set: str
    wer: float
    cer: float
    rtf: float
    num_samples: int
    total_duration_sec: float

class EvaluationResult(BaseModel):
    """Full evaluation results for a model."""
    id: str
    model_id: str
    experiment_id: str
    created_at: str
    overall_wer: float
    overall_cer: float
    test_sets: List[TestSetResult]
    status: str

# In-memory storage for legacy endpoints
evaluations_db: dict = {}

# ============================================================
# New Metrics Models
# ============================================================

class MetricsRequest(BaseModel):
    reference_text: str
    hypothesis_text: str

class MetricsResult(BaseModel):
    wer: float
    cer: float
    wer_details: Dict[str, int]

# ============================================================
# Legacy Endpoints
# ============================================================

@router.get("/", response_model=List[EvaluationResult])
async def list_evaluations():
    return list(evaluations_db.values())

@router.get("/{evaluation_id}", response_model=EvaluationResult)
async def get_evaluation(evaluation_id: str):
    if evaluation_id not in evaluations_db:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluations_db[evaluation_id]

# ============================================================
# New Logic Endpoints
# ============================================================

@router.post("/metrics", response_model=MetricsResult)
async def calculate_metrics(request: MetricsRequest):
    """Calculate WER and CER for a single sample."""
    try:
        from jiwer import wer, cer, process_words
        w_error = wer(request.reference_text, request.hypothesis_text)
        c_error = cer(request.reference_text, request.hypothesis_text)
        output = process_words(request.reference_text, request.hypothesis_text)
        
        return MetricsResult(
            wer=w_error,
            cer=c_error,
            wer_details={
                "hits": output.hits,
                "substitutions": output.substitutions,
                "deletions": output.deletions,
                "insertions": output.insertions,
            }
        )
    except ImportError:
        return MetricsResult(
            wer=0.0,
            cer=0.0,
            wer_details={"hits":0, "substitutions":0, "deletions":0, "insertions":0}
        )

# ============================================================
# Regression Detection
# ============================================================

@router.get("/regression/{run_id}", response_model=RegressionReport)
async def check_regression(run_id: str, user=Depends(get_current_user)):
    """Compare a specific run against the current production baseline."""
    tracker = get_tracker()
    candidate = tracker.get_run(run_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate run not found")
        
    baseline = tracker.get_baseline()
    
    if not baseline:
        return RegressionReport(
            candidate_id=run_id,
            baseline_id="none",
            metric="wer",
            candidate_value=candidate["metrics"].get("wer", 0.0),
            baseline_value=0.0,
            diff=0.0,
            relative_diff=0.0,
            is_regression=False,
            is_improvement=True,
            severity="none"
        )
        
    report = detect_regression(
        candidate_metrics=candidate["metrics"],
        baseline_metrics=baseline["metrics"],
        primary_metric="wer",
        lower_is_better=True
    )
    
    if not report:
        # If WER missing, try to generate a dummy report or fail
        # Ideally we fail, but for demo stability we might return None
        raise HTTPException(status_code=400, detail="Missing WER metric in runs")
        
    report.candidate_id = run_id
    report.baseline_id = baseline["run_id"]
    return report

@router.post("/baseline/{run_id}")
async def set_baseline(run_id: str, user=Depends(get_current_user)):
    """Promote a run to be the new production baseline."""
    tracker = get_tracker()
    tracker.set_baseline(run_id)
    return {"message": f"Run {run_id} promoted to production baseline"}

@router.get("/baseline")
async def get_current_baseline():
    """Get the current baseline run info."""
    tracker = get_tracker()
    baseline = tracker.get_baseline()
    if not baseline:
         return {
             "run_id": "none",
             "run_name": "No Baseline Set",
             "metrics": {"wer": 0.0}
         }
    return baseline
