"""
MLflow Integration for Experiment Tracking

Provides a unified interface for logging experiments, metrics, and models to MLflow.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import os

from backend.core.config import settings
from backend.core.logging import logger

# MLflow imports with graceful fallback
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning("MLflow not installed. Experiment tracking disabled.")


class ExperimentTracker:
    """
    MLflow-based experiment tracker for SpeechLab.
    
    Handles:
    - Experiment creation and management
    - Metric logging (loss, WER, etc.)
    - Parameter logging
    - Model artifact storage
    - Run comparison
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "speechlab",
    ):
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self.experiment_name = experiment_name
        self.client: Optional[MlflowClient] = None
        self.experiment_id: Optional[str] = None
        self.active_run: Optional[mlflow.ActiveRun] = None
        
        if HAS_MLFLOW:
            self._initialize()
        else:
            logger.warning("MLflow not available. Using mock tracking.")
    
    def _initialize(self):
        """Initialize MLflow connection and experiment."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.client = MlflowClient(self.tracking_uri)
            
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created MLflow experiment: {self.experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            self.client = None
    
    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> Optional[str]:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this training run
            tags: Optional tags to add
            nested: Whether this is a nested run
            
        Returns:
            Run ID if successful, None otherwise
        """
        if not HAS_MLFLOW or self.client is None:
            logger.info(f"[Mock] Starting run: {run_name}")
            return f"mock_run_{datetime.utcnow().timestamp()}"
        
        try:
            self.active_run = mlflow.start_run(
                run_name=run_name,
                nested=nested,
                tags=tags,
            )
            run_id = self.active_run.info.run_id
            logger.info(f"Started MLflow run: {run_name} ({run_id})")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return None
    
    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        if not HAS_MLFLOW or self.active_run is None:
            logger.info("[Mock] Ending run")
            return
        
        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
            self.active_run = None
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters for the current run."""
        if not HAS_MLFLOW:
            logger.info(f"[Mock] Logging params: {params}")
            return
        
        try:
            # Handle nested dicts by flattening
            flat_params = self._flatten_dict(params)
            mlflow.log_params(flat_params)
        except Exception as e:
            logger.error(f"Failed to log params: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """
        Log metrics for the current run.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number for the metrics
        """
        if not HAS_MLFLOW:
            logger.debug(f"[Mock] Logging metrics at step {step}: {metrics}")
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ):
        """Log a single metric."""
        self.log_metrics({key: value}, step=step)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ):
        """
        Log a model to MLflow.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path within the run's artifacts
            registered_model_name: If provided, register the model
        """
        if not HAS_MLFLOW:
            logger.info(f"[Mock] Logging model to {artifact_path}")
            return
        
        try:
            # Log PyTorch model
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
            )
            logger.info(f"Logged model to MLflow: {artifact_path}")
            
            if registered_model_name:
                logger.info(f"Registered model: {registered_model_name}")
                
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file or directory as an artifact."""
        if not HAS_MLFLOW:
            logger.info(f"[Mock] Logging artifact: {local_path}")
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
    
    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        if not HAS_MLFLOW:
            logger.info(f"[Mock] Setting tag: {key}={value}")
            return
        
        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logger.error(f"Failed to set tag: {e}")
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific run."""
        if not HAS_MLFLOW or self.client is None:
            return None
        
        try:
            run = self.client.get_run(run_id)
            return {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            return None
    
    def list_runs(
        self,
        filter_string: str = "",
        max_results: int = 100,
        order_by: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List runs in the current experiment.
        
        Args:
            filter_string: MLflow filter string
            max_results: Maximum number of runs to return
            order_by: Ordering of results
            
        Returns:
            List of run dictionaries
        """
        if not HAS_MLFLOW or self.client is None:
            return []
        
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by or ["start_time DESC"],
            )
            
            return [
                {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "metrics": run.data.metrics,
                }
                for run in runs
            ]
        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
            return []
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple runs by their metrics.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Specific metrics to compare (all if None)
            
        Returns:
            Dictionary mapping run_id -> metrics
        """
        if not HAS_MLFLOW or self.client is None:
            return {}
        
        comparison = {}
        for run_id in run_ids:
            run_data = self.get_run(run_id)
            if run_data:
                if metrics:
                    comparison[run_id] = {
                        k: v for k, v in run_data["metrics"].items()
                        if k in metrics
                    }
                else:
                    comparison[run_id] = run_data["metrics"]
        
        return comparison
    
    def get_best_run(
        self,
        metric: str = "val_loss",
        ascending: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric to optimize
            ascending: True if lower is better
            
        Returns:
            Best run dictionary
        """
        order = "ASC" if ascending else "DESC"
        runs = self.list_runs(
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )
        
        return runs[0] if runs else None
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow params."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


# ============================================================
# Model Registry
# ============================================================

class ModelRegistry:
    """
    MLflow Model Registry for versioning and deploying models.
    """
    
    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self.client: Optional[MlflowClient] = None
        
        if HAS_MLFLOW:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                self.client = MlflowClient(self.tracking_uri)
            except Exception as e:
                logger.error(f"Failed to initialize Model Registry: {e}")
    
    def register_model(
        self,
        run_id: str,
        artifact_path: str,
        model_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Register a model from a run.
        
        Args:
            run_id: MLflow run ID containing the model
            artifact_path: Path to model within run artifacts
            model_name: Name for the registered model
            tags: Optional tags for the model version
            
        Returns:
            Model version if successful
        """
        if not HAS_MLFLOW or self.client is None:
            logger.info(f"[Mock] Registering model: {model_name}")
            return "1"
        
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            result = mlflow.register_model(model_uri, model_name)
            
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        model_name,
                        result.version,
                        key,
                        value,
                    )
            
            logger.info(f"Registered model {model_name} version {result.version}")
            return result.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,  # "Staging", "Production", "Archived"
    ) -> bool:
        """
        Transition a model version to a new stage.
        
        Args:
            model_name: Name of the registered model
            version: Model version to transition
            stage: Target stage
            
        Returns:
            True if successful
        """
        if not HAS_MLFLOW or self.client is None:
            logger.info(f"[Mock] Transitioning {model_name} v{version} to {stage}")
            return True
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            return True
        except Exception as e:
            logger.error(f"Failed to transition model: {e}")
            return False
    
    def load_model(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Any:
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the registered model
            stage: Stage to load from
            
        Returns:
            Loaded PyTorch model
        """
        if not HAS_MLFLOW:
            logger.warning(f"[Mock] Cannot load model {model_name}")
            return None
        
        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded model {model_name} from {stage}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        if not HAS_MLFLOW or self.client is None:
            return []
        
        try:
            models = self.client.search_registered_models()
            return [
                {
                    "name": model.name,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "status": v.status,
                        }
                        for v in model.latest_versions
                    ],
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


# ============================================================
# Convenience Functions
# ============================================================

def get_tracker(experiment_name: str = "speechlab") -> ExperimentTracker:
    """Get a configured experiment tracker."""
    return ExperimentTracker(experiment_name=experiment_name)


def get_registry() -> ModelRegistry:
    """Get a configured model registry."""
    return ModelRegistry()
