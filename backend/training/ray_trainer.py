"""
Ray Train Integration for Distributed Training

This module provides the infrastructure for distributed training using Ray Train.
Optimized for GTX 1650 (4GB VRAM) with gradient accumulation and mixed precision.
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.core.logging import logger

# Ray imports with graceful fallback
try:
    import ray
    from ray import train
    from ray.train import ScalingConfig, RunConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer, prepare_model, prepare_data_loader
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    logger.warning("Ray not installed. Distributed training disabled.")


@dataclass
class RayTrainConfig:
    """Configuration for Ray Train distributed training."""
    
    # Cluster settings
    num_workers: int = 1
    use_gpu: bool = True
    resources_per_worker: Dict[str, float] = field(default_factory=lambda: {"GPU": 1})
    
    # Training settings
    max_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Optimization
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3
    
    # Experiment tracking
    experiment_name: str = "speechlab_training"
    run_name: Optional[str] = None


class RayTrainerWrapper:
    """
    Wrapper for Ray Train distributed training.
    
    Handles:
    - GPU-optimized training with mixed precision
    - Gradient accumulation for limited VRAM
    - Checkpointing and resume
    - Real-time metrics reporting
    """
    
    def __init__(self, config: RayTrainConfig):
        self.config = config
        self.trainer: Optional[TorchTrainer] = None
        
        if not HAS_RAY:
            logger.warning("Ray not available. Using local training fallback.")
    
    def train(
        self,
        train_func: Callable,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        model: Optional[nn.Module] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Launch distributed training with Ray Train.
        
        Args:
            train_func: Training function to run on each worker
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            model: PyTorch model (if not created in train_func)
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Dictionary with training results
        """
        if not HAS_RAY:
            return self._train_local(train_func, train_dataset, val_dataset, model)
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(
                runtime_env={"pip": ["torch", "torchaudio"]},
                ignore_reinit_error=True,
            )
        
        # Configure scaling
        scaling_config = ScalingConfig(
            num_workers=self.config.num_workers,
            use_gpu=self.config.use_gpu,
            resources_per_worker=self.config.resources_per_worker,
        )
        
        # Configure checkpointing
        checkpoint_config = CheckpointConfig(
            num_to_keep=self.config.keep_last_n_checkpoints,
        )
        
        # Configure run
        run_config = RunConfig(
            name=self.config.run_name or self.config.experiment_name,
            storage_path=self.config.checkpoint_dir,
            checkpoint_config=checkpoint_config,
        )
        
        # Create trainer
        self.trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config={
                "config": self.config,
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
            },
            scaling_config=scaling_config,
            run_config=run_config,
        )
        
        # Run training
        logger.info(f"Starting Ray Train with {self.config.num_workers} workers")
        result = self.trainer.fit()
        
        return {
            "metrics": result.metrics,
            "checkpoint": result.checkpoint,
            "error": result.error,
        }
    
    def _train_local(
        self,
        train_func: Callable,
        train_dataset: Any,
        val_dataset: Optional[Any],
        model: Optional[nn.Module],
    ) -> Dict[str, Any]:
        """Fallback local training when Ray is not available."""
        logger.info("Running local training (Ray not available)")
        
        config = {
            "config": self.config,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
        }
        
        # Run training function directly
        result = train_func(config)
        
        return {
            "metrics": result if isinstance(result, dict) else {},
            "checkpoint": None,
            "error": None,
        }


def create_train_loop(
    model_loader: Callable[[], nn.Module],
    criterion: nn.Module,
    metrics_callback: Optional[Callable[[Dict], None]] = None,
):
    """
    Create a training loop function for Ray Train.
    
    Args:
        model_loader: Function that returns a model instance
        criterion: Loss function
        metrics_callback: Optional callback for real-time metrics
        
    Returns:
        Training loop function compatible with Ray Train
    """
    
    def train_loop_per_worker(config: Dict[str, Any]):
        """Training loop executed on each Ray worker."""
        train_config: RayTrainConfig = config["config"]
        train_dataset = config["train_dataset"]
        val_dataset = config.get("val_dataset")
        
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        model = model_loader()
        model = model.to(device)
        
        # Prepare for distributed training
        if HAS_RAY:
            model = prepare_model(model)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        
        # Setup learning rate scheduler
        total_steps = train_config.max_epochs * len(train_dataset) // train_config.batch_size
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=train_config.learning_rate * 0.01,
        )
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler(enabled=train_config.mixed_precision)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        
        if HAS_RAY:
            train_loader = prepare_data_loader(train_loader)
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=train_config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            if HAS_RAY:
                val_loader = prepare_data_loader(val_loader)
        
        # Training loop
        global_step = 0
        best_val_loss = float("inf")
        
        for epoch in range(train_config.max_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast(enabled=train_config.mixed_precision):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss = loss / train_config.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % train_config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_config.max_grad_norm,
                    )
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    global_step += 1
                
                epoch_loss += loss.item() * train_config.gradient_accumulation_steps
                num_batches += 1
                
                # Report metrics periodically
                if global_step % 10 == 0 and metrics_callback:
                    metrics_callback({
                        "epoch": epoch + 1,
                        "step": global_step,
                        "train_loss": epoch_loss / num_batches,
                        "learning_rate": scheduler.get_last_lr()[0],
                    })
            
            # Epoch-level metrics
            avg_train_loss = epoch_loss / num_batches
            
            # Validation
            val_loss = None
            if val_loader:
                model.eval()
                total_val_loss = 0.0
                num_val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        inputs = batch["input"].to(device)
                        targets = batch["target"].to(device)
                        
                        with torch.cuda.amp.autocast(enabled=train_config.mixed_precision):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                        
                        total_val_loss += loss.item()
                        num_val_batches += 1
                
                val_loss = total_val_loss / num_val_batches
                
                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            # Report to Ray Train
            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            
            if HAS_RAY:
                # Create checkpoint
                checkpoint_dict = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                }
                
                train.report(
                    metrics=metrics,
                    checkpoint=train.Checkpoint.from_dict(checkpoint_dict),
                )
            
            logger.info(
                f"Epoch {epoch + 1}/{train_config.max_epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f if val_loss else 'N/A'}"
            )
        
        return metrics
    
    return train_loop_per_worker


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler state
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
    }
