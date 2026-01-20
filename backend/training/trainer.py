"""
SpeechLab Training Loop

Ray Train integration for distributed speech model training.
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from backend.core.logging import logger
from backend.core.config import settings


@dataclass
class TrainingConfig:
    """Training configuration."""
    max_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    warmup_steps: int = 500
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 1000
    

class SpeechTrainer:
    """
    Speech model trainer with distributed training support.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Gradient clipping
    - Learning rate scheduling
    - Checkpoint saving/resuming
    - Real-time metrics streaming
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics_callback: Optional[Callable] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            config: Training configuration
            optimizer: Custom optimizer (AdamW used if None)
            scheduler: LR scheduler (Cosine used if None)
            metrics_callback: Function called with metrics each step
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or TrainingConfig()
        self.metrics_callback = metrics_callback
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler
        total_steps = len(train_dataloader) * self.config.max_epochs
        self.scheduler = scheduler or torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.mixed_precision else None
        
        # State
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"  - Epochs: {self.config.max_epochs}")
        logger.info(f"  - Batch size: {self.config.batch_size}")
        logger.info(f"  - Mixed precision: {self.config.mixed_precision}")
    
    def train(self) -> Dict[str, float]:
        """
        Run full training loop.
        
        Returns:
            Final metrics dict
        """
        logger.info("Starting training...")
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self._train_epoch()
            
            # Validation
            val_metrics = {}
            if self.val_dataloader:
                val_metrics = self._validate()
            
            # Log epoch metrics
            logger.info(
                f"Epoch {epoch+1}/{self.config.max_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics.get('loss', 0):.4f}"
            )
        
        return {"train": train_metrics, "val": val_metrics}
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move to device
            batch = self._to_device(batch)
            
            # Forward pass with optional AMP
            if self.config.mixed_precision and self.scaler:
                with autocast():
                    loss = self._compute_loss(batch)
                    loss = loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                loss = self._compute_loss(batch)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.mixed_precision and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
                
                # Optimizer step
                if self.config.mixed_precision and self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
                
                # Metrics callback
                if self.metrics_callback and self.global_step % self.config.log_every_n_steps == 0:
                    self.metrics_callback({
                        "epoch": self.current_epoch,
                        "step": self.global_step,
                        "loss": loss.item() * self.config.gradient_accumulation_steps,
                        "lr": self.scheduler.get_last_lr()[0],
                    })
            
            num_batches += 1
        
        return {"loss": total_loss / num_batches}
    
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._to_device(batch)
                
                if self.config.mixed_precision:
                    with autocast():
                        loss = self._compute_loss(batch)
                else:
                    loss = self._compute_loss(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            # TODO: Save checkpoint
        
        return {"loss": avg_loss}
    
    def _compute_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute loss for a batch.
        
        Override this in subclasses for specific loss functions.
        """
        outputs = self.model(**batch)
        return outputs.loss if hasattr(outputs, "loss") else outputs
    
    def _to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")
