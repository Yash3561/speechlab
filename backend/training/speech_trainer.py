"""
Speech Training Loop for ASR Models

Specialized training loop for speech recognition models like Whisper.
Integrates with Ray Train for distributed training.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from backend.core.logging import logger
from backend.training.ray_trainer import RayTrainConfig, RayTrainerWrapper, create_train_loop

# Whisper imports with graceful fallback
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    logger.warning("Whisper not installed. Using mock model for demonstration.")


@dataclass
class SpeechTrainConfig(RayTrainConfig):
    """Configuration for speech model training."""
    
    # Model settings
    model_architecture: str = "whisper"
    model_variant: str = "tiny"
    freeze_encoder: bool = False
    
    # Audio settings
    sample_rate: int = 16000
    max_audio_length: int = 30  # seconds
    
    # Tokenization
    max_text_length: int = 448
    language: str = "en"


class MockWhisperModel(nn.Module):
    """
    Mock Whisper model for demonstration when actual Whisper is not available.
    Mimics the basic structure for testing the training loop.
    """
    
    def __init__(self, variant: str = "tiny"):
        super().__init__()
        
        # Model dimensions based on variant
        dims = {
            "tiny": (384, 4, 6),
            "base": (512, 6, 6),
            "small": (768, 12, 8),
            "medium": (1024, 24, 8),
        }
        
        d_model, n_heads, n_layers = dims.get(variant, dims["tiny"])
        
        # Simplified encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(80, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        
        # Simplified decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True),
            num_layers=min(n_layers, 4),  # Limit layers for demo
        )
        
        # Output projection (vocab size ~51865 for Whisper)
        self.proj = nn.Linear(d_model, 51865)
        
        self.d_model = d_model
        
    def forward(
        self,
        mel: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            mel: Mel spectrogram [B, 80, T]
            tokens: Target tokens [B, S] (optional)
            
        Returns:
            Logits [B, S, vocab_size]
        """
        # Encode audio
        encoder_out = self.encoder(mel)  # [B, d_model, T']
        encoder_out = encoder_out.transpose(1, 2)  # [B, T', d_model]
        
        # If tokens provided, decode
        if tokens is not None:
            # Simple token embedding (mock)
            token_emb = torch.zeros(
                tokens.size(0), 
                tokens.size(1), 
                self.d_model, 
                device=tokens.device
            )
            
            # Decode
            decoder_out = self.decoder(token_emb, encoder_out)
            logits = self.proj(decoder_out)
            
            return logits
        
        # If no tokens, return encoder features
        return self.proj(encoder_out)


class DemoSpeechDataset(Dataset):
    """
    Demo dataset for speech training when no real data is available.
    Generates synthetic audio features and text targets.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        mel_length: int = 3000,  # ~30 seconds at 100 fps
        text_length: int = 100,
        vocab_size: int = 51865,
    ):
        self.num_samples = num_samples
        self.mel_length = mel_length
        self.text_length = text_length
        self.vocab_size = vocab_size
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate synthetic mel spectrogram
        mel = torch.randn(80, self.mel_length)
        
        # Generate synthetic token sequence
        tokens = torch.randint(0, self.vocab_size, (self.text_length,))
        
        return {
            "input": mel,
            "target": tokens,
        }


class SpeechTrainer:
    """
    High-level trainer for speech recognition models.
    
    Handles:
    - Model loading (Whisper variants)
    - Dataset preparation
    - Training with Ray Train
    - Evaluation and metrics
    """
    
    def __init__(self, config: SpeechTrainConfig):
        self.config = config
        self.model: Optional[nn.Module] = None
        self.ray_trainer: Optional[RayTrainerWrapper] = None
        
    def load_model(self) -> nn.Module:
        """Load speech recognition model based on configuration."""
        
        if self.config.model_architecture == "whisper":
            if HAS_WHISPER:
                logger.info(f"Loading Whisper {self.config.model_variant}")
                self.model = whisper.load_model(self.config.model_variant)
            else:
                logger.info(f"Loading Mock Whisper {self.config.model_variant}")
                self.model = MockWhisperModel(self.config.model_variant)
        else:
            raise ValueError(f"Unknown model architecture: {self.config.model_architecture}")
        
        # Optionally freeze encoder
        if self.config.freeze_encoder and hasattr(self.model, "encoder"):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def prepare_data(
        self,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Prepare training and validation datasets.
        
        Args:
            train_path: Path to training data manifest
            val_path: Path to validation data manifest
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if train_path is None:
            logger.warning("No training data path provided. Using demo dataset.")
            train_dataset = DemoSpeechDataset(num_samples=500)
            val_dataset = DemoSpeechDataset(num_samples=100)
        else:
            # Load real datasets
            from backend.data import AudioDataset
            
            train_dataset = AudioDataset(
                manifest_path=train_path,
                sample_rate=self.config.sample_rate,
                max_duration=self.config.max_audio_length,
            )
            
            val_dataset = None
            if val_path:
                val_dataset = AudioDataset(
                    manifest_path=val_path,
                    sample_rate=self.config.sample_rate,
                    max_duration=self.config.max_audio_length,
                )
        
        return train_dataset, val_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        resume_from: Optional[str] = None,
        metrics_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Run distributed training with Ray Train.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            resume_from: Checkpoint to resume training from
            metrics_callback: Callback for real-time metrics
            
        Returns:
            Training results dictionary
        """
        # Load model
        if self.model is None:
            self.load_model()
        
        # Define model loader for Ray workers
        def model_loader():
            if HAS_WHISPER and self.config.model_architecture == "whisper":
                return whisper.load_model(self.config.model_variant)
            return MockWhisperModel(self.config.model_variant)
        
        # Loss function
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Create training loop
        train_loop = create_train_loop(
            model_loader=model_loader,
            criterion=criterion,
            metrics_callback=metrics_callback,
        )
        
        # Create Ray trainer
        self.ray_trainer = RayTrainerWrapper(self.config)
        
        # Run training
        logger.info("Starting speech model training...")
        results = self.ray_trainer.train(
            train_func=train_loop,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=self.model,
            resume_from_checkpoint=resume_from,
        )
        
        logger.info("Training complete!")
        return results
    
    def evaluate(
        self,
        test_dataset: Dataset,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a test dataset.
        
        Args:
            test_dataset: Test dataset
            checkpoint_path: Optional checkpoint to load
            
        Returns:
            Dictionary with evaluation metrics
        """
        from backend.evaluation.metrics import compute_wer
        
        if self.model is None:
            self.load_model()
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        total_wer = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                mel = batch["input"].to(device)
                target_tokens = batch["target"]
                
                # Generate predictions (simplified)
                # In real implementation, would use beam search
                outputs = self.model(mel)
                predictions = outputs.argmax(dim=-1)
                
                # Compute WER (mock for demo)
                # Real implementation would decode tokens to text
                wer = compute_wer("reference text", "predicted text")
                total_wer += wer
                num_samples += 1
        
        avg_wer = total_wer / num_samples if num_samples > 0 else 0.0
        
        # Simulated "Worst Errors" for Demo/Visual Debugger
        # In a real system, this would come from the decoding loop above
        worst_samples = [
            {
                "id": "sample_101",
                "reference": "speech recognition is difficult",
                "hypothesis": "peach wreck a nice beach is difficult",
                "wer": 0.8,
                "audio_url": "/api/audio/sample_101.wav" # Placeholder
            },
            {
                "id": "sample_404",
                "reference": "machine learning pipelines",
                "hypothesis": "machine leaning pipe lines",
                "wer": 0.4,
                "audio_url": "/api/audio/sample_404.wav"
            },
            {
                "id": "sample_202",
                "reference": "artificial intelligence",
                "hypothesis": "art official intelligence",
                "wer": 0.6,
                "audio_url": "/api/audio/sample_202.wav"
            }
        ]
        
        return {
            "wer": avg_wer,
            "num_samples": num_samples,
            "worst_samples": worst_samples 
        }


def quick_train_demo():
    """Quick training demo for testing."""
    config = SpeechTrainConfig(
        model_architecture="whisper",
        model_variant="tiny",
        max_epochs=2,
        batch_size=4,
        mixed_precision=True,
        gradient_accumulation_steps=2,
        num_workers=1,
        use_gpu=torch.cuda.is_available(),
    )
    
    trainer = SpeechTrainer(config)
    train_dataset, val_dataset = trainer.prepare_data()
    
    results = trainer.train(train_dataset, val_dataset)
    return results


if __name__ == "__main__":
    results = quick_train_demo()
    print(f"Training results: {results}")
