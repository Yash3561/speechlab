"""
SpeechLab Model Registry

Model architectures for speech recognition.
"""

from typing import Optional, Dict, Any
from enum import Enum
import torch
import torch.nn as nn

from backend.core.logging import logger


class ModelArchitecture(str, Enum):
    """Available model architectures."""
    WHISPER = "whisper"
    WAV2VEC2 = "wav2vec2"
    CONFORMER = "conformer"


def load_whisper_model(variant: str = "tiny", pretrained: bool = True) -> nn.Module:
    """
    Load Whisper model.
    
    Args:
        variant: Model size (tiny, base, small)
        pretrained: Load pretrained weights
        
    Returns:
        Whisper model
    """
    try:
        import whisper
        
        logger.info(f"Loading Whisper {variant} (pretrained={pretrained})")
        model = whisper.load_model(variant)
        
        return model
        
    except ImportError:
        logger.error("whisper package not installed. Run: pip install openai-whisper")
        raise


def load_wav2vec2_model(
    variant: str = "base",
    pretrained: bool = True,
) -> nn.Module:
    """
    Load Wav2Vec2 model from HuggingFace.
    
    Args:
        variant: Model size (base, large)
        pretrained: Load pretrained weights
        
    Returns:
        Wav2Vec2 model
    """
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        
        model_name = f"facebook/wav2vec2-{variant}-960h"
        logger.info(f"Loading Wav2Vec2 from {model_name}")
        
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        return model
        
    except ImportError:
        logger.error("transformers package not installed. Run: pip install transformers")
        raise


class ModelRegistry:
    """
    Central registry for model architectures.
    
    Provides unified interface for loading different model types.
    """
    
    _loaders = {
        ModelArchitecture.WHISPER: load_whisper_model,
        ModelArchitecture.WAV2VEC2: load_wav2vec2_model,
    }
    
    @classmethod
    def load(
        cls,
        architecture: str,
        variant: str = "base",
        pretrained: bool = True,
        **kwargs,
    ) -> nn.Module:
        """
        Load model by architecture name.
        
        Args:
            architecture: Model architecture (whisper, wav2vec2)
            variant: Model variant/size
            pretrained: Load pretrained weights
            **kwargs: Additional arguments for loader
            
        Returns:
            Loaded model
        """
        try:
            arch = ModelArchitecture(architecture.lower())
        except ValueError:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Available: {[a.value for a in ModelArchitecture]}"
            )
        
        loader = cls._loaders.get(arch)
        if loader is None:
            raise NotImplementedError(f"Loader for {architecture} not implemented")
        
        return loader(variant=variant, pretrained=pretrained, **kwargs)
    
    @classmethod
    def list_architectures(cls) -> list:
        """List available architectures."""
        return [a.value for a in ModelArchitecture]
    
    @classmethod
    def get_variants(cls, architecture: str) -> list:
        """Get available variants for an architecture."""
        variants = {
            "whisper": ["tiny", "base", "small", "medium", "large"],
            "wav2vec2": ["base", "large"],
            "conformer": ["small", "medium", "large"],
        }
        return variants.get(architecture.lower(), [])


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict with total, trainable, and frozen params
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
