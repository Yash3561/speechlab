"""
Tests for data pipeline.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json

# Test fixtures
@pytest.fixture
def sample_waveform():
    """Create a sample waveform for testing."""
    # 1 second of audio at 16kHz
    return torch.randn(16000)


@pytest.fixture
def sample_manifest(tmp_path):
    """Create a sample manifest file for testing."""
    manifest_path = tmp_path / "manifest.json"
    
    # Create dummy audio file
    audio_path = tmp_path / "sample.wav"
    import torchaudio
    waveform = torch.randn(1, 16000)
    torchaudio.save(str(audio_path), waveform, 16000)
    
    # Write manifest
    entry = {
        "audio_filepath": str(audio_path),
        "text": "hello world",
        "duration": 1.0,
        "speaker_id": "speaker1"
    }
    with open(manifest_path, "w") as f:
        f.write(json.dumps(entry) + "\n")
    
    return manifest_path


class TestFeatureExtractor:
    """Tests for feature extraction."""
    
    def test_mel_spectrogram_shape(self, sample_waveform):
        """Test mel spectrogram output shape."""
        from backend.data.features import FeatureExtractor, FeatureConfig
        
        config = FeatureConfig(feature_type="mel_spectrogram", n_mels=80)
        extractor = FeatureExtractor(config)
        
        features = extractor.extract(sample_waveform)
        
        assert features.dim() == 2
        assert features.shape[0] == 80  # n_mels
        assert features.shape[1] > 0  # time frames
    
    def test_mfcc_shape(self, sample_waveform):
        """Test MFCC output shape."""
        from backend.data.features import FeatureExtractor, FeatureConfig
        
        config = FeatureConfig(feature_type="mfcc", n_mfcc=40)
        extractor = FeatureExtractor(config)
        
        features = extractor.extract(sample_waveform)
        
        assert features.dim() == 2
        assert features.shape[0] == 40  # n_mfcc
    
    def test_normalization(self, sample_waveform):
        """Test feature normalization."""
        from backend.data.features import FeatureExtractor, FeatureConfig
        
        config = FeatureConfig(normalize=True)
        extractor = FeatureExtractor(config)
        
        features = extractor.extract(sample_waveform)
        
        # Mean should be close to 0
        assert abs(features.mean().item()) < 0.1


class TestAugmentation:
    """Tests for audio augmentation."""
    
    def test_spec_augment_shape(self, sample_waveform):
        """Test SpecAugment preserves shape."""
        from backend.data.features import FeatureExtractor
        from backend.data.augmentation import SpecAugment
        
        extractor = FeatureExtractor()
        features = extractor.extract(sample_waveform)
        
        spec_augment = SpecAugment()
        augmented = spec_augment(features)
        
        assert augmented.shape == features.shape
    
    def test_time_shift(self, sample_waveform):
        """Test time shift augmentation."""
        from backend.data.augmentation import AudioAugmenter, AugmentationConfig
        
        config = AugmentationConfig(
            time_shift=True,
            speed_perturb=False,
            spec_augment=False,
        )
        augmenter = AudioAugmenter(config)
        
        augmented = augmenter.augment_waveform(sample_waveform)
        
        assert augmented.shape == sample_waveform.shape


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_wer_identical(self):
        """Test WER for identical strings."""
        from backend.evaluation.metrics import compute_wer
        
        result = compute_wer("hello world", "hello world")
        
        assert result.wer == 0.0
        assert result.cer == 0.0
    
    def test_wer_completely_different(self):
        """Test WER for completely different strings."""
        from backend.evaluation.metrics import compute_wer
        
        result = compute_wer("hello world", "goodbye moon")
        
        assert result.wer == 1.0  # All words wrong
    
    def test_wer_partial_match(self):
        """Test WER for partial match."""
        from backend.evaluation.metrics import compute_wer
        
        result = compute_wer("the cat sat", "the dog sat")
        
        assert result.wer == pytest.approx(1/3, rel=0.01)  # 1 wrong out of 3
        assert result.substitutions == 1
    
    def test_rtf_computation(self):
        """Test RTF computation."""
        from backend.evaluation.metrics import compute_rtf
        
        # Processing 10 seconds of audio in 5 seconds = RTF 0.5
        rtf = compute_rtf(10.0, 5.0)
        
        assert rtf == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
