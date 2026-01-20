#!/usr/bin/env python
"""
SpeechLab Training Script

Run experiments from YAML configuration files.

Usage:
    python scripts/train.py --config configs/experiments/demo_whisper_tiny.yaml
"""

import argparse
from pathlib import Path
import yaml
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.logging import logger
from backend.core.config import settings


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="SpeechLab Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without training",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    exp_name = config["experiment"]["name"]
    
    logger.info(f"=" * 60)
    logger.info(f"SpeechLab Training: {exp_name}")
    logger.info(f"=" * 60)
    logger.info(f"Config: {args.config}")
    
    if args.dry_run:
        logger.info("Dry run - configuration:")
        print(yaml.dump(config, default_flow_style=False))
        return
    
    # Initialize MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "default"))
        logger.info(f"MLflow tracking: {settings.mlflow_tracking_uri}")
    except Exception as e:
        logger.warning(f"MLflow not available: {e}")
    
    # Initialize Ray
    try:
        import ray
        ray.init(address=settings.ray_address, ignore_reinit_error=True)
        logger.info(f"Ray initialized: {ray.cluster_resources()}")
    except Exception as e:
        logger.warning(f"Ray not available, using single-GPU training: {e}")
    
    # Load model
    from backend.training.models import ModelRegistry
    model_config = config["model"]
    model = ModelRegistry.load(
        architecture=model_config["architecture"],
        variant=model_config["variant"],
        pretrained=model_config.get("pretrained", True),
    )
    logger.info(f"Loaded model: {model_config['architecture']}-{model_config['variant']}")
    
    # TODO: Set up data loaders
    # TODO: Initialize trainer
    # TODO: Run training loop
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
