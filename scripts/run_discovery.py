#!/usr/bin/env python3
"""
Run SAE Feature Discovery Pipeline

This script runs the offline feature discovery process to identify
"incorrectness" features in the SAE latent space.

Usage:
    python scripts/run_discovery.py --config configs/default.yaml
    python scripts/run_discovery.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace_code.sae_anomaly.pipeline import SAEAnomalyPipeline, PipelineConfig
from ace_code.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAE Feature Discovery for Anomaly Detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name to use",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Target layer for extraction",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of MBPP samples to use (None = all)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top features to select",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="8bit",
        choices=["8bit", "4bit", "none"],
        help="Quantization mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level="INFO")

    logger.info("=" * 60)
    logger.info("SAE Feature Discovery Pipeline")
    logger.info("=" * 60)

    # Create configuration
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = PipelineConfig.from_yaml(args.config)
    else:
        quant = args.quantization if args.quantization != "none" else None
        config = PipelineConfig(
            model_name=args.model,
            target_layer=args.layer,
            device=args.device,
            quantization=quant,
            n_samples=args.n_samples,
            top_k_features=args.top_k,
            output_dir=args.output_dir,
        )

    logger.info(f"Model: {config.model_name}")
    logger.info(f"Target Layer: {config.target_layer}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Output: {config.output_dir}")

    # Create and run pipeline
    pipeline = SAEAnomalyPipeline(config)

    logger.info("Loading model...")
    pipeline.load_model()

    logger.info("Running feature discovery...")
    result = pipeline.run_discovery()

    logger.info("=" * 60)
    logger.info("Discovery Results")
    logger.info("=" * 60)
    logger.info(f"Correct samples: {result.n_correct_samples}")
    logger.info(f"Incorrect samples: {result.n_incorrect_samples}")
    logger.info(f"Total SAE features: {result.n_total_features}")
    logger.info(f"Selected features: {result.n_selected_features}")
    logger.info(f"Top t-statistics: {result.feature_t_statistics[:5]}")

    # Save artifacts
    logger.info("Saving artifacts...")
    pipeline.save()

    logger.info("=" * 60)
    logger.info("Discovery complete!")
    logger.info(f"Artifacts saved to: {config.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
