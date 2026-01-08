#!/usr/bin/env python3
"""
Run SAE-Based Anomaly Detection

This script loads a trained pipeline and runs anomaly detection
on input prompts or code samples.

Usage:
    python scripts/run_detection.py --artifacts artifacts/
    python scripts/run_detection.py --artifacts artifacts/ --input "def sort(lst):"
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace_code.sae_anomaly.pipeline import SAEAnomalyPipeline
from ace_code.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAE-Based Anomaly Detection"
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        required=True,
        help="Path to pipeline artifacts directory",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input text to analyze (or use --file)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File containing inputs (one per line)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom anomaly threshold",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed feature information",
    )
    return parser.parse_args()


def analyze_input(pipeline, text, verbose=False):
    """Analyze a single input and display results."""
    result = pipeline.detect(text)

    print("\n" + "=" * 60)
    print("Detection Result")
    print("=" * 60)
    print(f"Input: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Is Anomaly: {result.is_anomaly}")
    print(f"Anomaly Score: {result.anomaly_score:.4f}")
    print(f"Anomaly Level: {result.anomaly_level.value}")
    print(f"Threshold: {result.threshold}")
    print(f"Active Features: {result.n_active_features}")

    if verbose and result.top_features:
        print("\nTop Contributing Features:")
        for feat in result.top_features[:5]:
            print(f"  Feature {feat['feature_index']}: "
                  f"activation={feat['activation']:.4f}, "
                  f"contribution={feat['contribution']:.4f}, "
                  f"t-stat={feat['t_statistic']:.2f}")

    if result.is_anomaly:
        print("\n⚠️  ANOMALY DETECTED - Consider routing to formal verification")
    else:
        print("\n✓ No anomaly detected")

    return result


def main():
    args = parse_args()
    setup_logging(level="INFO")

    logger.info("Loading pipeline from artifacts...")
    pipeline = SAEAnomalyPipeline.load(args.artifacts)

    if args.threshold:
        pipeline.detector.config.anomaly_threshold = args.threshold
        logger.info(f"Using custom threshold: {args.threshold}")

    # Get inputs
    inputs = []
    if args.input:
        inputs.append(args.input)
    elif args.file:
        with open(args.file) as f:
            inputs = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("Enter prompts to analyze (Ctrl+D to exit):")
        try:
            while True:
                text = input("> ")
                if text.strip():
                    analyze_input(pipeline, text.strip(), args.verbose)
        except EOFError:
            print("\nExiting.")
            return

    # Process inputs
    results = []
    for text in inputs:
        result = analyze_input(pipeline, text, args.verbose)
        results.append(result)

    # Summary
    if len(results) > 1:
        summary = pipeline.detector.get_detection_summary(results)
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total inputs: {summary['n_total']}")
        print(f"Anomalies detected: {summary['n_anomalies']}")
        print(f"Anomaly rate: {summary['anomaly_rate']:.2%}")
        print(f"Mean score: {summary['mean_score']:.4f}")


if __name__ == "__main__":
    main()
