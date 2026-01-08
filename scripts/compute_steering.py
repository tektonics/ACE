#!/usr/bin/env python3
"""
Compute and save steering vectors for code correction.

Usage:
    python scripts/compute_steering.py --model gemma-2-2b --layer 12 \
        --positive "def foo(x: int) -> int:" --negative "def foo(x):"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ace_code import load_hooked_model
from ace_code.intervention.steering import SteeringVector


def main():
    parser = argparse.ArgumentParser(
        description="Compute steering vectors from positive/negative prompts"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-2-2b",
        help="Model name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--positive",
        type=str,
        required=True,
        help="Positive (correct) prompt",
    )
    parser.add_argument(
        "--negative",
        type=str,
        required=True,
        help="Negative (buggy) prompt",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Layer to compute steering vector for",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="steering_vector.pt",
        help="Output file path",
    )

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_hooked_model(args.model, device=args.device)

    print(f"Computing steering vector at layer {args.layer}...")
    sv = SteeringVector.from_prompts(
        model=model,
        positive_prompt=args.positive,
        negative_prompt=args.negative,
        layer=args.layer,
    )

    print(f"Steering vector norm: {sv.norm:.4f}")

    # Save
    sv.save(args.output)
    print(f"Saved to: {args.output}")

    # Print vocabulary projection of the direction
    from ace_code.detection.feature_analysis import FeatureAnalyzer

    analyzer = FeatureAnalyzer(model)
    proj = analyzer.project_to_vocab(direction=sv.vector, top_k=10)

    print("\nSteering vector predicts:")
    print("  Top tokens:", ", ".join(f"'{t}':{s:.2f}" for t, s in proj.top_tokens[:5]))
    print("  Bottom tokens:", ", ".join(f"'{t}':{s:.2f}" for t, s in proj.bottom_tokens[:5]))


if __name__ == "__main__":
    main()
