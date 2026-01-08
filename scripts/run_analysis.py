#!/usr/bin/env python3
"""
Run ACE-Code analysis on code snippets.

Usage:
    python scripts/run_analysis.py --model gemma-2-2b --input examples.txt
    python scripts/run_analysis.py --model codellama-7b --code "def foo(): pass"
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace_code import ACEPipeline
from ace_code.pipeline import PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run ACE-Code analysis on code snippets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-2-2b",
        help="Model name or path (default: gemma-2-2b)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to file containing code snippets (one per line or separated by ---)",
    )
    parser.add_argument(
        "--code",
        type=str,
        help="Direct code snippet to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ace_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--sae-path",
        type=str,
        help="Path to pre-trained SAE checkpoint",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=2,
        help="Number of mutation pairs per snippet",
    )

    args = parser.parse_args()

    # Collect code snippets
    snippets = []

    if args.code:
        snippets.append(args.code)

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)

        content = input_path.read_text()

        # Split by --- or treat each non-empty block as a snippet
        if "---" in content:
            snippets.extend([s.strip() for s in content.split("---") if s.strip()])
        else:
            snippets.append(content.strip())

    if not snippets:
        print("Error: No code snippets provided. Use --code or --input.")
        sys.exit(1)

    print(f"Analyzing {len(snippets)} code snippet(s)...")

    # Configure pipeline
    config = PipelineConfig(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        sae_path=args.sae_path,
    )

    # Initialize and run pipeline
    pipeline = ACEPipeline(config=config)
    pipeline.load()

    results = pipeline.full_analysis(snippets, n_pairs_per_snippet=args.n_pairs)

    # Print results
    print("\n" + "=" * 60)
    print(results.summary())
    print("=" * 60)

    # Save results
    state_path = pipeline.save_state()
    print(f"\nResults saved to: {state_path}")

    # Print detailed findings
    if results.error_features:
        print("\nTop Error Features:")
        for i, ef in enumerate(results.error_features[:5]):
            print(f"  {i+1}. Feature {ef.feature_idx}: t={ef.t_statistic:.2f}")
            if ef.vocab_projection:
                top_tokens = ", ".join(f"'{t}'" for t, _ in ef.vocab_projection[:3])
                print(f"      Top tokens: {top_tokens}")

    if results.critical_heads:
        print("\nCritical Attention Heads:")
        for (layer, head), _ in results.critical_heads[:5]:
            print(f"  Layer {layer}, Head {head}")


if __name__ == "__main__":
    main()
