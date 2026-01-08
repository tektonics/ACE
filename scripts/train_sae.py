#!/usr/bin/env python3
"""
Train SAE on Code Corpus

This script trains a JumpReLU Sparse Autoencoder on residual stream
activations collected from a code corpus.

Usage:
    python scripts/train_sae.py --corpus the_stack_python --n-activations 1000000
    python scripts/train_sae.py --corpus-file code_samples.txt --output sae.pt
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ace_code.sae_anomaly.sidecar import SidecarModel, SidecarConfig
from ace_code.sae_anomaly.trainer import SAETrainer, SAETrainerConfig, ActivationBuffer
from ace_code.sae_anomaly.sae import JumpReLUSAE
from ace_code.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def load_code_corpus(source: str, n_samples: int = 100000):
    """Load code samples from various sources."""

    if source == "the_stack_python":
        try:
            from datasets import load_dataset
            logger.info("Loading The Stack (Python) dataset...")
            dataset = load_dataset(
                "bigcode/the-stack",
                data_dir="data/python",
                split="train",
                streaming=True,
            )
            for i, item in enumerate(dataset):
                if i >= n_samples:
                    break
                yield item["content"][:2048]  # Truncate long files
        except Exception as e:
            logger.error(f"Failed to load The Stack: {e}")
            raise

    elif source == "mbpp":
        from datasets import load_dataset
        logger.info("Loading MBPP dataset...")
        dataset = load_dataset("mbpp", split="train")
        for item in dataset:
            yield item["code"]

    elif Path(source).exists():
        logger.info(f"Loading from file: {source}")
        with open(source) as f:
            for line in f:
                yield line.strip()

    else:
        raise ValueError(f"Unknown corpus source: {source}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train JumpReLU SAE on code activations"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="mbpp",
        help="Corpus source: 'the_stack_python', 'mbpp', or path to file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Sidecar model to use",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Target layer for activation extraction",
    )
    parser.add_argument(
        "--n-activations",
        type=int,
        default=500000,
        help="Number of activations to collect",
    )
    parser.add_argument(
        "--expansion-factor",
        type=int,
        default=8,
        help="SAE expansion factor (8 = 32k latents for 4k model)",
    )
    parser.add_argument(
        "--l1-coef",
        type=float,
        default=5e-3,
        help="L1 sparsity coefficient",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/sae.pt",
        help="Output path for trained SAE",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="8bit",
        choices=["8bit", "4bit", "none"],
        help="Model quantization",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level="INFO")

    logger.info("=" * 60)
    logger.info("SAE Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"N activations: {args.n_activations}")
    logger.info(f"Expansion factor: {args.expansion_factor}")

    # Load sidecar model
    logger.info("Loading sidecar model...")
    quant = args.quantization if args.quantization != "none" else None
    sidecar_config = SidecarConfig(
        model_name=args.model,
        target_layer=args.layer,
        device=args.device,
        quantization=quant,
    )
    sidecar = SidecarModel(sidecar_config).load()

    # Initialize SAE
    logger.info("Initializing SAE...")
    sae = JumpReLUSAE(
        d_model=sidecar.model.model_dim,
        expansion_factor=args.expansion_factor,
        device=args.device,
    )
    logger.info(f"SAE dimensions: {sae.d_model} -> {sae.n_latents}")

    # Collect activations
    logger.info("Collecting activations...")
    buffer = ActivationBuffer(
        sidecar,
        buffer_size=args.n_activations,
        batch_size=32,
    )

    corpus_iter = load_code_corpus(args.corpus, args.n_activations * 2)
    activations = buffer.collect_from_texts(corpus_iter, args.n_activations)

    logger.info(f"Collected {len(activations)} activations")
    logger.info(f"Activation shape: {activations.shape}")

    # Train SAE
    logger.info("Training SAE...")
    trainer_config = SAETrainerConfig(
        expansion_factor=args.expansion_factor,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        l1_coefficient=args.l1_coef,
    )

    trainer = SAETrainer(sae, trainer_config)
    history = trainer.train(activations)

    # Report final metrics
    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    if history["loss/total"]:
        logger.info(f"Final loss: {history['loss/total'][-1]:.4f}")
        logger.info(f"Final MSE: {history['loss/mse'][-1]:.4f}")

    # Test sparsity
    test_batch = activations[:100].to(sae._device)
    stats = sae.get_sparsity_stats(test_batch)
    logger.info(f"Mean active features: {stats['mean_active']:.1f}")
    logger.info(f"Sparsity ratio: {stats['sparsity_ratio']:.4f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sae.save(output_path)
    logger.info(f"Saved SAE to {output_path}")


if __name__ == "__main__":
    main()
