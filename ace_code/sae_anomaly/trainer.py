"""
SAE Training Module

Trains a JumpReLU Sparse Autoencoder on residual stream activations.

Loss Function:
    L(x) = ||x - SAE(x)||²₂ + λ||a(x)||₀

Where:
    - First term: Reconstruction error (MSE)
    - Second term: L0 sparsity penalty (approximated via L1 or straight-through estimator)

Training requires harvesting activations from the model on a large corpus,
then training the SAE to reconstruct those activations with sparse features.
"""

from typing import Optional, Dict, Any, List, Iterator, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from tqdm import tqdm
import numpy as np

from ace_code.sae_anomaly.sae import JumpReLUSAE
from ace_code.sae_anomaly.sidecar import SidecarModel
from ace_code.utils.logging import get_logger
from ace_code.utils.device import get_device

logger = get_logger(__name__)


@dataclass
class SAETrainerConfig:
    """Configuration for SAE training."""

    # Architecture
    expansion_factor: int = 8
    initial_threshold: float = 0.001

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_epochs: int = 1  # Usually 1 epoch over large activation dataset
    warmup_steps: int = 1000

    # Loss coefficients
    l1_coefficient: float = 5e-3  # λ for sparsity penalty
    use_l0_approximation: bool = True  # Use straight-through estimator for L0

    # Regularization
    decoder_norm_constraint: bool = True  # Keep decoder columns unit norm
    gradient_clip: float = 1.0

    # Checkpointing
    save_every_n_steps: int = 10000
    checkpoint_dir: Optional[str] = None

    # Logging
    log_every_n_steps: int = 100


class ActivationBuffer:
    """
    Buffer for collecting and storing model activations.

    Collects activations from the sidecar model and provides
    them as training data for the SAE.
    """

    def __init__(
        self,
        sidecar: SidecarModel,
        buffer_size: int = 500_000,
        batch_size: int = 32,
    ):
        """
        Initialize activation buffer.

        Args:
            sidecar: Loaded sidecar model
            buffer_size: Number of activations to store
            batch_size: Batch size for collection
        """
        self.sidecar = sidecar
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self._buffer: Optional[torch.Tensor] = None
        self._write_idx = 0

    def collect_from_texts(
        self,
        texts: Iterator[str],
        n_activations: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Collect activations from text samples.

        Args:
            texts: Iterator of text samples
            n_activations: Number of activations to collect

        Returns:
            Tensor of activations, shape (n_activations, d_model)
        """
        n_activations = n_activations or self.buffer_size
        d_model = self.sidecar.model.model_dim

        self._buffer = torch.zeros(
            n_activations, d_model,
            dtype=torch.float32,
        )
        self._write_idx = 0

        logger.info(f"Collecting {n_activations} activations...")

        pbar = tqdm(total=n_activations, desc="Collecting activations")
        batch = []

        for text in texts:
            batch.append(text)

            if len(batch) >= self.batch_size:
                self._process_batch(batch)
                pbar.update(len(batch))
                batch = []

                if self._write_idx >= n_activations:
                    break

        # Process remaining
        if batch and self._write_idx < n_activations:
            self._process_batch(batch)
            pbar.update(len(batch))

        pbar.close()

        actual_collected = min(self._write_idx, n_activations)
        logger.info(f"Collected {actual_collected} activations")

        return self._buffer[:actual_collected]

    def _process_batch(self, texts: List[str]) -> None:
        """Process a batch of texts and store activations."""
        activations = self.sidecar.extract_activation(texts, position="final")

        # activations shape: (batch, d_model)
        n_new = activations.size(0)
        end_idx = min(self._write_idx + n_new, self.buffer_size)
        n_to_store = end_idx - self._write_idx

        self._buffer[self._write_idx:end_idx] = activations[:n_to_store].cpu()
        self._write_idx = end_idx


class ActivationDataset(IterableDataset):
    """
    Iterable dataset for streaming activations during training.

    Used when activations are too large to fit in memory.
    """

    def __init__(
        self,
        sidecar: SidecarModel,
        text_iterator: Iterator[str],
        buffer_size: int = 10000,
    ):
        self.sidecar = sidecar
        self.text_iterator = text_iterator
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = ActivationBuffer(self.sidecar, self.buffer_size)
        while True:
            try:
                activations = buffer.collect_from_texts(
                    self.text_iterator,
                    n_activations=self.buffer_size,
                )
                for i in range(len(activations)):
                    yield activations[i]
            except StopIteration:
                break


class SAETrainer:
    """
    Trainer for JumpReLU Sparse Autoencoders.

    Trains the SAE to minimize:
        L(x) = ||x - SAE(x)||²₂ + λ||a(x)||₀

    The L0 norm is approximated using either:
    1. L1 penalty (smoother, more common)
    2. Straight-through estimator (closer to true L0)
    """

    def __init__(
        self,
        sae: JumpReLUSAE,
        config: Optional[SAETrainerConfig] = None,
    ):
        """
        Initialize SAE trainer.

        Args:
            sae: JumpReLU SAE to train
            config: Training configuration
        """
        self.sae = sae
        self.config = config or SAETrainerConfig()
        self.device = sae._device

        # Optimizer
        self.optimizer = optim.Adam(
            sae.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler with warmup
        self.scheduler = None  # Set in train()

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')

        logger.info(
            f"Initialized SAETrainer: lr={self.config.learning_rate}, "
            f"l1_coef={self.config.l1_coefficient}"
        )

    def _compute_loss(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.

        L(x) = ||x - SAE(x)||²₂ + λ||a(x)||₀

        Args:
            x: Input activations, shape (batch, d_model)

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Forward pass
        x_hat, features = self.sae(x, return_features=True)

        # Reconstruction loss (MSE)
        mse_loss = torch.mean((x - x_hat) ** 2)

        # Sparsity loss
        if self.config.use_l0_approximation:
            # L0 approximation via straight-through estimator
            # Count non-zero activations
            l0_loss = (features > 0).float().sum(dim=-1).mean()
            sparsity_loss = l0_loss / self.sae.n_latents  # Normalize
        else:
            # L1 penalty (smoother approximation)
            sparsity_loss = features.abs().mean()

        # Total loss
        total_loss = mse_loss + self.config.l1_coefficient * sparsity_loss

        # Metrics
        with torch.no_grad():
            n_active = (features > 0).float().sum(dim=-1).mean().item()
            dead_features = ((features > 0).float().sum(dim=0) == 0).sum().item()

        metrics = {
            "loss/total": total_loss.item(),
            "loss/mse": mse_loss.item(),
            "loss/sparsity": sparsity_loss.item(),
            "features/n_active": n_active,
            "features/dead": dead_features,
            "features/dead_pct": dead_features / self.sae.n_latents * 100,
        }

        return total_loss, metrics

    def _normalize_decoder(self) -> None:
        """Normalize decoder columns to unit norm."""
        if self.config.decoder_norm_constraint:
            with torch.no_grad():
                norms = self.sae.W_dec.norm(dim=1, keepdim=True)
                self.sae.W_dec.div_(norms.clamp(min=1e-8))

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Batch of activations, shape (batch_size, d_model)

        Returns:
            Dictionary of metrics
        """
        self.sae.train()
        batch = batch.to(self.device)

        # Forward and loss
        self.optimizer.zero_grad()
        loss, metrics = self._compute_loss(batch)

        # Backward
        loss.backward()

        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.sae.parameters(),
                self.config.gradient_clip,
            )

        # Update
        self.optimizer.step()

        # Normalize decoder
        self._normalize_decoder()

        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
            metrics["lr"] = self.scheduler.get_last_lr()[0]

        self.global_step += 1

        return metrics

    def train(
        self,
        activations: torch.Tensor,
        n_epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Train SAE on collected activations.

        Args:
            activations: Activation tensor, shape (n_samples, d_model)
            n_epochs: Number of epochs (defaults to config)

        Returns:
            Dictionary of training history
        """
        n_epochs = n_epochs or self.config.n_epochs

        # Create dataloader
        dataset = TensorDataset(activations)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        n_steps = len(dataloader) * n_epochs

        # Setup scheduler with warmup
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            return 1.0

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        logger.info(
            f"Training SAE for {n_epochs} epochs, "
            f"{len(dataloader)} steps/epoch, "
            f"{n_steps} total steps"
        )

        history = {
            "loss/total": [],
            "loss/mse": [],
            "loss/sparsity": [],
            "features/n_active": [],
        }

        for epoch in range(n_epochs):
            epoch_metrics = {k: [] for k in history.keys()}

            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{n_epochs}",
            )

            for batch_tuple in pbar:
                batch = batch_tuple[0]
                metrics = self.train_step(batch)

                # Accumulate metrics
                for k in epoch_metrics:
                    if k in metrics:
                        epoch_metrics[k].append(metrics[k])

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['loss/total']:.4f}",
                    "mse": f"{metrics['loss/mse']:.4f}",
                    "active": f"{metrics['features/n_active']:.1f}",
                })

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    logger.debug(
                        f"Step {self.global_step}: "
                        f"loss={metrics['loss/total']:.4f}, "
                        f"mse={metrics['loss/mse']:.4f}, "
                        f"n_active={metrics['features/n_active']:.1f}"
                    )

                # Checkpointing
                if (self.config.checkpoint_dir and
                    self.global_step % self.config.save_every_n_steps == 0):
                    self._save_checkpoint()

            # Epoch summary
            for k, v in epoch_metrics.items():
                if v:
                    avg = sum(v) / len(v)
                    history[k].append(avg)
                    logger.info(f"Epoch {epoch + 1} - {k}: {avg:.4f}")

        # Final save
        if self.config.checkpoint_dir:
            self._save_checkpoint(final=True)

        return history

    def train_streaming(
        self,
        sidecar: SidecarModel,
        text_iterator: Iterator[str],
        n_steps: int,
    ) -> Dict[str, List[float]]:
        """
        Train SAE with streaming activations.

        Use this when activations are too large to fit in memory.

        Args:
            sidecar: Sidecar model for collecting activations
            text_iterator: Iterator of text samples
            n_steps: Number of training steps

        Returns:
            Dictionary of training history
        """
        buffer = ActivationBuffer(
            sidecar,
            buffer_size=self.config.batch_size * 100,
            batch_size=32,
        )

        history = {
            "loss/total": [],
            "loss/mse": [],
        }

        logger.info(f"Training SAE for {n_steps} steps (streaming)")

        pbar = tqdm(total=n_steps, desc="Training")
        texts_batch = []

        for text in text_iterator:
            texts_batch.append(text)

            if len(texts_batch) >= self.config.batch_size:
                # Get activations for batch
                activations = sidecar.extract_activation(
                    texts_batch, position="final"
                )

                # Train step
                metrics = self.train_step(activations)

                history["loss/total"].append(metrics["loss/total"])
                history["loss/mse"].append(metrics["loss/mse"])

                pbar.update(1)
                pbar.set_postfix({"loss": f"{metrics['loss/total']:.4f}"})

                texts_batch = []

                if self.global_step >= n_steps:
                    break

        pbar.close()
        return history

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if final:
            path = checkpoint_dir / "sae_final.pt"
        else:
            path = checkpoint_dir / f"sae_step_{self.global_step}.pt"

        self.sae.save(path)
        logger.info(f"Saved checkpoint to {path}")


def train_sae_on_code_corpus(
    sidecar: SidecarModel,
    code_texts: List[str],
    config: Optional[SAETrainerConfig] = None,
    output_path: Optional[str] = None,
) -> JumpReLUSAE:
    """
    Convenience function to train SAE on a code corpus.

    Args:
        sidecar: Loaded sidecar model
        code_texts: List of code/text samples
        config: Training configuration
        output_path: Path to save trained SAE

    Returns:
        Trained JumpReLUSAE
    """
    config = config or SAETrainerConfig()

    # Initialize SAE
    sae = JumpReLUSAE(
        d_model=sidecar.model.model_dim,
        expansion_factor=config.expansion_factor,
        initial_threshold=config.initial_threshold,
        device=str(sidecar.model.device),
    )

    # Collect activations
    buffer = ActivationBuffer(sidecar)
    activations = buffer.collect_from_texts(iter(code_texts))

    # Train
    trainer = SAETrainer(sae, config)
    trainer.train(activations)

    # Save
    if output_path:
        sae.save(output_path)

    return sae
