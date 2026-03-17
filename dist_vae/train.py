"""Training loop for the Distribution VAE."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from dist_vae.model import DistributionVAE


class Trainer:
    """Training loop with KL warmup, gradient clipping, and checkpointing.

    Args:
        model: DistributionVAE instance.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        config: Training configuration dictionary with keys matching configs/default.yaml.
    """

    def __init__(
        self,
        model: DistributionVAE,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: dict,
    ) -> None:
        raise NotImplementedError

    def train(self, n_epochs: int | None = None) -> dict[str, list[float]]:
        """Run the training loop.

        Args:
            n_epochs: Number of epochs. If None, uses config value.

        Returns:
            Training history dictionary with keys like 'train_loss', 'val_loss',
            'train_recon', 'train_kl', 'val_recon', 'val_kl', 'lr', 'beta'.
        """
        raise NotImplementedError

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of average training metrics for this epoch.
        """
        raise NotImplementedError

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run validation.

        Returns:
            Dictionary of average validation metrics.
        """
        raise NotImplementedError

    def _save_checkpoint(self, path: str, metrics: dict[str, float]) -> None:
        """Save model checkpoint.

        Args:
            path: File path for the checkpoint.
            metrics: Current metrics to store in the checkpoint.
        """
        raise NotImplementedError
