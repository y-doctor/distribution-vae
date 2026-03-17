"""Training loop for the Distribution VAE."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

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
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        train_cfg = config.get("training", {})
        self.batch_size = train_cfg.get("batch_size", 256)
        self.n_epochs = train_cfg.get("epochs", 100)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.seed = train_cfg.get("seed", 42)

        model_cfg = config.get("model", {})
        self.target_beta = model_cfg.get("beta", 0.01)
        self.beta_warmup_epochs = model_cfg.get("beta_warmup_epochs", 10)

        log_cfg = config.get("logging", {})
        self.print_every = log_cfg.get("print_every", 10)
        self.checkpoint_dir = Path(log_cfg.get("checkpoint_dir", "checkpoints/"))
        self.use_wandb = log_cfg.get("wandb", False)

        # Dataloaders
        g = torch.Generator()
        g.manual_seed(self.seed)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, generator=g,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
        )

        # Optimizer and scheduler
        lr = train_cfg.get("lr", 1e-3)
        wd = train_cfg.get("weight_decay", 1e-4)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.n_epochs,
        )

        self.best_val_loss = float("inf")

        # wandb
        self._wandb = None
        if self.use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                pass

    def train(self, n_epochs: int | None = None) -> dict[str, list[float]]:
        """Run the training loop.

        Args:
            n_epochs: Number of epochs. If None, uses config value.

        Returns:
            Training history dictionary.
        """
        if n_epochs is None:
            n_epochs = self.n_epochs

        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_recon": [], "train_kl": [],
            "val_recon": [], "val_kl": [],
            "lr": [], "beta": [],
        }

        for epoch in range(n_epochs):
            # KL warmup
            if self.beta_warmup_epochs > 0 and epoch < self.beta_warmup_epochs:
                self.model.current_beta = self.target_beta * (epoch / self.beta_warmup_epochs)
            else:
                self.model.current_beta = self.target_beta

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate()

            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])
            history["train_recon"].append(train_metrics["recon"])
            history["train_kl"].append(train_metrics["kl"])
            history["val_recon"].append(val_metrics["recon"])
            history["val_kl"].append(val_metrics["kl"])
            history["lr"].append(self.scheduler.get_last_lr()[0])
            history["beta"].append(self.model.current_beta)

            self.scheduler.step()

            # Checkpointing
            if val_metrics["recon"] < self.best_val_loss:
                self.best_val_loss = val_metrics["recon"]
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self._save_checkpoint(
                    str(self.checkpoint_dir / "best.pt"), val_metrics
                )

            # Logging
            if (epoch + 1) % self.print_every == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:4d}/{n_epochs} | "
                    f"train: {train_metrics['total']:.4f} (recon={train_metrics['recon']:.4f}, kl={train_metrics['kl']:.4f}) | "
                    f"val: {val_metrics['total']:.4f} (recon={val_metrics['recon']:.4f}) | "
                    f"beta={self.model.current_beta:.4f} lr={self.scheduler.get_last_lr()[0]:.2e}"
                )

            if self._wandb is not None:
                self._wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["total"],
                    "train/recon": train_metrics["recon"],
                    "train/kl": train_metrics["kl"],
                    "val/loss": val_metrics["total"],
                    "val/recon": val_metrics["recon"],
                    "val/kl": val_metrics["kl"],
                    "lr": self.scheduler.get_last_lr()[0],
                    "beta": self.model.current_beta,
                })

        return history

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_metrics: dict[str, float] = {"total": 0, "recon": 0, "kl": 0}
        n_batches = 0

        for batch in self.train_loader:
            grids = batch[0].to(self.device)

            self.optimizer.zero_grad()
            recon, mu, logvar, z = self.model(grids)
            losses = self.model.compute_loss(grids, recon, mu, logvar)

            losses["total"].backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_metrics["total"] += losses["total"].item()
            total_metrics["recon"] += losses["recon"].item()
            total_metrics["kl"] += losses["kl"].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_metrics: dict[str, float] = {"total": 0, "recon": 0, "kl": 0}
        n_batches = 0

        for batch in self.val_loader:
            grids = batch[0].to(self.device)
            recon, mu, logvar, z = self.model(grids)
            losses = self.model.compute_loss(grids, recon, mu, logvar)

            total_metrics["total"] += losses["total"].item()
            total_metrics["recon"] += losses["recon"].item()
            total_metrics["kl"] += losses["kl"].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}

    def _save_checkpoint(self, path: str, metrics: dict[str, float]) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "grid_size": self.model.grid_size,
            "latent_dim": self.model.latent_dim,
        }, path)
