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
        self.snapshot_every = log_cfg.get("snapshot_every", 25)

        # Cache a few fixed val examples for reconstruction snapshots
        self._snapshot_grids = self._get_snapshot_examples(val_dataset, n=6)

        # wandb
        self._wandb = None
        if self.use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                pass

    def train(
        self,
        n_epochs: int | None = None,
        epoch_callback: callable | None = None,
    ) -> dict[str, list[float]]:
        """Run the training loop.

        Args:
            n_epochs: Number of epochs. If None, uses config value.
            epoch_callback: Optional callback called after each epoch with
                (epoch, metrics_dict). Useful for hyperparameter optimization
                pruning. The metrics dict contains train_loss, val_loss,
                val_recon, and val_kl.

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

            # Epoch callback (e.g., for Optuna pruning)
            if epoch_callback is not None:
                epoch_callback(epoch, {
                    "train_loss": train_metrics["total"],
                    "val_loss": val_metrics["total"],
                    "val_recon": val_metrics["recon"],
                    "val_kl": val_metrics["kl"],
                })

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

            # Reconstruction snapshots
            if self.snapshot_every > 0 and (
                epoch == 0
                or (epoch + 1) % self.snapshot_every == 0
                or epoch == n_epochs - 1
            ):
                self._save_reconstruction_snapshot(epoch + 1)

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

        # Save training curves
        self._save_training_curves(history)

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

    @staticmethod
    def _get_snapshot_examples(dataset: Dataset, n: int = 6) -> torch.Tensor:
        """Extract a fixed set of examples for reconstruction snapshots."""
        indices = list(range(min(n, len(dataset))))
        grids = torch.stack([dataset[i][0] for i in indices])
        return grids

    @torch.no_grad()
    def _save_reconstruction_snapshot(self, epoch: int) -> None:
        """Save input vs reconstruction plots for fixed val examples."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return

        self.model.eval()
        grids = self._snapshot_grids.to(self.device)
        recon, _, _, _ = self.model(grids)
        grids_np = grids.cpu().numpy()
        recon_np = recon.cpu().numpy()

        n = len(grids_np)
        fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 6), constrained_layout=True)
        x = np.linspace(0, 1, grids_np.shape[1])

        for i in range(n):
            # Top row: overlay quantile functions
            ax = axes[0, i]
            ax.plot(x, grids_np[i], label="input", color="steelblue", linewidth=1.5)
            ax.plot(x, recon_np[i], label="recon", color="coral", linewidth=1.5, linestyle="--")
            ax.set_title(f"Example {i}", fontsize=9)
            ax.set_xlabel("quantile", fontsize=8)
            ax.set_ylabel("value", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7)

            # Bottom row: histogram view (sample from quantile grids)
            ax = axes[1, i]
            ax.hist(grids_np[i], bins=40, density=True, alpha=0.5, color="steelblue", label="input", edgecolor="none")
            ax.hist(recon_np[i], bins=40, density=True, alpha=0.5, color="coral", label="recon", edgecolor="none")
            ax.set_xlabel("value", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_yticks([])
            if i == 0:
                ax.legend(fontsize=7)

        fig.suptitle(f"Reconstruction at epoch {epoch}", fontsize=12)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"recon_epoch_{epoch:04d}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def _save_training_curves(self, history: dict[str, list[float]]) -> None:
        """Save training curve plots to checkpoint directory."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        # Total loss
        ax = axes[0, 0]
        ax.plot(epochs, history["train_loss"], label="train")
        ax.plot(epochs, history["val_loss"], label="val")
        ax.set_ylabel("Total loss")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.set_title("Total loss")

        # Recon loss
        ax = axes[0, 1]
        ax.plot(epochs, history["train_recon"], label="train")
        ax.plot(epochs, history["val_recon"], label="val")
        ax.set_ylabel("Recon loss (Cramer)")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.set_title("Reconstruction loss")

        # KL divergence
        ax = axes[1, 0]
        ax.plot(epochs, history["train_kl"], label="train KL")
        ax.plot(epochs, history["beta"], label="beta", linestyle="--", color="gray")
        ax.set_ylabel("KL / beta")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.set_title("KL divergence & beta warmup")

        # Learning rate
        ax = axes[1, 1]
        ax.plot(epochs, history["lr"])
        ax.set_ylabel("Learning rate")
        ax.set_xlabel("Epoch")
        ax.set_title("Learning rate schedule")

        path = self.checkpoint_dir / "training_curves.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Training curves saved to {path}")

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
