"""GRPO trainer for the perturbation-classifier.

Given a policy network that outputs per-class logits over P perturbations,
samples G actions per input, computes a reward per action via cosine
similarity against a precomputed per-perturbation profile table, and updates
the policy with a group-normalized REINFORCE-style advantage (Shao et al.
2024, DeepSeek GRPO).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from dist_vae.losses import cosine_similarity, pearson_correlation


def apply_hinge(
    R: torch.Tensor,
    threshold: torch.Tensor | float,
    rescale: bool = False,
) -> torch.Tensor:
    """Apply a reward hinge.

    If ``rescale`` is False (legacy): zero out entries <= threshold, keep raw
    values above::

        r_eff = R if R > theta else 0

    If ``rescale`` is True: zero out entries <= threshold AND linearly map the
    above-threshold range to [0, 1]::

        r_eff = relu((R - theta) / (1 - theta))

    Rescaling removes the step discontinuity at the threshold (values barely
    above theta map to ~0, not theta) and expands the usable dynamic range so
    group-relative advantages are less squished. Assumes R is bounded above by
    1 (Pearson / cosine); larger thresholds would overflow the [0, 1] target.

    Args:
        R: Reward tensor.
        threshold: Scalar or broadcast-compatible tensor.
        rescale: Whether to linearly rescale above-threshold values to [0, 1].

    Returns:
        Hinged reward tensor, same shape as ``R``.
    """
    if rescale:
        thr = threshold if isinstance(threshold, torch.Tensor) else torch.tensor(
            threshold, dtype=R.dtype, device=R.device,
        )
        denom = (1.0 - thr).clamp_min(1e-6)
        return torch.clamp((R - thr) / denom, min=0.0)
    return torch.where(R > threshold, R, torch.zeros_like(R))


class GRPOTrainer:
    """Group-Relative Policy Optimization trainer.

    Args:
        model: PerturbationClassifier (or any module returning (B, P) logits).
        dataset: PerturbationClassificationDataset yielding
            (ntc_tokens, pert_tokens, pert_idx).
        profiles: Reward-profile tensor of shape (P, n_expression_genes).
        gene_ids: Shared gene-index tensor of shape (G,) passed to forward.
        config: Dict with keys {'rl', 'training', 'logging'}.
        device: torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset,
        profiles: torch.Tensor,
        gene_ids: torch.Tensor,
        config: dict,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.profiles = profiles.to(self.device)
        self.gene_ids = gene_ids.to(self.device)
        self.config = config

        # Precompute the (P, P) pairwise reward table.
        # Row i col j = reward if true was i and we predicted j.
        reward_cfg = config.get("reward", {})
        self.reward_metric = str(reward_cfg.get("metric", "cosine"))
        self.row_normalize = bool(reward_cfg.get("row_normalize", False))
        self.hinge = str(reward_cfg.get("hinge", "none"))   # "none" | "fixed" | "ntc_baseline"
        self.hinge_value = float(reward_cfg.get("hinge_value", 0.0))
        self.hinge_quantile = float(reward_cfg.get("hinge_quantile", 0.95))
        self.hinge_K = int(reward_cfg.get("hinge_K", 200))
        # Multiply the threshold by this factor (e.g. 2.0 = "2x noise floor").
        self.hinge_multiplier = float(reward_cfg.get("hinge_multiplier", 1.0))
        # If True, map the above-threshold range linearly to [0, 1]:
        #   r_eff = relu((r - theta) / (1 - theta))
        # If False, keep the raw reward above threshold (legacy behavior).
        self.hinge_rescale = bool(reward_cfg.get("hinge_rescale", False))

        assert self.reward_metric in ("cosine", "pearson"), self.reward_metric
        metric_fn = pearson_correlation if self.reward_metric == "pearson" else cosine_similarity
        R = metric_fn(
            self.profiles.unsqueeze(1),   # (P, 1, G)
            self.profiles.unsqueeze(0),   # (1, P, G)
            dim=-1,
        )  # (P, P)

        if self.hinge == "ntc_baseline":
            n_cells_hinge = int(reward_cfg.get(
                "hinge_n_cells",
                config.get("data", {}).get("n_cells_per_pert", 100),
            ))
            baseline = dataset.compute_ntc_noise_baseline(
                self.profiles.cpu(),
                n_cells=n_cells_hinge,
                metric=self.reward_metric,
                K=self.hinge_K,
                quantile=self.hinge_quantile,
            )   # (P,)
            self.hinge_threshold = (baseline * self.hinge_multiplier).to(self.device)
            R = self._apply_hinge(R, self.hinge_threshold.unsqueeze(1))
        elif self.hinge == "fixed":
            eff = self.hinge_value * self.hinge_multiplier
            self.hinge_threshold = torch.full(
                (R.shape[0],), eff, device=R.device, dtype=R.dtype,
            )
            R = self._apply_hinge(R, self.hinge_threshold.unsqueeze(1))
        else:
            self.hinge_threshold = None

        if self.row_normalize:
            mu = R.mean(dim=1, keepdim=True)
            sd = R.std(dim=1, keepdim=True).clamp_min(1e-6)
            R = (R - mu) / sd
        self.reward_table = R.to(self.device)

        rl = config["rl"]
        train = config["training"]
        self.group_size = int(rl.get("group_size", 4))
        self.kl_coef = float(rl.get("kl_coef", 0.0))
        self.entropy_coef = float(rl.get("entropy_coef", 0.01))
        self.grad_clip = float(rl.get("grad_clip", 1.0))

        self.batch_size = int(train.get("batch_size", 8))
        self.n_epochs = int(train.get("epochs", 50))

        self.loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(rl.get("lr", 1e-3)),
            weight_decay=float(rl.get("weight_decay", 0.0)),
        )

        # Frozen reference policy snapshot for optional KL regularization.
        self.ref_model = copy.deepcopy(self.model).eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        log = config.get("logging", {})
        self.ckpt_dir = Path(log.get("checkpoint_dir", "checkpoints/rl_perturbation/"))
        self.eval_dir = Path(log.get("eval_dir", "eval_results/rl_perturbation/"))
        self.print_every = int(log.get("print_every", 1))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def _apply_hinge(
        self, R: torch.Tensor, threshold: torch.Tensor
    ) -> torch.Tensor:
        return apply_hinge(R, threshold, rescale=self.hinge_rescale)

    def _compute_reward(
        self, actions: torch.Tensor, true_p: torch.Tensor
    ) -> torch.Tensor:
        """(B, G) -> (B, G) reward from the (optionally row-normalized) table.

        reward[b, g] = reward_table[true_p[b], actions[b, g]]
        """
        return self.reward_table[true_p[:, None], actions]

    def _kl_to_ref(self, logits: torch.Tensor) -> torch.Tensor:
        """KL( current_policy || ref_policy ), scalar."""
        with torch.no_grad():
            ref_logits = self.ref_model(
                self._cur_ntc, self._cur_pert, self.gene_ids
            )
            ref_logp = F.log_softmax(ref_logits, dim=-1)
        cur_logp = F.log_softmax(logits, dim=-1)
        cur_p = cur_logp.exp()
        return (cur_p * (cur_logp - ref_logp)).sum(dim=-1).mean()

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        metrics: dict[str, list[float]] = {
            k: [] for k in (
                "mean_reward",
                "max_reward",
                "pg_loss",
                "entropy",
                "kl_to_ref",
                "grad_norm",
                "top1_acc",
            )
        }
        for ntc, pert, pert_idx in self.loader:
            ntc = ntc.to(self.device)
            pert = pert.to(self.device)
            pert_idx = pert_idx.to(self.device)

            self._cur_ntc, self._cur_pert = ntc, pert  # for optional KL path

            logits = self.model(ntc, pert, self.gene_ids)  # (B, P)
            dist = Categorical(logits=logits)

            actions = dist.sample((self.group_size,)).T    # (B, G)
            log_probs = dist.log_prob(actions.T).T         # (B, G)

            with torch.no_grad():
                rewards = self._compute_reward(actions, pert_idx)  # (B, G)
                adv = rewards - rewards.mean(dim=1, keepdim=True)
                adv = adv / (rewards.std(dim=1, keepdim=True) + 1e-6)

            pg_loss = -(adv * log_probs).mean()
            entropy = dist.entropy().mean()

            loss = pg_loss - self.entropy_coef * entropy
            if self.kl_coef > 0.0:
                kl = self._kl_to_ref(logits)
                loss = loss + self.kl_coef * kl
            else:
                kl = torch.tensor(0.0, device=self.device)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.optimizer.step()

            with torch.no_grad():
                top1 = (logits.argmax(dim=-1) == pert_idx).float().mean()

            metrics["mean_reward"].append(float(rewards.mean().item()))
            metrics["max_reward"].append(float(rewards.max().item()))
            metrics["pg_loss"].append(float(pg_loss.item()))
            metrics["entropy"].append(float(entropy.item()))
            metrics["kl_to_ref"].append(float(kl.item()))
            metrics["grad_norm"].append(float(grad_norm.item()))
            metrics["top1_acc"].append(float(top1.item()))

        return {k: float(sum(v) / max(len(v), 1)) for k, v in metrics.items()}

    def train(self, n_epochs: int | None = None) -> dict[str, list[float]]:
        n_epochs = n_epochs or self.n_epochs
        history: dict[str, list[float]] = {
            k: [] for k in (
                "mean_reward",
                "max_reward",
                "pg_loss",
                "entropy",
                "kl_to_ref",
                "grad_norm",
                "top1_acc",
            )
        }
        best_reward = -float("inf")

        for epoch in range(1, n_epochs + 1):
            epoch_metrics = self._train_epoch(epoch)
            for k, v in epoch_metrics.items():
                history[k].append(v)

            if epoch % self.print_every == 0 or epoch == n_epochs:
                print(
                    f"[ep {epoch:3d}/{n_epochs}] "
                    f"reward={epoch_metrics['mean_reward']:+.4f}  "
                    f"top1={epoch_metrics['top1_acc']:.3f}  "
                    f"entropy={epoch_metrics['entropy']:.3f}  "
                    f"pg_loss={epoch_metrics['pg_loss']:+.4f}  "
                    f"grad_norm={epoch_metrics['grad_norm']:.3f}"
                )

            if epoch_metrics["mean_reward"] > best_reward:
                best_reward = epoch_metrics["mean_reward"]
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "epoch": epoch,
                        "metrics": epoch_metrics,
                        "config": self.config,
                    },
                    self.ckpt_dir / "best.pt",
                )

        with open(self.eval_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        return history
