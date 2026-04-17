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

from dist_vae.losses import cosine_similarity


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

        # Precompute the (P, P) pairwise reward table so we can optionally
        # row-normalize before the trainer uses it. Row i col j = reward if
        # true was i and we predicted j, i.e. cos-sim(profiles[i], profiles[j]).
        reward_cfg = config.get("reward", {})
        self.row_normalize = bool(reward_cfg.get("row_normalize", False))
        R = cosine_similarity(
            self.profiles.unsqueeze(1),   # (P, 1, G)
            self.profiles.unsqueeze(0),   # (1, P, G)
            dim=-1,
        )  # (P, P)
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
