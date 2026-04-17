"""Perturbation-classifier model for RL training.

Given (NTC phenotype, perturbed phenotype) as K-point quantile tokens per gene,
plus a shared gene-embedding lookup, produces logits over the P perturbations.
The model computes a delta-token per gene (pert - ntc), combines it with the
gene embedding through a per-gene MLP, pools with mean+max across genes, and
scores each perturbation via a learned embedding of its target gene(s).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PerturbationClassifier(nn.Module):
    """Per-gene MLP over (delta-token, gene-embed), mean+max pool, embedding-dot head.

    Args:
        n_all_genes: Size of the shared gene-embedding vocabulary.
        n_perts: Number of perturbations (classes).
        pert_target_gene_ids: For each perturbation, a list of embedding
            indices of its target genes (1 or more for compound perts).
        grid_size: K, the size of each per-gene quantile token.
        d_embed: Gene embedding dim.
        hidden_dim: Per-gene MLP hidden dim.
        d_feat: Per-gene output feature dim.
    """

    def __init__(
        self,
        n_all_genes: int,
        n_perts: int,
        pert_target_gene_ids: list[list[int]],
        grid_size: int = 64,
        d_embed: int = 32,
        hidden_dim: int = 128,
        d_feat: int = 64,
        n_attn_layers: int = 0,
        n_heads: int = 4,
        attn_dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        self.n_all_genes = n_all_genes
        self.n_perts = n_perts
        self.grid_size = grid_size
        self.d_embed = d_embed
        self.d_feat = d_feat
        self.n_attn_layers = n_attn_layers

        self.gene_embed = nn.Embedding(n_all_genes, d_embed)
        nn.init.normal_(self.gene_embed.weight, mean=0.0, std=0.1)

        # Per-gene MLP input: concat(ntc_token, pert_token, delta_token, gene_emb).
        in_dim = 3 * grid_size + d_embed
        self.per_gene_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_feat),
        )

        # Optional transformer encoder over the G per-gene feature tokens.
        # Allows cross-gene attention so the representation of one gene can
        # depend on the state of others (pathway / co-regulation signal).
        if n_attn_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d_feat,
                nhead=n_heads,
                dim_feedforward=attn_dim_feedforward or 4 * d_feat,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.gene_attn = nn.TransformerEncoder(layer, num_layers=n_attn_layers)
        else:
            self.gene_attn = nn.Identity()

        # Global head: 2*d_feat (mean + max pool) -> pert logits.
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_feat),
            nn.Linear(2 * d_feat, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_perts),
        )

        # Padded (P, T_max) tensor of target-gene embedding ids and a mask, kept
        # for reuse in downstream reward / embedding-dot heads if desired.
        t_max = max(len(t) for t in pert_target_gene_ids)
        tgt = torch.zeros(n_perts, t_max, dtype=torch.long)
        mask = torch.zeros(n_perts, t_max, dtype=torch.float32)
        for i, ids in enumerate(pert_target_gene_ids):
            for j, g in enumerate(ids):
                tgt[i, j] = g
                mask[i, j] = 1.0
        self.register_buffer("pert_target_ids", tgt)
        self.register_buffer("pert_target_mask", mask)

    def pert_embeddings(self) -> torch.Tensor:
        """(P, d_embed) perturbation embeddings averaged over target genes."""
        emb = self.gene_embed(self.pert_target_ids)        # (P, T_max, d_embed)
        masked = emb * self.pert_target_mask.unsqueeze(-1)
        summed = masked.sum(dim=1)
        denom = self.pert_target_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return summed / denom

    def forward(
        self,
        ntc_tokens: torch.Tensor,    # (B, G, K)
        pert_tokens: torch.Tensor,   # (B, G, K)
        gene_ids: torch.Tensor,      # (G,) embedding indices shared across batch
    ) -> torch.Tensor:               # logits (B, P)
        B, G, K = pert_tokens.shape
        assert ntc_tokens.shape == pert_tokens.shape
        assert gene_ids.shape == (G,)

        delta = pert_tokens - ntc_tokens               # (B, G, K)
        g_emb = self.gene_embed(gene_ids)              # (G, d_embed)
        g_emb = g_emb.unsqueeze(0).expand(B, G, -1)    # (B, G, d_embed)

        x = torch.cat([ntc_tokens, pert_tokens, delta, g_emb], dim=-1)  # (B, G, 3K+d)
        feats = self.per_gene_mlp(x)                   # (B, G, d_feat)
        feats = self.gene_attn(feats)                  # (B, G, d_feat) - cross-gene attention (or identity)

        mean_feat = feats.mean(dim=1)                  # (B, d_feat)
        max_feat = feats.max(dim=1).values             # (B, d_feat)
        pooled = torch.cat([mean_feat, max_feat], dim=-1)  # (B, 2*d_feat)

        logits = self.head(pooled)                     # (B, P)
        return logits
