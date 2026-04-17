"""Per-cell set-transformer classifier with learned gene modules.

See labbook/DECISIONS.md [2026-04-17] "Per-cell set-transformer classifier
with learned gene modules" for the full rationale.

Shapes follow this convention throughout:
  B      : batch of (pert, NTC) set pairs
  n_p    : cells per pert set
  n_n    : cells per NTC set
  G      : number of expression genes
  d      : shared model dim
  K      : number of learned gene modules
  P      : number of perturbations (classes)

Signatures match the existing GRPO trainer: the model's ``forward`` accepts
``(ntc, pert, gene_ids)`` and returns ``(B, P)`` logits — the trainer does
not know or care whether ``ntc``/``pert`` are quantile tokens or raw cells.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeneModuleAttention(nn.Module):
    """Map per-cell (G,) expression → (K, d) module activations via attention.

    Each gene gets a shared embedding ``gene_embed[g] ∈ ℝ^d``. For a cell
    with expression ``x ∈ ℝ^G``, gene tokens are ``token[g] = x[g] *
    gene_embed[g]`` — a simple multiplicative modulation that scales each
    gene's embedding by how strongly that gene is expressed in this cell.
    Then K learned module queries cross-attend into those G tokens; the
    output is a ``(K, d)`` module fingerprint per cell.

    Shared across all cells (pert or NTC), so "module 1 = cell-cycle" etc.
    is a global vocabulary. Readable from the attention weights.

    Args:
        n_all_genes: Size of the shared gene embedding vocabulary.
        d: Shared hidden dim.
        n_modules: K, number of learned module queries.
        n_heads: Number of attention heads.
        gene_embed: Optional existing embedding to share across modules
            (e.g. with a separate pert-target-embedding head). If None,
            a fresh ``nn.Embedding`` is created.
    """

    def __init__(
        self,
        n_all_genes: int,
        d: int,
        n_modules: int,
        n_heads: int = 4,
        gene_embed: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.n_all_genes = n_all_genes
        self.d = d
        self.n_modules = n_modules

        if gene_embed is None:
            self.gene_embed = nn.Embedding(n_all_genes, d)
            nn.init.normal_(self.gene_embed.weight, mean=0.0, std=0.1)
        else:
            assert gene_embed.embedding_dim == d
            self.gene_embed = gene_embed

        # K learned module queries, shared across all cells.
        self.module_queries = nn.Parameter(torch.randn(n_modules, d) * 0.1)

        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm_tok = nn.LayerNorm(d)
        self.norm_out = nn.LayerNorm(d)

    def forward(
        self,
        expression: torch.Tensor,   # (..., G)
        gene_ids: torch.Tensor,     # (G,)
    ) -> torch.Tensor:              # (..., K, d)
        """Return (K, d) module activations per leading-dim position.

        ``expression`` may have any number of leading batch dims — e.g.
        ``(B, n_cells, G)`` works and returns ``(B, n_cells, K, d)``.
        """
        *lead, G = expression.shape
        assert gene_ids.shape == (G,), (
            f"expected gene_ids shape ({G},), got {tuple(gene_ids.shape)}"
        )

        g_emb = self.gene_embed(gene_ids)                      # (G, d)
        tok = expression.unsqueeze(-1) * g_emb                 # (..., G, d)
        tok = self.norm_tok(tok)

        B = int(torch.tensor(lead).prod().item()) if lead else 1
        tok_flat = tok.reshape(B, G, self.d)
        q = self.module_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)
        out, _ = self.attn(q, tok_flat, tok_flat, need_weights=False)  # (B, K, d)
        out = self.norm_out(out)

        return out.reshape(*lead, self.n_modules, self.d)


class CellSetTransformer(nn.Module):
    """Self-attention over a set of ``n_cells`` tokens of dim ``d``.

    Wraps a standard ``nn.TransformerEncoder`` with ``batch_first=True`` and
    pre-norm. Cells are exchangeable (no positional encoding) — the model is
    treating them as a permutation-invariant set.

    Args:
        d: Token dim.
        n_heads: Attention heads.
        n_layers: Number of encoder layers.
        dim_feedforward: MLP hidden dim; defaults to ``4 * d``.
        dropout: Dropout inside the encoder block.
    """

    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=dim_feedforward or 4 * d,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_cells, d) → (B, n_cells, d)."""
        return self.encoder(x)


class _CrossAttnBlock(nn.Module):
    """One pre-norm cross-attention block: (pert, ntc) → updated pert.

    A standard transformer decoder layer is overkill (it has a self-attn
    sub-layer we don't need here). Build a tight version: cross-attn,
    residual, norm, FFN, residual, norm — all with pre-norm convention.
    """

    def __init__(self, d: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False
        )
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return q


class PertNTCCrossAttention(nn.Module):
    """Stack of cross-attention blocks where pert cells attend to NTC cells.

    Gives each pert cell a per-cell "what does the baseline look like for
    me?" context instead of a global NTC summary. Stack depth controls how
    deeply the pert representation is refined against the control baseline.
    """

    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dff = dim_feedforward or 4 * d
        self.blocks = nn.ModuleList([
            _CrossAttnBlock(d, n_heads, dff, dropout) for _ in range(n_layers)
        ])

    def forward(self, pert: torch.Tensor, ntc: torch.Tensor) -> torch.Tensor:
        """pert: (B, n_p, d), ntc: (B, n_n, d) → (B, n_p, d)."""
        for blk in self.blocks:
            pert = blk(pert, ntc)
        return pert


class PerturbationCellClassifier(nn.Module):
    """Per-cell set-transformer perturbation classifier.

    Pipeline (shapes use B,n_p,n_n,G,K,d,P above):

      1. Module attention (shared) maps each cell's (G,) expression → (K, d)
         module fingerprint. Applied to both pert cells and NTC cells.
      2. Pool modules → (d,) per cell.
      3. Add a learned per-stream type embedding (pert vs ntc) so downstream
         attention can distinguish them.
      4. Self-attend cells within each stream with ``CellSetTransformer``.
      5. Cross-attend: pert-cell queries attend into NTC-cell keys/values.
      6. CLS pool: a learned query cross-attends over the refined pert set
         to produce a single (B, d) pooled representation.
      7. Linear head → (B, P) logits.

    Same ``forward(ntc, pert, gene_ids)`` signature as
    ``rl_model.PerturbationClassifier`` so the existing GRPOTrainer works
    unchanged — but ``ntc`` / ``pert`` are now raw ``(B, n_cells, G)`` cell
    tensors, not quantile-grid tokens.

    Args:
        n_all_genes: Shared gene-embedding vocabulary size.
        n_perts: Number of perturbation classes (P).
        d: Shared model dim.
        n_modules: K — number of learned gene modules.
        n_cell_layers: Depth of within-stream cell self-attn.
        n_cross_layers: Depth of pert→NTC cross-attn.
        n_heads: Attention heads.
        dropout: Used in the two transformer stacks and inside cross-attn.
    """

    def __init__(
        self,
        n_all_genes: int,
        n_perts: int,
        d: int = 64,
        n_modules: int = 32,
        n_cell_layers: int = 2,
        n_cross_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_all_genes = n_all_genes
        self.n_perts = n_perts
        self.d = d
        self.n_modules = n_modules

        # Shared gene embedding used inside the module attention. Exposed
        # as an attribute so external code can inspect or freeze it.
        self.gene_embed = nn.Embedding(n_all_genes, d)
        nn.init.normal_(self.gene_embed.weight, mean=0.0, std=0.1)

        self.module_attn = GeneModuleAttention(
            n_all_genes=n_all_genes,
            d=d,
            n_modules=n_modules,
            n_heads=n_heads,
            gene_embed=self.gene_embed,
        )

        # Per-stream type embeddings — added to cell representations to
        # distinguish pert from NTC during cross-attention.
        self.pert_type = nn.Parameter(torch.randn(d) * 0.1)
        self.ntc_type = nn.Parameter(torch.randn(d) * 0.1)

        self.pert_self_attn = CellSetTransformer(
            d=d, n_heads=n_heads, n_layers=n_cell_layers, dropout=dropout
        )
        self.ntc_self_attn = CellSetTransformer(
            d=d, n_heads=n_heads, n_layers=n_cell_layers, dropout=dropout
        )
        self.cross_attn = PertNTCCrossAttention(
            d=d, n_heads=n_heads, n_layers=n_cross_layers, dropout=dropout
        )

        # Learned CLS query for final pooling over pert cells.
        self.cls_query = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        self.cls_pool = _CrossAttnBlock(d, n_heads, 4 * d, dropout)

        self.norm_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_perts)

    def _encode_stream(
        self,
        cells: torch.Tensor,          # (B, n_cells, G)
        gene_ids: torch.Tensor,       # (G,)
        type_emb: torch.Tensor,       # (d,)
        self_attn: CellSetTransformer,
    ) -> torch.Tensor:                # (B, n_cells, d)
        modules = self.module_attn(cells, gene_ids)       # (B, n_cells, K, d)
        cell_feat = modules.mean(dim=2)                   # (B, n_cells, d)
        cell_feat = cell_feat + type_emb                  # broadcast (d,)
        return self_attn(cell_feat)

    def forward(
        self,
        ntc: torch.Tensor,            # (B, n_n, G)
        pert: torch.Tensor,           # (B, n_p, G)
        gene_ids: torch.Tensor,       # (G,)
    ) -> torch.Tensor:                # (B, P)
        assert ntc.dim() == 3 and pert.dim() == 3, "cells must be (B, n, G)"
        assert ntc.shape[0] == pert.shape[0], "batch dims must match"
        assert ntc.shape[-1] == pert.shape[-1] == gene_ids.shape[0]

        H_p = self._encode_stream(pert, gene_ids, self.pert_type, self.pert_self_attn)
        H_n = self._encode_stream(ntc, gene_ids, self.ntc_type, self.ntc_self_attn)

        H_p = self.cross_attn(H_p, H_n)                   # (B, n_p, d)

        B = H_p.shape[0]
        q = self.cls_query.expand(B, 1, -1)               # (B, 1, d)
        pooled = self.cls_pool(q, H_p).squeeze(1)         # (B, d)
        pooled = self.norm_out(pooled)

        return self.head(pooled)
