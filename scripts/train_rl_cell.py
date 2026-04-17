"""CLI for GRPO training of the per-cell set-transformer classifier.

Mirrors ``scripts/train_rl.py`` but uses:
  - ``PerturbationClassificationDataset(..., return_cells=True)`` so each
    item is a raw (n_cells, G) cell matrix instead of quantile-grid tokens.
  - ``PerturbationCellClassifier`` from ``dist_vae.rl_cell_model``.

The existing ``GRPOTrainer`` is reused unmodified; it only sees the model's
``(ntc, pert, gene_ids) → logits`` interface and doesn't know whether the
inputs are tokens or raw cells.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import apply_overrides, get_device, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO training for the per-cell set-transformer classifier"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--adata", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--override", type=str, nargs="*", default=[])
    args = parser.parse_args()

    import anndata
    import torch

    from dist_vae.rl_cell_model import PerturbationCellClassifier
    from dist_vae.rl_data import PerturbationClassificationDataset
    from dist_vae.rl_train import GRPOTrainer

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed
    if args.output_dir is not None:
        log = config.setdefault("logging", {})
        log["checkpoint_dir"] = args.output_dir
        log["eval_dir"] = args.output_dir

    device = get_device()
    print(f"Using device: {device}")

    seed = int(config.get("training", {}).get("seed", 42))
    torch.manual_seed(seed)

    print(f"Loading {args.adata} ...")
    adata = anndata.read_h5ad(args.adata)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    data_cfg = config.get("data", {})
    val_fraction = float(data_cfg.get("val_fraction", 0.0))
    split_seed = int(data_cfg.get("split_seed", 123))
    dataset = PerturbationClassificationDataset(
        adata,
        perturbation_key=data_cfg.get("perturbation_key", "perturbation"),
        control_label=data_cfg.get("control_label", "control"),
        n_cells_per_pert=int(data_cfg.get("n_cells_per_pert", 100)),
        n_cells_ntc=int(data_cfg.get("n_cells_ntc", 100)),
        grid_size=int(data_cfg.get("grid_size", 64)),    # unused but required
        min_cells=int(data_cfg.get("min_cells", 30)),
        samples_per_epoch=int(data_cfg.get("samples_per_epoch", 1000)),
        seed=seed,
        val_fraction=val_fraction,
        split_seed=split_seed,
        mode="train",
        return_cells=True,
    )
    P = len(dataset.perturbation_names)
    print(f"  perturbations: {P}")
    print(f"  train NTC cells: {dataset._ntc_cells.shape[0]} "
          f"(full {dataset._ntc_cells_full.shape[0]})")
    if val_fraction > 0:
        print(f"  val_fraction={val_fraction}, split_seed={split_seed}")

    profiles = dataset.compute_delta_mean_profiles()
    print(f"  profiles shape: {tuple(profiles.shape)}")

    model_cfg = config.get("model", {})
    model = PerturbationCellClassifier(
        n_all_genes=len(dataset.vocab.names),
        n_perts=P,
        d=int(model_cfg.get("d", 64)),
        n_modules=int(model_cfg.get("n_modules", 32)),
        n_cell_layers=int(model_cfg.get("n_cell_layers", 2)),
        n_cross_layers=int(model_cfg.get("n_cross_layers", 2)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: {n_params:,}")

    trainer = GRPOTrainer(
        model=model,
        dataset=dataset,
        profiles=profiles,
        gene_ids=dataset.vocab.expression_gene_ids,
        config=config,
        device=device,
    )

    history = trainer.train()

    print(
        f"Done. best mean_reward = {max(history['mean_reward']):.4f}  "
        f"final top1 = {history['top1_acc'][-1]:.3f}"
    )


if __name__ == "__main__":
    main()
