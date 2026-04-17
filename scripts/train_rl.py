"""CLI script for GRPO training of the perturbation classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import the config/device helpers from scripts/train.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import apply_overrides, get_device, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO training for the perturbation-classifier"
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

    from dist_vae.rl_data import PerturbationClassificationDataset
    from dist_vae.rl_model import PerturbationClassifier
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
    ds_kwargs = dict(
        perturbation_key=data_cfg.get("perturbation_key", "perturbation"),
        control_label=data_cfg.get("control_label", "control"),
        n_cells_per_pert=int(data_cfg.get("n_cells_per_pert", 100)),
        n_cells_ntc=int(data_cfg.get("n_cells_ntc", 100)),
        grid_size=int(data_cfg.get("grid_size", 64)),
        min_cells=int(data_cfg.get("min_cells", 30)),
        samples_per_epoch=int(data_cfg.get("samples_per_epoch", 1000)),
        seed=seed,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )
    dataset = PerturbationClassificationDataset(adata, mode="train", **ds_kwargs)
    print(
        f"  perturbations ({len(dataset.perturbation_names)}): "
        f"{len(dataset.perturbation_names)} total"
    )
    print(f"  train NTC cells: {dataset._ntc_cells.shape[0]} (full {dataset._ntc_cells_full.shape[0]})")
    if val_fraction > 0:
        print(f"  val_fraction={val_fraction}, split_seed={split_seed}")

    profiles = dataset.compute_delta_mean_profiles()
    print(f"  profiles shape: {tuple(profiles.shape)}")

    model_cfg = config.get("model", {})
    model = PerturbationClassifier(
        n_all_genes=len(dataset.vocab.names),
        n_perts=len(dataset.perturbation_names),
        pert_target_gene_ids=dataset.vocab.pert_target_gene_ids,
        grid_size=int(model_cfg.get("grid_size", 64)),
        d_embed=int(model_cfg.get("d_embed", 32)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        d_feat=int(model_cfg.get("d_feat", 32)),
        n_attn_layers=int(model_cfg.get("n_attn_layers", 0)),
        n_heads=int(model_cfg.get("n_heads", 4)),
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
