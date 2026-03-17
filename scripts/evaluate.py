"""CLI script for running evaluation and generating plots."""

import argparse

import torch


def main() -> None:
    """Run evaluation on a trained model and generate report."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Distribution VAE")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--adata", type=str, default=None, help="Path to AnnData .h5ad file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for evaluation")
    args = parser.parse_args()

    from dist_vae.eval import generate_eval_report
    from dist_vae.model import DistributionVAE

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    grid_size = checkpoint.get("grid_size", 256)
    latent_dim = checkpoint.get("latent_dim", 32)

    model_cfg = config.get("model", {})
    model = DistributionVAE(
        grid_size=grid_size,
        latent_dim=latent_dim,
        hidden_dim=model_cfg.get("hidden_dim", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dataset
    if args.synthetic:
        from dist_vae.data import SyntheticDistributionDataset
        dataset = SyntheticDistributionDataset(
            n_distributions=500, grid_size=grid_size, seed=123,
        )
    elif args.adata:
        import anndata as ad
        from dist_vae.data import PerturbationDistributionDataset
        adata = ad.read_h5ad(args.adata)
        data_cfg = config.get("data", {})
        dataset = PerturbationDistributionDataset(
            adata=adata,
            perturbation_key=data_cfg.get("perturbation_key", "perturbation"),
            grid_size=grid_size,
            min_cells=data_cfg.get("min_cells", 20),
        )
    else:
        raise ValueError("Must specify either --synthetic or --adata")

    print(f"Evaluating on {len(dataset)} distributions...")
    metrics = generate_eval_report(model, dataset, args.output_dir)

    print("\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
