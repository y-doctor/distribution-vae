"""CLI script for encoding all distributions in a dataset to latent vectors."""

import argparse
from pathlib import Path

import torch


def main() -> None:
    """Encode all distributions in an AnnData file to latent representations."""
    parser = argparse.ArgumentParser(description="Encode distributions to latent vectors")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--adata", type=str, required=True, help="Path to AnnData .h5ad file")
    parser.add_argument("--output", type=str, required=True, help="Output path for latent AnnData")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    args = parser.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader

    from dist_vae.data import PerturbationDistributionDataset
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

    # Load data
    adata = ad.read_h5ad(args.adata)
    data_cfg = config.get("data", {})
    dataset = PerturbationDistributionDataset(
        adata=adata,
        perturbation_key=data_cfg.get("perturbation_key", "perturbation"),
        grid_size=grid_size,
        min_cells=data_cfg.get("min_cells", 20),
        gene_subset=data_cfg.get("gene_subset"),
    )

    print(f"Encoding {len(dataset)} distributions...")

    # Encode all distributions
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    all_mu = []

    with torch.no_grad():
        for batch in loader:
            grids = batch[0].to(device)
            mu, _ = model.encoder(grids)
            all_mu.append(mu.cpu())

    latents = torch.cat(all_mu).numpy()

    # Build output AnnData
    obs_data = []
    for i in range(len(dataset)):
        meta = dataset.get_metadata(i)
        obs_data.append(meta)

    obs_df = pd.DataFrame(obs_data)
    var_df = pd.DataFrame(index=[f"z_{i}" for i in range(latent_dim)])

    output_adata = ad.AnnData(X=latents, obs=obs_df, var=var_df)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_adata.write_h5ad(output_path)
    print(f"Saved latent representations to {output_path}")
    print(f"  Shape: {output_adata.shape}")
    print(f"  Obs columns: {list(output_adata.obs.columns)}")


if __name__ == "__main__":
    main()
