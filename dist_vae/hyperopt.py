"""Hyperparameter optimization for the Distribution VAE using Optuna.

Provides search space definitions, objective functions, and a high-level
``run_hyperopt`` entry point for automated hyperparameter tuning.

Requires the ``hyperopt`` optional dependency group::

    pip install distribution-vae[hyperopt]
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import optuna

try:
    import optuna
except ImportError:
    raise ImportError(
        "Hyperparameter optimization requires optuna. "
        "Install with: pip install distribution-vae[hyperopt]"
    )


def default_search_space() -> dict[str, dict]:
    """Return the default hyperparameter search space.

    Each key is a dot-notation config path. Values are dicts describing
    the parameter type and range for Optuna sampling.

    Supported types:
        - ``categorical``: requires ``choices`` list
        - ``int``: requires ``low`` and ``high``
        - ``float``: requires ``low`` and ``high``
        - ``log_float``: requires ``low`` and ``high`` (log-uniform)

    Returns:
        Search space dictionary.
    """
    return {
        "model.latent_dim": {"type": "categorical", "choices": [16, 32, 64]},
        "model.hidden_dim": {"type": "categorical", "choices": [64, 128, 256]},
        "model.beta": {"type": "log_float", "low": 1e-4, "high": 0.1},
        "model.beta_warmup_epochs": {"type": "int", "low": 0, "high": 20},
        "training.lr": {"type": "log_float", "low": 1e-5, "high": 1e-2},
        "training.weight_decay": {"type": "log_float", "low": 1e-6, "high": 1e-2},
        "training.batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
    }


def _suggest_param(
    trial: optuna.Trial, name: str, spec: dict
) -> int | float | str:
    """Sample a single hyperparameter from an Optuna trial."""
    param_type = spec["type"]
    if param_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    elif param_type == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    elif param_type == "float":
        return trial.suggest_float(name, spec["low"], spec["high"])
    elif param_type == "log_float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    else:
        raise ValueError(f"Unknown parameter type: {param_type!r}")


def _set_nested(config: dict, dotted_key: str, value: object) -> None:
    """Set a value in a nested dict using dot-notation key."""
    parts = dotted_key.split(".")
    d = config
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def build_config_from_trial(
    trial: optuna.Trial,
    base_config: dict,
    search_space: dict[str, dict] | None = None,
) -> dict:
    """Sample hyperparameters from a trial and merge into a config.

    Args:
        trial: Optuna trial object.
        base_config: Base configuration dictionary (not modified).
        search_space: Search space definition. If None, uses
            :func:`default_search_space`.

    Returns:
        New config dict with sampled hyperparameters applied.
    """
    if search_space is None:
        search_space = default_search_space()

    config = copy.deepcopy(base_config)
    for name, spec in search_space.items():
        value = _suggest_param(trial, name, spec)
        _set_nested(config, name, value)

    return config


def create_objective(
    train_dataset: Dataset,
    val_dataset: Dataset,
    base_config: dict,
    search_space: dict[str, dict] | None = None,
    n_epochs: int = 30,
    device: torch.device | None = None,
) -> callable:
    """Create an Optuna objective function for the Distribution VAE.

    The returned callable trains a fresh model for each trial and returns
    the best validation reconstruction loss. It supports Optuna pruning
    via the Trainer's epoch callback.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        base_config: Base configuration dictionary.
        search_space: Search space definition. If None, uses default.
        n_epochs: Number of training epochs per trial.
        device: Device for training. If None, auto-detects.

    Returns:
        Objective function compatible with ``study.optimize()``.
    """
    from dist_vae.model import DistributionVAE
    from dist_vae.train import Trainer

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    def objective(trial: optuna.Trial) -> float:
        config = build_config_from_trial(trial, base_config, search_space)

        # Override training epochs and suppress noisy output
        config.setdefault("training", {})["epochs"] = n_epochs
        config.setdefault("logging", {})["print_every"] = n_epochs + 1
        config.setdefault("logging", {})["snapshot_every"] = 0

        # Use a temp dir for checkpoints to avoid polluting the workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            config["logging"]["checkpoint_dir"] = tmpdir

            model_cfg = config.get("model", {})
            loss_cfg = config.get("loss", {})
            loss_weights = {
                "cramer": loss_cfg.get("cramer", 1.0),
                "wasserstein1": loss_cfg.get("wasserstein1", 0.0),
                "kl_divergence": loss_cfg.get("kl_divergence", 0.0),
            }

            model = DistributionVAE(
                grid_size=model_cfg.get("grid_size", 256),
                latent_dim=model_cfg.get("latent_dim", 32),
                hidden_dim=model_cfg.get("hidden_dim", 128),
                beta=model_cfg.get("beta", 0.01),
                loss_config=loss_weights,
            ).to(device)

            trainer = Trainer(model, train_dataset, val_dataset, config)

            def _pruning_callback(epoch: int, metrics: dict) -> None:
                trial.report(metrics["val_recon"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            history = trainer.train(
                n_epochs=n_epochs, epoch_callback=_pruning_callback
            )

        return min(history["val_recon"])

    return objective


def run_hyperopt(
    train_dataset: Dataset,
    val_dataset: Dataset,
    base_config: dict,
    n_trials: int = 50,
    n_epochs: int = 30,
    search_space: dict[str, dict] | None = None,
    device: torch.device | None = None,
    study_name: str | None = None,
    storage: str | None = None,
) -> tuple[dict, "optuna.Study"]:
    """Run hyperparameter optimization and return the best configuration.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        base_config: Base configuration dictionary.
        n_trials: Number of Optuna trials to run.
        n_epochs: Training epochs per trial.
        search_space: Search space definition. If None, uses default.
        device: Device for training. If None, auto-detects.
        study_name: Optional study name for Optuna.
        storage: Optional Optuna storage URL (e.g., sqlite:///study.db)
            for resumable or distributed search.

    Returns:
        Tuple of (best_config, study). best_config is the base config
        with the best hyperparameters applied. study is the Optuna
        Study object for further analysis.
    """
    objective = create_objective(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        base_config=base_config,
        search_space=search_space,
        n_epochs=n_epochs,
        device=device,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
    )

    study = optuna.create_study(
        study_name=study_name or "distribution-vae-hyperopt",
        storage=storage,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials)

    # Reconstruct best config
    best_config = copy.deepcopy(base_config)
    for name, value in study.best_params.items():
        _set_nested(best_config, name, value)

    return best_config, study


def best_config_to_yaml(config: dict, path: str | Path) -> None:
    """Save a configuration dictionary as a YAML file.

    Args:
        config: Configuration dictionary.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
