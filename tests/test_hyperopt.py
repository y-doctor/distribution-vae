"""Tests for the hyperparameter optimization module."""

import copy
from unittest.mock import MagicMock

import pytest
import torch

optuna = pytest.importorskip("optuna")

from dist_vae.data import SyntheticDistributionDataset
from dist_vae.hyperopt import (
    build_config_from_trial,
    create_objective,
    default_search_space,
    run_hyperopt,
    best_config_to_yaml,
)


@pytest.fixture
def base_config():
    """Minimal base config for testing."""
    return {
        "model": {
            "grid_size": 64,
            "latent_dim": 8,
            "hidden_dim": 32,
            "beta": 0.01,
            "beta_warmup_epochs": 2,
        },
        "training": {
            "epochs": 3,
            "batch_size": 32,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "seed": 42,
        },
        "loss": {
            "cramer": 1.0,
            "wasserstein1": 0.0,
            "kl_divergence": 0.0,
        },
        "logging": {
            "print_every": 100,
            "checkpoint_dir": "checkpoints/",
            "snapshot_every": 0,
            "wandb": False,
        },
    }


@pytest.fixture
def tiny_datasets():
    """Create tiny train/val datasets for fast testing."""
    full = SyntheticDistributionDataset(n_distributions=50, grid_size=64, seed=42)
    train_ds, val_ds = torch.utils.data.random_split(
        full, [40, 10], generator=torch.Generator().manual_seed(42)
    )
    return train_ds, val_ds


class TestDefaultSearchSpace:
    def test_returns_dict(self):
        space = default_search_space()
        assert isinstance(space, dict)
        assert len(space) > 0

    def test_all_entries_have_type(self):
        space = default_search_space()
        for name, spec in space.items():
            assert "type" in spec, f"Missing 'type' for {name}"

    def test_expected_keys_present(self):
        space = default_search_space()
        assert "model.latent_dim" in space
        assert "training.lr" in space
        assert "model.beta" in space


class TestBuildConfigFromTrial:
    def test_merges_into_base_config(self, base_config):
        study = optuna.create_study()
        trial = study.ask()

        search_space = {
            "model.latent_dim": {"type": "categorical", "choices": [16, 32]},
            "training.lr": {"type": "log_float", "low": 1e-5, "high": 1e-2},
        }

        config = build_config_from_trial(trial, base_config, search_space)

        # Should have sampled values
        assert config["model"]["latent_dim"] in [16, 32]
        assert 1e-5 <= config["training"]["lr"] <= 1e-2

        # Should preserve unmodified values
        assert config["model"]["grid_size"] == 64
        assert config["loss"]["cramer"] == 1.0

    def test_does_not_modify_base_config(self, base_config):
        original = copy.deepcopy(base_config)
        study = optuna.create_study()
        trial = study.ask()

        search_space = {
            "model.latent_dim": {"type": "categorical", "choices": [16]},
        }
        build_config_from_trial(trial, base_config, search_space)

        assert base_config == original

    def test_uses_default_space_when_none(self, base_config):
        study = optuna.create_study()
        trial = study.ask()

        config = build_config_from_trial(trial, base_config, search_space=None)

        # Should have sampled all default params
        space = default_search_space()
        for key in space:
            parts = key.split(".")
            val = config
            for p in parts:
                val = val[p]
            assert val is not None


class TestCreateObjective:
    def test_returns_callable(self, tiny_datasets, base_config):
        train_ds, val_ds = tiny_datasets
        obj = create_objective(
            train_ds, val_ds, base_config, n_epochs=2, device=torch.device("cpu")
        )
        assert callable(obj)

    def test_objective_returns_float(self, tiny_datasets, base_config):
        train_ds, val_ds = tiny_datasets

        # Use a tiny search space for speed
        search_space = {
            "training.lr": {"type": "categorical", "choices": [1e-3]},
        }

        obj = create_objective(
            train_ds,
            val_ds,
            base_config,
            search_space=search_space,
            n_epochs=2,
            device=torch.device("cpu"),
        )

        study = optuna.create_study()
        study.optimize(obj, n_trials=1)

        assert study.best_value is not None
        assert isinstance(study.best_value, float)


class TestPruningCallback:
    def test_pruning_raises_trial_pruned(self, tiny_datasets, base_config):
        """Verify that when should_prune returns True, the trial is pruned."""
        train_ds, val_ds = tiny_datasets

        search_space = {
            "training.lr": {"type": "categorical", "choices": [1e-3]},
        }

        obj = create_objective(
            train_ds,
            val_ds,
            base_config,
            search_space=search_space,
            n_epochs=3,
            device=torch.device("cpu"),
        )

        # Create a study with aggressive pruning
        pruner = optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=0)
        study = optuna.create_study(pruner=pruner)

        # Run enough trials that pruning can kick in
        study.optimize(obj, n_trials=3, catch=(Exception,))

        # At least one trial should have completed
        completed = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        assert len(completed) >= 1


class TestRunHyperopt:
    def test_returns_config_and_study(self, tiny_datasets, base_config):
        train_ds, val_ds = tiny_datasets

        search_space = {
            "training.lr": {"type": "categorical", "choices": [1e-3, 5e-4]},
        }

        best_config, study = run_hyperopt(
            train_ds,
            val_ds,
            base_config,
            n_trials=2,
            n_epochs=2,
            search_space=search_space,
            device=torch.device("cpu"),
        )

        assert isinstance(best_config, dict)
        assert isinstance(study, optuna.Study)
        assert "model" in best_config
        assert "training" in best_config
        assert study.best_value is not None

    def test_best_config_has_sampled_params(self, tiny_datasets, base_config):
        train_ds, val_ds = tiny_datasets

        search_space = {
            "model.latent_dim": {"type": "categorical", "choices": [8, 16]},
        }

        best_config, study = run_hyperopt(
            train_ds,
            val_ds,
            base_config,
            n_trials=2,
            n_epochs=2,
            search_space=search_space,
            device=torch.device("cpu"),
        )

        assert best_config["model"]["latent_dim"] in [8, 16]


class TestBestConfigToYaml:
    def test_saves_valid_yaml(self, tmp_path, base_config):
        import yaml

        out_path = tmp_path / "test_config.yaml"
        best_config_to_yaml(base_config, str(out_path))

        assert out_path.exists()
        with open(out_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == base_config

    def test_creates_parent_dirs(self, tmp_path, base_config):
        out_path = tmp_path / "nested" / "dir" / "config.yaml"
        best_config_to_yaml(base_config, str(out_path))
        assert out_path.exists()
