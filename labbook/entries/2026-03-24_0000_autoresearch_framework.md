# Autoresearch framework for Distribution VAE
**Date**: 2026-03-24 00:00 UTC
**Duration**: ~30 minutes
**Goal**: Implement an autonomous AI experiment framework inspired by karpathy/autoresearch

## What I did
Created the `autoresearch/` directory with a complete autonomous experimentation framework. The system allows an AI agent to autonomously modify the model/training code, run 5-minute training experiments, evaluate results, and keep or discard changes — indefinitely without human intervention.

## Key changes
- `autoresearch/prepare.py`: Fixed evaluation infrastructure — data loading, train/val split, metric computation (val_cramer). This file is never modified by the agent.
- `autoresearch/train.py`: Self-contained model + training loop that the AI agent modifies. Contains all hyperparameters, model architecture (encoder/decoder/VAE), loss functions, optimizer, and training loop in a single file.
- `autoresearch/program.md`: Agent instructions — setup protocol, experiment loop, parsing rules, research ideas, and rules for keeping/discarding experiments.
- `autoresearch/analyze.py`: Results analysis script — summary statistics, top experiments, progress chart.
- `autoresearch/README.md`: Documentation.

## Design decisions
- Followed Karpathy's autoresearch pattern: single file modification, fixed time budget (5 min), git as experiment tracker, results.tsv as log
- Primary metric: val_cramer (MSE on quantile grids, lower is better)
- Secondary constraint: active_dims must stay > 0 (no posterior collapse)
- train.py is self-contained (duplicates model code from dist_vae/) so the agent can modify it freely without affecting the main library

## Results
- Pipeline verified end-to-end: data loading, model creation, training, evaluation all work
- All 47 existing tests still pass
- Ready for autonomous experimentation

## Next steps
- Point an AI agent at `autoresearch/program.md` and let it run
- Monitor results via `python autoresearch/analyze.py results.tsv`
