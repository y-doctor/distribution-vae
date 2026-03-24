# Autoresearch: Autonomous AI Experiments for Distribution VAE

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

An AI coding agent (e.g., Claude) autonomously modifies `train.py`, trains for 5 minutes, checks if the result improved, keeps or discards the change, and repeats — indefinitely, without human intervention.

## Quick Start

```bash
# 1. Make sure the project is installed
pip install -e ".[dev]"

# 2. Point your AI agent at program.md and let it run
#    Example with Claude Code:
#    "Read autoresearch/program.md and follow the instructions"
```

## How It Works

| File | Role | Modified by |
|------|------|-------------|
| `prepare.py` | Fixed infrastructure: data loading, evaluation metric | Nobody (read-only) |
| `train.py` | Model + training loop — the file the AI agent modifies | AI agent |
| `program.md` | Instructions for the AI agent | Human |
| `analyze.py` | Results analysis and plotting | Nobody (utility) |
| `results.tsv` | Experiment log (created at runtime) | AI agent |

## The Metric

**val_cramer** — mean squared error between input and reconstructed quantile grids on the validation set. Lower is better. Current baseline: ~0.0092.

## Analysis

After the agent has run for a while:

```bash
python autoresearch/analyze.py results.tsv
```

This prints a summary and saves `progress.png`.

## Design

- **Single file modification**: The agent only edits `train.py`
- **Fixed time budget**: 5 minutes per experiment (~12 experiments/hour)
- **Git as experiment tracker**: Successful experiments advance the branch; failures get reset
- **Self-contained**: No external services, no distributed training
