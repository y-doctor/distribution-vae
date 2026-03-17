# CLAUDE.md

## Project: distribution-vae

A PyTorch library for encoding arbitrary-sized 1D empirical distributions into fixed-dimensional latent representations via a VAE. Primary use case: compressing per-(gene, perturbation) expression distributions from Perturb-seq into compact vectors for downstream ML.

## First steps for any new session

**Before writing any code, do this:**

1. Read this entire file
2. Read `labbook/STATUS.md` for current project state
3. Read the latest 2-3 entries in `labbook/entries/` (sorted by timestamp)
4. Read `labbook/DECISIONS.md` for architectural decisions already made
5. Check `labbook/TASKS.md` for what's in progress, what's done, and what's open
6. If another agent is currently working (check `labbook/LOCKS.md`), pick a non-conflicting task

**During the session — periodic status check-ins (every 30 minutes):**

Every 30 minutes of active work, write a short status pulse to `labbook/entries/`. Rules:
1. Only write if you did meaningful work in that 30-minute window. If you were idle or waiting, skip it.
2. Use filename format: `YYYY-MM-DD_HHMM_pulse_<short_description>.md`
3. Keep it short — use this template:

```markdown
# Pulse: [what you're working on]
**Date**: YYYY-MM-DD HH:MM UTC
**Task**: [task from TASKS.md]
**Status**: [on track / blocked / pivoting]

## Progress this segment
- [1-3 bullet points of what happened]

## Current blockers
- [any blockers, or "None"]

## Next 30 min plan
- [what you'll do next]
```

4. Also update `labbook/STATUS.md` → "What's in progress" section with your current state
5. Commit the pulse with message: `[labbook] pulse: <brief description>`
6. If blocked, note it clearly so the director or other agents can see it in STATUS.md

**Before ending any session, do this:**

1. Write a labbook entry (see format below)
2. Update `labbook/STATUS.md` with current state
3. Update `labbook/TASKS.md` — mark tasks done, add new ones discovered
4. Update `labbook/DECISIONS.md` if any architectural choices were made
5. Release any locks in `labbook/LOCKS.md`
6. Commit with a descriptive message

## Labbook system

The `labbook/` directory is the shared memory of this project. It must be kept clean and current.

### Directory structure

```
labbook/
├── STATUS.md              # Single-file project snapshot (always current)
├── TASKS.md               # Task board: TODO / IN PROGRESS / DONE
├── DECISIONS.md            # Architectural decisions log (append-only)
├── LOCKS.md                # File-level locks for parallel agents
└── entries/                # Timestamped session logs (append-only, never edit old entries)
    ├── 2026-03-17_1430_initial_scaffold.md
    ├── 2026-03-17_1545_loss_functions.md
    └── ...
```

### `labbook/STATUS.md` format

This is the quick-start file. Any agent should be able to read ONLY this file and know where things stand.

```markdown
# Project Status

**Last updated**: 2026-03-17 14:30 UTC
**Updated by**: Agent session [brief description]

## What works
- [list of functional components with test status]

## What's broken / blocked
- [list of known issues]

## What's in progress
- [what's actively being worked on, by whom if known]

## Next priorities
- [ordered list of what to do next]

## Environment
- Python version: 3.11
- PyTorch version: X.X
- Last tested on: [CPU/GPU, OS]
```

### `labbook/TASKS.md` format

```markdown
# Task Board

## TODO (pick from here)
- [ ] Task description — estimated complexity (S/M/L) — dependencies if any

## IN PROGRESS
- [ ] Task description — started 2026-03-17 14:30 — [brief context]

## DONE
- [x] Task description — completed 2026-03-17 15:00 — [link to labbook entry]
```

### `labbook/DECISIONS.md` format

Append-only. Never modify existing entries.

```markdown
# Architectural Decisions

## [YYYY-MM-DD] Decision title
**Context**: Why this came up
**Decision**: What we chose
**Alternatives considered**: What else we could have done
**Rationale**: Why this choice
```

### `labbook/LOCKS.md` format

Simple mutex for parallel agents. Check before starting work on a module.

```markdown
# File Locks

| File/Module | Locked by | Since | Purpose |
|---|---|---|---|
| dist_vae/model.py | agent-session-X | 2026-03-17 14:30 | Implementing decoder |

If a lock is older than 2 hours, assume it's stale and safe to take.
```

### Labbook entry format

Filename: `YYYY-MM-DD_HHMM_short_description.md`

```markdown
# [Short title]
**Date**: YYYY-MM-DD HH:MM UTC
**Duration**: ~X minutes
**Goal**: What I set out to do

## What I did
[Concise summary of actions taken]

## Key changes
- `path/to/file.py`: [what changed and why]

## Results
[Any metrics, test outputs, error messages, observations]

## Problems encountered
[What went wrong, how it was resolved or left open]

## Next steps
[What the next agent should do]

## Open questions
[Anything unresolved that needs human input or further investigation]
```

## Environment

Portable — runs on any machine with PyTorch. No site-specific dependencies.

### Setup

```bash
git clone https://github.com/<user>/distribution-vae.git
cd distribution-vae
pip install -e ".[dev]"
```

`pyproject.toml` handles all dependencies. For GPU, install the appropriate PyTorch CUDA build: https://pytorch.org/get-started/locally/

### Running

```bash
# Tests (always run after changes)
pytest tests/ -v

# Train on synthetic data (no download needed)
python scripts/train.py --config configs/default.yaml --synthetic

# Download sample Perturb-seq data
python scripts/download_sample_data.py

# Train on real data
python scripts/train.py --config configs/example_perturb_seq.yaml --adata data/sample_perturb_seq.h5ad

# Encode dataset → latent matrix
python scripts/encode_dataset.py --checkpoint checkpoints/best.pt --adata path/to/adata.h5ad --output latents.h5ad

# Evaluate + plots
python scripts/evaluate.py --checkpoint checkpoints/best.pt --adata path/to/adata.h5ad --output-dir eval_results/
```

## Code style & conventions

- Type hints on all function signatures
- Docstrings on all public classes and methods
- `torch.Tensor` annotations for anything touching the model
- Keep AnnData/scipy imports isolated to `data.py` and scripts — model code is pure PyTorch
- All operations must work on both CPU and CUDA
- Tests must run on CPU with no data downloads
- Never import wandb at module level — always behind a flag/try-except

## Module ownership (for parallel work)

These modules are designed to be independently developable:

| Module | Dependencies | Notes |
|---|---|---|
| `dist_vae/losses.py` | torch only | Pure functions, no state |
| `dist_vae/model.py` | losses.py | Encoder, decoder, VAE |
| `dist_vae/data.py` | anndata, scipy | Dataset classes |
| `dist_vae/train.py` | model, data, losses | Training loop |
| `dist_vae/eval.py` | model, data, matplotlib | Eval + plotting |
| `scripts/download_sample_data.py` | anndata, scanpy | Standalone |
| `scripts/train.py` | train module | CLI wrapper |
| `scripts/encode_dataset.py` | model, data | CLI wrapper |
| `tests/*` | mirrors src modules | Each test file is independent |

Two agents CAN work simultaneously on e.g. `losses.py` and `data.py` without conflict. Two agents should NOT both work on `model.py` at the same time.

## Git conventions

- Commit after each logical unit of work (not at end of session)
- Commit messages: `[module] brief description` e.g. `[losses] implement cramer and wasserstein-1 distances`
- Branch only if doing something experimental; main is the working branch for now
- Always run `pytest tests/ -v` before committing (if tests exist for that module)
