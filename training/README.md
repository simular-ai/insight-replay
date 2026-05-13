# Training Recipe — InsightReplay on Qwen3-4B-Base

This folder bundles a **slimmed copy of [verl](https://github.com/verl-project/verl)** with our InsightReplay agent loop already wired in, plus the multi-node cluster launchers used in the paper.

```
training/
├── README.md              # this file
├── baseline_cluster.py    # 128-GPU (16-node) baseline launcher used in the paper
├── insight_cluster.py     # 128-GPU (16-node) InsightReplay launcher used in the paper
└── verl/                  # bundled verl (Apache-2.0, pip-installable)
    ├── verl/              #   Python package
    │   └── experimental/agent_loop/
    │       ├── insight_replay_agent_loop.py   # ← our addition
    │       └── __init__.py                    # ← registers InsightReplayAgentLoop
    ├── setup.py, pyproject.toml, requirements.txt
    ├── LICENSE, Notice.txt
    └── README.md          #   verl's own README
```

Snapshot pinned to upstream verl commit [`2239fd0f`](https://github.com/verl-project/verl/commit/2239fd0f) (Apr 2026).

**What we kept inside `training/verl/`**: only the bare minimum to `pip install -e .` and run training — the Python package itself, the packaging files, the Apache-2.0 `LICENSE` and `Notice.txt` (required for redistribution), and verl's own `README.md`. Everything else from the upstream tree (`tests/`, `docs/`, `docker/`, `scripts/`, `recipe/`, `examples/`, `.github/`, `CONTRIBUTING.md`, runtime artifacts, etc.) has been stripped.

## Two conda envs

verl has heavy and opinionated dependencies (vLLM + torch + CUDA + Ray + FSDP). It will not coexist cleanly with the lighter eval-side env in the project root. **Use one conda env per side**:

| Env | Used for | How |
|---|---|---|
| `insightreplay-eval` (or whatever you call it) | the sampling / grading pipeline in this repo's root | `pip install -r ../requirements.txt` |
| `insightreplay-train` | RL training | `pip install -e training/verl` |

### Set up the training env

```bash
conda create -n insightreplay-train python=3.10 -y
conda activate insightreplay-train

# verl itself + its deps
cd training/verl
pip install -e .
```

vLLM (which verl uses as the rollout engine) pulls in `torch` / `cuda` runtime via its own wheel. If you need a torch build matched to a specific CUDA version, install torch separately first.

## Prepare datasets

The launchers expect parquet files at a shared NAS path (configurable inside each script):

| File | Source |
|---|---|
| `dapo-math-15k-rl.parquet` (InsightReplay train) | [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k); drop the SFT-style examples to match the paper |
| `aime-2025-verl.parquet` (eval) | [AIME-2025](https://huggingface.co/datasets/opencompass/AIME2025) reformatted to verl's prompt schema |

For preprocessing scripts (parquet conversion, prompt schema reformat), pull upstream verl's `examples/data_preprocess/` from the pinned commit linked above — we strip that folder from the bundle to keep the slim.

## Run

`baseline_cluster.py` and `insight_cluster.py` are the **exact 128-GPU (16 nodes × 8 H800) launchers** used to produce the paper's main results. They submit a Ray-on-Kubernetes job through a proprietary RL job-submission SDK (`pypai` + `aistudio_common`) that is **not** part of this release; you cannot run these files as-is on a different cluster.

Treat them as a reference for two things:

1. **The exact distributed hyperparameters** at the 128-GPU scale — Ulysses SP=2, FSDP_SIZE=8, train_batch_size=256, ppo_mini_batch_size=128, save_freq=50, total_epochs=10, etc.
2. **A skeleton you can port to your cluster.** All internal paths, registry URLs, k8s app names, store names, and git endpoints have been replaced with `<your-...>` / `${SHARED_NAS}/...` placeholders. The actual training command embedded inside (the long `python3 -m verl.trainer.main_ppo ...` string) is portable — adapt the `RLJobBuilder(...)` block at the bottom to whatever your scheduler is (Slurm `sbatch`, Ray Job Submission API, SkyPilot, raw `kubectl apply -f`, etc.).

The InsightReplay-only delta over baseline is **one config line**:
```
actor_rollout_ref.rollout.agent.default_agent_loop=insight_replay_agent
```
plus an `exp_name` change. Everything else is byte-for-byte identical.

## How InsightReplay works (at the rollout level)

`training/verl/verl/experimental/agent_loop/insight_replay_agent_loop.py` implements a **two-phase rollout** in place of verl's stock single-turn rollout:

1. **Phase 1.** Generate normally.
2. **Inject decision.** If phase 1 emitted a natural EOS *and* there is room left in the response-length budget, splice in a fixed reflection prompt (`WAIT_TEMPLATE`) that steers phase 2 to verify the answer by a **completely different method** rather than re-running the same derivation.
3. **Phase 2.** Generate again, conditioned on `[original prompt | phase 1 output | injected wait text]`. Phase 2's tokens get policy gradient; the injected wait tokens are masked out (mask = 0) so they don't contribute to the loss.
4. **No injection.** If phase 1 hit the response-length cap without a natural EOS, return the phase 1 rollout as-is — the reward manager treats it as overlong via DAPO's soft-overlong buffer.

The "completely different method" steering is load-bearing. Without it, RL-trained phase 2 collapses into "echo phase 1" within a few hundred steps and the reflection adds nothing — see the ablation in the paper.

## Logging

The cluster launchers log to console + TensorBoard under `project_name=InsightReplay`. The TensorBoard event files get synced to `${LOG_DIR}` (a shared NAS path) every 60 s by a background `cp -ru` loop spawned in `init_cmd`. Switch to a different logger (e.g. Weights & Biases) by editing the `trainer.logger=...` line inside `training_command`.

## License

The bundled `training/verl/` directory is verl's source, distributed under the **Apache License 2.0**. See `training/verl/LICENSE` and `training/verl/Notice.txt`. Our modifications are limited to:

- `training/verl/verl/experimental/agent_loop/insight_replay_agent_loop.py` (new file)
- `training/verl/verl/experimental/agent_loop/__init__.py` (registration of `InsightReplayAgentLoop`)
- `training/baseline_cluster.py` and `training/insight_cluster.py` (multi-node reference launchers we wrote — all infra-specific identifiers replaced with placeholders)

These additions inherit the Apache-2.0 license from upstream. The rest of this repository ships under the license declared at the project root.
