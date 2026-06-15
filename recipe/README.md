# OmniAgent Recipes

This directory contains public training recipes and notes for reproducing OmniAgent.

## Cold-Start SFT

`sft_agent_final.yaml` is the public OmniAgent SFT reference recipe. It keeps the final training hyperparameters.

Before training:

1. Convert your trajectories to the SFT JSONL format described in the [root README](../README.md); for collected trajectories, see [inference/parallel_eval_usage.md](../inference/parallel_eval_usage.md).
2. Preprocess multimodal fields with the repo-local `qwen-omni-utils/` and `qwen-vl-utils/` packages.
3. Replace every `dataset_path` entry with your local JSONL shards.
4. Replace `model_name_or_path` with your base Qwen-Omni checkpoint or a framework-compatible thinker-only checkpoint.

This file documents the SFT parameters we used. It does not prescribe a specific public trainer. Users may run SFT with any compatible Qwen-Omni training stack; ms-swift is one possible reference codebase:

https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/omni/sft.sh

The ms-swift script is not intended to be an exact copy of the internal OmniAgent launcher.

## Agentic RL

The TAURA and GRPO launch scripts live under `examples/omniagent_train/`:

- `train_TAURA.sh`: OmniAgent RL recipe with entropy-weighted TAURA credit assignment.
- `train_GRPO.sh`: GRPO baseline recipe.

Keep private data roots, machine-specific paths, and internal service URLs out of committed recipe files. Use local environment variables or untracked config overlays for machine-specific settings.
