# OmniAgent: Native Active Perception as Reasoning for Omni-Modal Understanding

<p align="center">
  <a href="https://arxiv.org/abs/TODO">Paper</a> |
  <a href="https://huggingface.co/TODO">Model</a> |
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="assets/main_framework.png" width="90%"/>
</p>

> **OmniAgent** is a POMDP-based active perception framework that fundamentally decouples reasoning complexity from raw video duration. Through an iterative **Observation-Thought-Action (OTA)** cycle, a 7B agent selectively distills audio-visual signals into persistent textual memory, achieving state-of-the-art results across **10 benchmarks** — outperforming even the 10x larger Qwen2.5-VL-72B on LVBench **(50.5 vs. 47.3)**.

---

## Highlights

<table>
<tr>
<td width="50%">

**Accuracy vs. Frame Count on LVBench**

OmniAgent-7B outperforms Qwen2.5-VL-72B while using ~73% fewer frames (203 vs. 768), demonstrating superior efficiency in both model size and data consumption.

<img src="assets/frames_lvbench_trendline.png" width="100%"/>

</td>
<td width="50%">

**Test-Time Scaling on VideoMME-Long**

Accuracy improves monotonically with the max turn budget (+6.1%), while actual turns saturate at ~11.7 — the agent adaptively stops once it has enough evidence.

<img src="assets/test_time_scaling_mme_long.png" width="100%"/>

</td>
</tr>
</table>

### Why OmniAgent?

- **Native Active Perception**: Unlike passive models that process video uniformly, OmniAgent actively explores via on-demand `{frames, audio, clip}` actions, decoupling reasoning cost from video duration.
- **Omni-Modal**: Natively handles video, audio, and text jointly — the agent uses auditory cues as temporal anchors to guide visual sampling.
- **TAURA**: Entropy-steered credit assignment that resolves the advantage homogenization problem in long-horizon agentic RL.
- **Test-Time Scaling**: Performance improves as reasoning turns increase (+6.1% on VideoMME-Long), validating adaptive computation.

### Key Results

| | Benchmark | OmniAgent-7B |
|:---|:---|:---:|
| **Video Understanding** | LVBench | **50.5** |
| | VideoMME | **67.8** |
| | Minerva | **41.4** |
| | MLVU | **71.1** |
| | VSI-Bench | **46.2** |
| **Audio-Visual Understanding** | OmniVideoBench | **37.1** |
| | DailyOmni | **64.8** |
| | WorldSense | **47.2** |
| **Temporal Grounding** | LongVALE | **39.1** |
| | VUE-TR (Vision+Audio) | **36.5** |
| | VUE-TR (Vision) | **46.1** |

---

### Two-Stage Training

1. **Cold-Start SFT**: 58K synthetic trajectories bootstrapped via teacher-model exploration (Gemini-3.0-Pro) with self-correction traces and dual-stage quality control
2. **Agentic RL with TAURA**: Reinforcement learning with verifiable rewards (exact match for MCQ, IoU for temporal grounding, MRA for continuous tasks)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [SFT Recipe and Preprocessing](#sft-recipe-and-preprocessing)
- [Quick Start](#quick-start)
- [Inference](#inference)
- [Batch Evaluation](#batch-evaluation)
- [Web Demo](#web-demo)
- [Training](#training)
- [Test-Time Scaling](#test-time-scaling)
- [Reward System](#reward-system)
- [Troubleshooting](#troubleshooting)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)

---

## Project Structure

```
omniagent/
├── agent_system/              # Core agent logic
│   ├── environments/          # VideoEnv, EnvManager, prompts
│   │   ├── env_package/       # Video environment implementations
│   │   └── prompts/           # System prompts for the agent
│   ├── multi_turn_rollout/    # Multi-turn OTA interaction loop
│   └── reward_manager/        # Episode-level reward aggregation
├── verl/                      # RL training framework (maintained fork of verl)
│   ├── trainer/               # PPO/GRPO trainer with TAURA
│   └── utils/reward_score/    # Reward functions (MCQ, TR, FF)
├── demo/                      # Inference, evaluation, and web demo
│   ├── launch_demo.sh         # Web demo launcher
│   ├── launch_eval.sh         # Batch evaluation launcher
│   ├── launch_inference.sh    # Single-sample inference launcher
│   ├── omniagent_demo_pro.py  # Gradio web demo
│   ├── omniagent_eval.py      # Batch evaluation script
│   └── omniagent_inference.py # CLI inference script
├── examples/                  # Training launch scripts
│   └── omniagent_train/
│       ├── train-TAURA.sh     # TAURA (our method)
│       └── train-GRPO.sh      # GRPO baseline
├── recipe/                    # Public training recipes and recipe notes
├── data/                      # Training data (SFT + RL)
├── inference/                 # Data generation & filtering pipelines
├── qwen-vl-utils/             # Local fork with OmniAgent modifications
├── qwen-omni-utils/           # Local fork with OmniAgent modifications
├── assets/                    # Example videos and figures
├── checkpoints/               # Model weights (download separately)
└── .env                       # API keys (create manually, gitignored)
```

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1× NVIDIA A100 (40GB) | 1× A100/H100 (80GB) |
| GPU (training) | 8× A100 (80GB) | 16+ GPUs (multi-node) |
| RAM | 32GB | 64GB+ |
| Disk | 30GB (model + code) | 50GB+ (with data) |

> **Note:** Inference with `gpu_memory_utilization=0.6` and `tensor_parallel_size=1` requires ~40GB VRAM. Reduce `gpu_memory_utilization` or increase tensor parallelism for smaller GPUs.

### Software

| Package | Version |
|---------|---------|
| Python | ≥ 3.11 |
| PyTorch | 2.7.0 (CUDA 12.6) |
| flash-attn | 2.7.4.post1 |
| vLLM | 0.9.2 |
| Ray | 2.47.1 |
| Gradio | 5.35.0 |
| transformers | 4.52.4 |

---

## Installation

### 1. Create Environment

```bash
conda create -n omniagent python=3.11 -y
conda activate omniagent
```

### 2. Verify System Dependencies

```bash
nvidia-smi  # Tested with CUDA 12.4
nvcc --version  # Tested with CUDA 12.4
ffmpeg -version  # Tested with ffmpeg 4.4.2
ffprobe -version  # Included with ffmpeg
```

### 3. Install PyTorch (CUDA 12.6)

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```

### 4. Install flash-attn (build from source, ~10 min)

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

> **Tip:** If compilation fails, ensure your CUDA toolkit version matches PyTorch's CUDA version. You may need to set `CUDA_HOME=/usr/local/cuda-12.6`.

### 5. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### 6. Install Local Packages (editable mode)

```bash
pip install -e qwen-vl-utils/
pip install -e qwen-omni-utils/
pip install -e .
```

> **Note:** `pip install -e .` installs the local `verl` package (OmniAgent-maintained fork). `qwen-vl-utils` and `qwen-omni-utils` are local forks with OmniAgent-specific modifications.

### 7. Verify Installation

```bash
python -c "import verl; import vllm; import flash_attn; print('All imports OK')"
```

---

## Configuration

### `.env` File

Create a `.env` file in the project root directory:

```bash
# Required for Free-Form (FF) LLM-as-judge scoring
DASHSCOPE_API_KEY="your-api-key-here"

# Optional: custom API endpoint (default shown below)
# DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
```

**Behavior without API key:**
- MCQ (Multiple Choice): works normally (exact match, no API needed)
- TR (Temporal Range): works normally (IoU computation, no API needed)
- FF (Free-Form): returns reward = 0.0 with a warning message

**How to obtain a DashScope API key:**
1. Visit [Alibaba Cloud DashScope](https://dashscope.aliyun.com/)
2. Register and create an API key
3. The key is used by `verl/utils/reward_score/omni_agent.py` for LLM-as-judge scoring

### Video Root Directories (for demo and batch evaluation)

Set these environment variables to point to your video datasets:

```bash
export OMNIAGENT_VIDEOMME_ROOT=/path/to/videomme/videos
export OMNIAGENT_LVBENCH_ROOT=/path/to/lvbench/videos
export OMNIAGENT_VIDI_ROOT=/path/to/vidi/videos
export OMNIAGENT_ALLOWED_PATHS="/path/to/data:/path/to/models"
```

| Variable | Description |
|----------|-------------|
| `OMNIAGENT_VIDEOMME_ROOT` | Root directory for VideoMME video files |
| `OMNIAGENT_LVBENCH_ROOT` | Root directory for LVBench video files |
| `OMNIAGENT_VIDI_ROOT` | Root directory for VIDI video files |
| `OMNIAGENT_ALLOWED_PATHS` | Colon-separated paths that vLLM is allowed to access |

---

## Data Preparation

### Evaluation / Inference Data Format

Evaluation and single-sample inference use a simple JSONL format (one JSON object per line):

```json
{
  "video": "videos/Video-MME/lMxFbRc3Luk.mp4",
  "question_type": "MCQ",
  "question": "As depicted in the video, why is the teacher still in the museum after the security alarm?",
  "options": ["A. She wants to steal the crown.", "B. She checks the security.", "C. She comes to find her students.", "D. She has a talk with the girl and the boy."],
  "answer": "A"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video` | str | Yes | Path to video file (absolute or relative to dataset root) |
| `question_type` | str | Yes | Question type: `MCQ`, `TR`, `FF`, or `SIZE` |
| `question` | str | Yes | Question text |
| `options` | list | MCQ only | Option list: `["A. ...", "B. ...", "C. ...", "D. ..."]`. Use `[]` or `null` for TR/FF/SIZE |
| `answer` | str | Yes | Ground truth (see format below) |
| `domain` | str | No | Optional content domain (e.g., Entertainment, Sports) |

### Answer Format by Question Type

| Type | Answer Format | Example |
|------|--------------|---------|
| MCQ | Single letter | `"A"` |
| TR (single span) | Nested list `[[start, end]]` | `"[[42.5, 47.8]]"` |
| TR (multi-span) | Multiple ranges `[[s1,e1],[s2,e2],...]` | `"[[15.2, 28.6], [52.0, 61.4]]"` |
| FF | Free-form text | `"White"` |
| SIZE | Numeric string | `"4"` |

### RL Training Data Format

RL training data extends the evaluation format with additional metadata:

```json
{
  "prompt": [{"content": "", "role": "user"}],
  "question_type": "MCQ_LongVR-Short",
  "question": "Based on the video, what is the most likely primary intention behind ...",
  "answer": "B",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "video": "videos/LongVideo-Reason/3WYfzz8_lQs.mp4",
  "fps": 29.97,
  "duration_seconds": 287.23,
  "has_audio": true,
  "data_source": "agent",
  "ability": "agent",
  "extra_info": {"traj_id": "8a049033-...", "error_reason": "LOGIC_WRONG_ANSWER"}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | list | Placeholder message list (required by verl trainer) |
| `question_type` | str | Type with dataset suffix (e.g., `MCQ_LongVR-Short`, `TR_LongVALE-Short`, `SIZE_VSI`). Code extracts the major type via `.split("_", 1)[0].upper()` |
| `video` | str | Path to video file |
| `fps` | float | Video frame rate |
| `duration_seconds` | float | Video duration in seconds |
| `has_audio` | bool | Whether the video has an audio track |
| `data_source` | str | Data source tag (e.g., `"agent"`) |
| `ability` | str | Ability category (e.g., `"agent"`) |
| `extra_info` | dict | Metadata: source trajectory ID, error reason |

> **Note:** RL training uses MCQ, TR, and SIZE types. FF is not used during training.

### SFT Training Data Format

SFT data stores complete agent trajectories. Each line is one step of a trajectory:

```json
{
  "question_id": "example_001",
  "traj_id": "a1b2c3d4-e5f6-...",
  "step": 1,
  "raw_input": [{"role": "system", "content": [...]}, {"role": "user", "content": [...]}],
  "output": "{\"observation\": \"...\", \"think\": \"...\", \"confidence\": 0.3, \"action\": {\"type\": \"get_frames\", ...}}",
  "step_reward": 0.0,
  "done": "False",
  "extra_info": {"video": "...", "question_type": "MCQ", "answer": "B", "won": false, "is_action_valid": "True", "reward": 0.0, ...},
  "episode_reward": 1.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | str | Sample identifier |
| `traj_id` | str | UUID grouping all steps of one trajectory |
| `step` | int | Step number within the trajectory (1-indexed) |
| `raw_input` | list | Full conversation history up to this step |
| `output` | str | Agent's JSON response: `{observation, think, confidence, action}` |
| `step_reward` | float | Reward for this step (typically 0 until final step) |
| `done` | str | `"True"` or `"False"` — whether this step ends the episode |
| `extra_info` | dict | Metadata: video path, question, answer, `won`, `is_action_valid`, `reward`, `token_stats`, `env_config` |
| `episode_reward` | float | Total reward for the full trajectory |

### SFT Recipe and Preprocessing

The public cold-start SFT recipe is provided at `recipe/sft_agent_final.yaml`. It is a sanitized version of the final SFT configuration: private dataset paths and internal checkpoint paths are replaced with placeholders, while the main optimization settings are kept.

For SFT data preprocessing, use the repo-local utility packages installed in editable mode:

```bash
pip install -e qwen-omni-utils/
pip install -e qwen-vl-utils/
```

`qwen-omni-utils/` handles Qwen-Omni multimodal inputs with audio, video, image, and text. `qwen-vl-utils/` handles vision-language inputs and keeps compatibility with vision-only preprocessing paths. These local forks are the recommended preprocessing entry points for OmniAgent SFT trajectories in the `raw_input` / `output` format above.

The recipe records the parameter settings we used for SFT. Users may run SFT with any compatible public training codebase; [ms-swift](https://github.com/modelscope/ms-swift) is one possible reference for Qwen-Omni SFT infrastructure.

OmniAgent uses the Qwen-Omni thinker path for agent reasoning. Audio-output / talker weights are not required for the OmniAgent reasoning loop unless your chosen SFT framework requires the full upstream checkpoint layout.

### Example Files

| File | Description |
|------|-------------|
| `data/example_eval.jsonl` | Evaluation dataset (2 MCQ + 2 TR + 2 SIZE) |
| `data/example_train_rl.jsonl` | RL training format (3 MCQ + 2 TR + 1 SIZE) |
| `data/example_train_sft.jsonl` | SFT training format (3 trajectories: MCQ 3-turn, TR 4-turn, SIZE 5-turn) |
| `assets/example_video_mcq.mp4` | Example video for MCQ inference |
| `assets/example_video_tr.mp4` | Example video for TR inference |
| `assets/example_video_ff.mp4` | Example video for FF inference |

---

## Quick Start

After installation, verify everything works with a single inference:

```bash
python demo/omniagent_inference.py \
  --model_path checkpoints/RL \
  --video_path assets/example_video_mcq.mp4 \
  --question 'Who or what lauds "Immigrant Diaries" as "A SURE FIRE HIT", according to the video?' \
  --question_type MCQ \
  --options "A. Remote Goat.\nB. The New York Times.\nC. Variety.\nD. IndieWire." \
  --answer "A"
```

Expected output: The agent performs multi-turn reasoning (observe → think → act) and arrives at answer `A` with reward `1.0`.

---

## Inference

### Single-Sample CLI

```bash
bash demo/launch_inference.sh /path/to/model /path/to/video.mp4
```

Or with environment variables:

```bash
MODEL_PATH=checkpoints/RL \
VIDEO_PATH=assets/example_video_mcq.mp4 \
QUESTION='Who or what lauds "Immigrant Diaries" as "A SURE FIRE HIT"?' \
QUESTION_TYPE=MCQ \
OPTIONS="A. Remote Goat.\nB. The New York Times.\nC. Variety.\nD. IndieWire." \
ANSWER="A" \
  bash demo/launch_inference.sh
```

### Shell Script Parameters (`launch_inference.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | **(required)** | Path to model checkpoint directory |
| `VIDEO_PATH` | **(required)** | Path to input video file |
| `QUESTION` | — | Question text |
| `ANSWER` | — | Ground truth answer (for reward computation) |
| `QUESTION_TYPE` | `MCQ` | One of: `MCQ`, `TR`, `FF` |
| `OPTIONS` | — | Newline-separated options (e.g., `"A. Foo\nB. Bar\nC. Baz\nD. Qux"`) |
| `OUTPUT_JSON` | `./inference_output/latest_run.json` | Path to save the result JSON |

### Advanced Parameters

Additional parameters can be appended after the shell script. These are passed directly to `omniagent_inference.py`:

```bash
bash demo/launch_inference.sh checkpoints/RL video.mp4 \
  --max_steps 32 --dynamic_step --temperature 1.0
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_steps` | `32` | Maximum reasoning turns |
| `--max_frames_len` | `60` | Maximum frames per `get_frames` action |
| `--max_audio_len` | `300` | Maximum audio duration (seconds) per `get_audio` |
| `--max_clip_len` | `60` | Maximum clip duration (seconds) per `get_clip` |
| `--mode` | `OmniAgent` | Agent mode |
| `--dynamic_step` | off | Enable dynamic step limit based on video duration |
| `--no_dynamic_step` | — | Explicitly disable dynamic step |
| `--use_tito` | off | Enable Test-Time Interaction Optimization |
| `--temperature` | `1.0` | Sampling temperature (**always use 1.0**) |
| `--top_p` | `0.95` | Top-p (nucleus) sampling |
| `--top_k` | `20` | Top-k sampling |
| `--tensor_parallel_size` | `1` | Tensor parallelism for vLLM |
| `--gpu_memory_utilization` | `0.6` | vLLM GPU memory fraction (0.0–1.0) |
| `--max_model_len` | `131072` | Maximum sequence length for vLLM |

### Examples by Question Type

**MCQ (Multiple Choice):**
```bash
MODEL_PATH=checkpoints/RL VIDEO_PATH=assets/example_video_mcq.mp4 \
QUESTION='Who or what lauds "Immigrant Diaries" as "A SURE FIRE HIT", according to the video?' \
QUESTION_TYPE=MCQ \
OPTIONS="A. Remote Goat.\nB. The New York Times.\nC. Variety.\nD. IndieWire." \
ANSWER="A" \
  bash demo/launch_inference.sh
```

**TR (Temporal Range):**
```bash
MODEL_PATH=checkpoints/RL VIDEO_PATH=assets/example_video_tr.mp4 \
QUESTION='What are all the time ranges corresponding to the text query: "A man with tousled dark hair..."?' \
QUESTION_TYPE=TR \
ANSWER="[51.72, 62.92]" \
  bash demo/launch_inference.sh
```

**FF (Free-Form):**
```bash
MODEL_PATH=checkpoints/RL VIDEO_PATH=assets/example_video_ff.mp4 \
QUESTION="During the montage, what color was the horse that the boy in yellow is riding?" \
QUESTION_TYPE=FF \
ANSWER="White" \
  bash demo/launch_inference.sh
```

---

## Batch Evaluation

### Launch

```bash
GPU_IDS=0,1,2,3 \
MODEL_PATH=checkpoints/RL \
DATASET_JSONL=/path/to/dataset.jsonl \
  bash demo/launch_eval.sh
```

### Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | **(required)** | Path to model checkpoint |
| `DATASET_JSONL` | **(required)** | Path to evaluation dataset (.jsonl) |
| `GPU_IDS` | `0` | Comma-separated GPU IDs (e.g., `0,1,2,3`) |
| `OUTPUT_DIR` | `./eval_output/auto` | Output directory. `auto` generates a descriptive name |
| `MAX_SAMPLES` | `-1` | Number of samples to evaluate (`-1` = all) |
| `MAX_STEPS` | `32` | Maximum reasoning turns per sample |
| `MAX_FRAMES_LEN` | `60` | Maximum frames per `get_frames` action |
| `MAX_AUDIO_LEN` | `300` | Maximum audio duration (seconds) per `get_audio` action |
| `MAX_CLIP_LEN` | `60` | Maximum clip duration (seconds) per `get_clip` action |
| `MODE` | `OmniAgent` | Agent mode |
| `TEMPERATURE` | `1.0` | Sampling temperature (**always use 1.0**) |
| `TOP_P` | `0.95` | Top-p (nucleus) sampling |
| `TOP_K` | `20` | Top-k sampling |
| `USE_TITO` | `false` | Enable Test-Time Interaction Optimization |
| `USE_DYNAMIC_STEP` | `true` | Enable dynamic step limit based on video duration |

Additional arguments can be appended after the shell script and are passed to `omniagent_eval.py`:

```bash
bash demo/launch_eval.sh --tensor_parallel_size 2 --gpu_memory_utilization 0.8
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tensor_parallel_size` | `1` | Tensor parallelism for vLLM |
| `--gpu_memory_utilization` | `0.6` | vLLM GPU memory fraction (0.0–1.0) |
| `--max_model_len` | `131072` | Maximum sequence length for vLLM |
| `--sample_indices` | — | Comma-separated indices to evaluate specific samples |

### Output Structure

When `OUTPUT_DIR=./eval_output/auto`, the script auto-generates a descriptive directory name:

```
eval_output/
└── dataset__RL__full__s32_f60_a300_c60_OmniAgent__dyn1_tito0__20260601_120000/
    ├── results.jsonl       # Per-sample detailed results
    ├── summary.json        # Aggregate metrics (accuracy, avg tokens, etc.)
    ├── summary.csv         # Same metrics in CSV format
    ├── run.log             # Full execution log
    └── progress.log        # Progress-filtered log (for monitoring)
```

### Test-Time Scaling Experiment

```bash
for steps in 12 22 32 42 52; do
  MAX_STEPS=$steps GPU_IDS=0,1,2,3 \
  MODEL_PATH=checkpoints/RL \
  DATASET_JSONL=/path/to/dataset.jsonl \
    bash demo/launch_eval.sh
done
```

---

## Web Demo

### Launch

```bash
# Simplest: pass model path as argument
bash demo/launch_demo.sh checkpoints/RL

# With custom settings
MODEL_PATH=checkpoints/RL PORT=8080 TENSOR_PARALLEL=2 \
  GPU_MEMORY_UTIL=0.8 bash demo/launch_demo.sh
```

Access the demo at `http://localhost:7862` (or your custom port) after startup.

### Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | **(required)** | Path to model checkpoint (1st positional arg or env var) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `7862` | Server port |
| `TENSOR_PARALLEL` | `1` | Tensor parallelism size (use 2+ for multi-GPU) |
| `GPU_MEMORY_UTIL` | `0.6` | vLLM GPU memory utilization (0.0–1.0) |
| `DEMO_TYPE` | `pro` | Demo variant |
| `AUTO_KILL` | `true` | Auto-kill processes occupying the port on startup |

### Features

The web demo provides:
- Upload any video file and ask questions
- Select question type (MCQ, TR, FF)
- Configure inference parameters: temperature, top_p, top_k, max_steps, max_frames
- Toggle dynamic step and TITO
- View full reasoning trace (Observation → Think → Action) at each step
- Real-time step-by-step visualization of the agent's decision process

---

## Training

### Prerequisites

- **GPU Cluster**: 8+ GPUs (A100 80GB recommended), multi-node supported
- **Shared Filesystem**: All nodes must access the same data paths (for example, NFS or another shared filesystem)
- **WandB** (optional): `export WANDB_API_KEY=your-key` for experiment tracking

### Cold-Start SFT

Use `recipe/sft_agent_final.yaml` as the public SFT reference recipe. It records the final OmniAgent SFT hyperparameters and uses placeholder paths that should be replaced with your local SFT JSONL shards and base Qwen-Omni checkpoint.

The recipe is intended to document the SFT parameters, not to prescribe a specific trainer. Users can reproduce the SFT stage with a compatible public framework such as [ms-swift](https://github.com/modelscope/ms-swift), or another Qwen-Omni SFT stack, by applying the same dataset format and hyperparameters.

### TAURA (Our Method)

```bash
cd examples/omniagent_train

TRAIN_FILE=/path/to/train_data.jsonl \
VAL_FILE=/path/to/val_data.jsonl \
MODEL_BASE_PATH=/path/to/models \
  bash train-TAURA.sh
```

### GRPO Baseline

```bash
cd examples/omniagent_train

TRAIN_FILE=/path/to/train_data.jsonl \
VAL_FILE=/path/to/val_data.jsonl \
MODEL_BASE_PATH=/path/to/models \
  bash train-GRPO.sh
```

### Dry Run (verify config without launching)

```bash
DRY_RUN=1 bash train-TAURA.sh
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_FILE` | **(required)** | Path to training data JSONL |
| `VAL_FILE` | **(required)** | Path to validation data JSONL |
| `MODEL_BASE_PATH` | **(required)** | Directory containing the base model |
| `train_data_size` | `32` | Prompts per training iteration |
| `N` | `8` | Rollouts per prompt (`total_batch = train_data_size × N`) |
| `max_steps` | `22` | Maximum agent turns during training rollout |
| `MIN_MAX_STEPS` | `5` | Dynamic step lower bound |
| `max_frames_len` | `60` | Maximum frames per `get_frames` action |
| `max_audio_len` | `300` | Maximum audio duration (seconds) per `get_audio` |
| `max_clip_len` | `60` | Maximum clip duration (seconds) per `get_clip` |
| `TP` | `1` | Tensor parallelism size |
| `RETRY_ON_FORMAT_ERROR` | `True` | Retry trajectories that have format errors |
| `USE_DYNAMIC_STEP` | `True` | Enable dynamic step limit |
| `ABS_ON_POLICY` | `True` | Absolute on-policy training mode |
| `MICRO_RATIO` | `4` | Samples per GPU for vLLM rollout |
| Learning rate | `1e-6` | Actor learning rate |
| `total_epochs` | `1` | Number of training epochs |
| `save_freq` | `10` | Checkpoint save frequency (in training steps) |
| `WANDB_API_KEY` | — | WandB API key for logging (optional) |

### Multi-Node Training

The training script automatically handles multi-node setup:

- **Head node** (`NODE_RANK=0`): Starts Ray cluster and launches training
- **Worker nodes** (`NODE_RANK>0`): Join the Ray cluster automatically

The script auto-detects cluster topology via environment variables:
- `MLP_WORKER_NUM` / `WORLD_SIZE`: Total number of nodes
- `MLP_WORKER_0_HOST` / `MASTER_ADDR`: Head node address
- `MLP_WORKER_0_PORT` / `MASTER_PORT`: Head node port
- `MLP_WORKER_RACK_RANK_INDEX` / `RANK`: Current node rank

---

## Test-Time Scaling

OmniAgent supports test-time scaling: increasing the maximum reasoning turns at inference improves accuracy monotonically.

### Dynamic Step

When `USE_DYNAMIC_STEP=true` (default), the effective step limit adapts to video duration:

```
effective_max_steps = min(MIN_MAX_STEPS + int(duration / max_clip_len), MAX_STEPS)
```

**Examples:**
| Video Duration | max_clip_len | Effective Steps (MAX_STEPS=32) |
|----------------|-------------|-------------------------------|
| 60s | 60 | min(5 + 1, 32) = 6 |
| 288s | 60 | min(5 + 4, 32) = 9 |
| 600s | 60 | min(5 + 10, 32) = 15 |
| 1800s | 60 | min(5 + 30, 32) = 32 |

### Disabling Dynamic Step

Set `USE_DYNAMIC_STEP=false` to always use the full `MAX_STEPS` budget regardless of video length. This is useful for:
- Short videos that need more reasoning turns
- Ablation studies comparing fixed vs. dynamic budgets

### Scaling Results

From the paper: +6.1% accuracy on VideoMME-Long when scaling from 12 → 52 maximum steps. Actual turns used saturate at ~11.7, showing the agent adaptively stops early when it has sufficient evidence.

---

## Reward System

OmniAgent uses reward functions corresponding to the question types:

| Type | Reward Function | Range | External API |
|------|----------------|-------|--------------|
| **MCQ** | Exact match (letter) | {0, 1} | Not required |
| **TR** | IoU of temporal ranges | [0, 1] continuous | Not required |
| **FF** | LLM-as-judge via DashScope | {0, 1} | Required (`DASHSCOPE_API_KEY`) |
| **SIZE** | Mean relative accuracy | [0, 1] continuous | Not required |

### MCQ (Multiple Choice Questions)

Direct string comparison of the predicted letter against the ground-truth letter. Case-insensitive.

### TR (Temporal Range)

Computes Intersection-over-Union (IoU) between predicted temporal range `[pred_start, pred_end]` and ground-truth range `[gt_start, gt_end]`:

```
IoU = intersection_length / union_length
```

### FF (Free-Form)

Sends `(question, predicted_answer, ground_truth)` to an LLM judge via the DashScope API. The judge evaluates semantic consistency and returns:
- `1` if the predicted answer is consistent with the ground truth
- `0` otherwise

**Without `DASHSCOPE_API_KEY`**: FF scoring defaults to `0.0` with a printed warning. MCQ and TR scoring are unaffected.

### SIZE (Numeric / Counting)

Evaluates numeric predictions using Mean Relative Accuracy (MRA) across multiple tolerance thresholds (50%–95%):

```
relative_error = |pred - target| / (|target| + eps)
MRA = mean(relative_error < (1 - threshold) for threshold in [0.50, 0.55, ..., 0.95])
```

Returns a continuous score in `[0, 1]`. A prediction within 5% of the target scores `1.0`.

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `DASHSCOPE_API_KEY not set` warning at startup | No `.env` file or key not set | Create `.env` with your API key. Only affects FF scoring — MCQ/TR work without it |
| `CUDA out of memory` during inference | GPU VRAM insufficient | Reduce `GPU_MEMORY_UTIL` (e.g., `0.5`) or use `TENSOR_PARALLEL=2` |
| Port `7862` already in use | Previous process still running | Script auto-kills with `AUTO_KILL=true`. Manual: `lsof -ti:7862 \| xargs kill -9` |
| `flash-attn` build fails | CUDA toolkit version mismatch | Ensure CUDA toolkit matches PyTorch CUDA (12.6). Set `CUDA_HOME=/usr/local/cuda-12.6` |
| Ray cluster nodes not joining | Network issue between nodes | Verify `MASTER_ADDR` is reachable from workers. Check firewall rules |
| Dynamic step too low for long videos | Formula caps at `MIN_MAX_STEPS + duration/clip_len` | Set `USE_DYNAMIC_STEP=false` to use full `MAX_STEPS`, or reduce `MAX_CLIP_LEN` |
| vLLM `max_model_len` error | Sequence too long for allocated memory | Reduce `max_num_batched_tokens` or increase `gpu_memory_utilization` |
| `ModuleNotFoundError: verl` | Local package not installed | Run `pip install -e .` from project root |
| Training OOM on large batches | Too many samples per GPU | Reduce `MICRO_RATIO` (default 4) or increase tensor parallelism |

---

## Acknowledgement

We thank the authors of [verl](https://github.com/volcengine/verl) and [verl-agent](https://github.com/langfengq/verl-agent) for their foundational infrastructure. OmniAgent substantially builds upon and redesigns these codebases to enable native active perception for omni-modal understanding.

## Citation

```bibtex
@inproceedings{xing2026omniagent,
  title={Native Active Perception as Reasoning for Omni-Modal Understanding},
  author={Zhenghao Xing and Ruiyang Xu and Yuxuan Wang and Jinzheng He and Ziyang Ma and Qize Yang and Yunfei Chu and Jin Xu and Junyang Lin and Chi-Wing Fu and Pheng-Ann Heng},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

This repository is released under the [Apache License 2.0](LICENSE).
