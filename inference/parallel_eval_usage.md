# Parallel Evaluation with Gemini

Ray-based parallel evaluation using `parallel_evaluate_gemini.py`.

## Quick Start

```bash
export DASHSCOPE_API_KEY="your-key-here"

python inference/parallel_evaluate_gemini.py \
  --dataset_path /path/to/dataset.json \
  --video_prefix /path/to/videos \
  --processor_path checkpoints/OmniAgent-RL-7B \
  --num_processes 32 \
  --model gemini-3-pro-preview \
  --max_steps 32 \
  --max_frames_len 60 \
  --max_audio_len 300 \
  --max_clip_len 60 \
  --mode OmniAgent
```

## Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset_path` | *(required)* | Path to the dataset JSON file |
| `--video_prefix` | `""` | Video file directory prefix |
| `--processor_path` | `$PROCESSOR_PATH` or `/path/to/Qwen2.5-Omni-7B` | Path to the Qwen processor for tokenization |
| `--results_path` | auto-generated | Output path for results JSON |
| `--model` | `gemini-3-pro-preview` | Gemini model name |
| `--think_level` | `high` | Gemini thinking level |
| `--num_processes` | `30` | Number of parallel Ray workers |
| `--max_steps` | `32` | Maximum interaction turns per episode |
| `--max_frames_len` | `32` | Maximum number of frames per browse action |
| `--max_audio_len` | `120.0` | Maximum audio length in seconds |
| `--max_clip_len` | `60.0` | Maximum clip length in seconds |
| `--mode` | `OmniAgent` | Environment mode |
| `--sft_attempts` | `1` | Number of SFT attempts per sample |
| `--max_samples` | unlimited | Maximum number of samples to evaluate (for testing) |

## Usage Examples

### 1. Standard Evaluation

```bash
python inference/parallel_evaluate_gemini.py \
  --dataset_path /path/to/Video-MME.json \
  --video_prefix /path/to/videos \
  --processor_path checkpoints/OmniAgent-RL-7B \
  --num_processes 32
```

### 2. Use a Different Model

```bash
python inference/parallel_evaluate_gemini.py \
  --dataset_path /path/to/dataset.json \
  --video_prefix /path/to/videos \
  --processor_path checkpoints/OmniAgent-RL-7B \
  --model gemini-2.5-flash \
  --num_processes 16
```

### 3. Adjust Resource Limits

```bash
python inference/parallel_evaluate_gemini.py \
  --dataset_path /path/to/dataset.json \
  --video_prefix /path/to/videos \
  --processor_path checkpoints/OmniAgent-RL-7B \
  --max_steps 50 \
  --max_frames_len 60 \
  --max_audio_len 300 \
  --max_clip_len 60
```

### 4. Test Mode (first 100 samples)

```bash
python inference/parallel_evaluate_gemini.py \
  --dataset_path /path/to/dataset.json \
  --video_prefix /path/to/videos \
  --processor_path checkpoints/OmniAgent-RL-7B \
  --max_samples 100 \
  --num_processes 8
```

## Build SFT Trajectories

For SFT data collection, run the parallel trajectory generator with an explicit `--results_path`:

```bash
python inference/parallel_evaluate_gemini.py \
  --dataset_path /path/to/sft_seed_dataset.json \
  --video_prefix /path/to/videos \
  --processor_path /path/to/Qwen2.5-Omni-7B \
  --results_path results/sft_collect.json \
  --model gemini-3-pro-preview \
  --num_processes 32 \
  --sft_attempts 4
```

This writes `results/sft_collect.json` for sample-level results and `results/sft_collect_steps.jsonl` for step-level multi-turn trajectories. Use the step log as the SFT export input:

```bash
python inference/results_final_v1/filter_and_export_sft.py \
  --input results/sft_collect_steps.jsonl \
  --out_dir results/sft_final \
  --base_path /path/to/video_env_tmp \
  --max_steps 50
```

The export script keeps successful trajectories, localizes media paths with `--base_path`, and writes `*_final_sft.jsonl` under `--out_dir`.
