# Environment Setup

OmniAgent uses a single unified Python environment for training, inference, and evaluation.
No separate per-task environment is required.

## Main Environment

Follow the installation instructions in the [root README](../../README.md#installation).

The validated stack is:

| Component | Version |
| --- | --- |
| Python | `3.11.11` |
| PyTorch | `2.7.0+cu126` |
| CUDA | `12.6` |
| flash-attn | `2.7.4.post1` |
| vLLM | `0.9.2` |
| transformers | `4.52.4` |
| Ray | `2.47.1` |

Quick install:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r requirements.txt
pip install -e qwen-vl-utils/
pip install -e qwen-omni-utils/
pip install -e .
```

## Video Environment

The OmniAgent video environment (`env_package/video_env.py`) requires no additional installation beyond the main stack above.
It depends on `qwen_omni_utils` and `qwen_vl_utils`, which are included in this repository under [`qwen-omni-utils/`](../../qwen-omni-utils) and [`qwen-vl-utils/`](../../qwen-vl-utils).
