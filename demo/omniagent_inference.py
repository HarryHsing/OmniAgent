#!/usr/bin/env python3
"""
Standalone OmniAgent inference entrypoint.

This script intentionally reuses the same runtime/model/environment code as
`omniagent_demo_pro.py` so demo and CLI inference stay aligned.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from omniagent_demo_pro import (  # noqa: E402
    DemoConfig,
    InferenceEngine,
    KEY_RUNTIME_ENVS,
    ModelManager,
    generate_history_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", default="")
    parser.add_argument("--question_type", default="MCQ", choices=["MCQ", "TR", "FF", "NUM", "SIZE"])
    parser.add_argument("--options", default="", help="MCQ options separated by literal \\n or real newlines")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_model_len", type=int, default=131072)
    parser.add_argument("--max_prompt_length", type=int, default=65536)
    parser.add_argument("--max_response_length", type=int, default=4096)
    parser.add_argument("--max_num_batched_tokens", type=int, default=131072)
    parser.add_argument("--max_steps", type=int, default=32)  # test-time scaling: try 12, 22, 32, 42, 52
    parser.add_argument("--max_frames_len", type=int, default=60)
    parser.add_argument("--max_audio_len", type=float, default=300.0)
    parser.add_argument("--max_clip_len", type=float, default=60.0)
    parser.add_argument("--mode", default="OmniAgent")
    parser.add_argument("--dynamic_step", action="store_true")
    parser.add_argument("--no_dynamic_step", action="store_true")
    parser.add_argument("--use_tito", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output_json", default="", help="Optional path to save final trajectory JSON")
    return parser.parse_args()


def print_runtime_envs():
    print("Runtime env summary:")
    for key in KEY_RUNTIME_ENVS:
        print(f"  {key}={os.environ.get(key, '')}")


def build_config(args) -> DemoConfig:
    use_dynamic_step = True
    if args.no_dynamic_step:
        use_dynamic_step = False
    elif args.dynamic_step:
        use_dynamic_step = True

    return DemoConfig(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_steps=args.max_steps,
        max_frames_len=args.max_frames_len,
        max_audio_len=args.max_audio_len,
        max_clip_len=args.max_clip_len,
        mode=args.mode,
        use_dynamic_step=use_dynamic_step,
        use_tito=bool(args.use_tito),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )


def main():
    args = parse_args()
    config = build_config(args)

    os.environ["USE_DYNAMIC_STEP"] = "True" if config.use_dynamic_step else "False"
    os.environ["USE_TITO"] = "True" if config.use_tito else "False"

    print_runtime_envs()
    print(f"Model: {config.model_path}")
    print(f"Video: {args.video_path}")
    print(f"Question Type: {args.question_type}")
    print(f"Question: {args.question}")

    options = args.options.replace("\\n", "\n")
    options_list = [o.strip() for o in options.splitlines() if o.strip()] if args.question_type == "MCQ" else None

    model = ModelManager()
    success, msg = model.load(config)
    if not success:
        raise RuntimeError(msg)
    print(msg)

    engine = InferenceEngine(model, config)
    final_result = None

    for event in engine.run(
        args.video_path,
        args.question,
        args.answer,
        args.question_type,
        options_list,
    ):
        if event["type"] == "step":
            step = event["step"]
            conf = "N/A" if step.confidence is None else f"{step.confidence:.3f}"
            print(
                f"[Step {step.step}] action={step.action_type} "
                f"confidence={conf} reward={step.reward:.4f} total_reward={event['total_reward']:.4f} "
                f"prompt={step.prompt_build_seconds:.2f}s mm={step.multimodal_prep_seconds:.2f}s "
                f"gen={step.generation_seconds:.2f}s env={step.env_step_seconds:.2f}s "
                f"total={step.step_wall_seconds:.2f}s "
                f"reuse=0/{step.prompt_total_tokens} "
                f"new={step.prompt_total_tokens} pid=0"
            )
        elif event["type"] == "done":
            final_result = event
            break
        elif event["type"] == "error":
            raise RuntimeError(event["error"])

    if final_result is None:
        raise RuntimeError("Inference terminated without a final result.")

    runtime_config_payload = {
        "dynamic_step_enabled": bool(config.use_dynamic_step),
        "tito_enabled": bool(config.use_tito),
        "max_steps": int(config.max_steps),
        "max_frames": int(config.max_frames_len),
        "max_audio_seconds": float(config.max_audio_len),
        "max_clip_seconds": float(config.max_clip_len),
        "mode": config.mode,
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "top_k": int(config.top_k),
        "kvcache_enabled": False,
        "prefix_caching_enabled": False,
        "prompt_token_ids_enabled": False,
        "internal_tito_enabled": False,
    }
    env_snapshot_payload = {key: os.environ.get(key, "") for key in KEY_RUNTIME_ENVS}
    history = json.loads(
        generate_history_json(
            final_result["steps"],
            engine.full_messages,
            args.video_path,
            args.question,
            runtime_config=runtime_config_payload,
            env_snapshot=env_snapshot_payload,
            include_messages=True,
        )
    )
    history["kvcache"] = {
        "enabled": False,
        "prefix_caching_enabled": False,
        "prompt_token_ids_enabled": False,
        "internal_tito_enabled": False,
    }
    for step_data in history.get("steps", []):
        prompt_total_tokens = int(step_data.get("prompt_total_tokens", 0))
        step_data.update({
            "common_prefix_tokens": 0,
            "common_prefix_ratio": 0.0,
            "effective_new_prompt_tokens": prompt_total_tokens,
            "used_prompt_token_ids": False,
            "prompt_token_ids_requested": False,
            "prompt_token_ids_available": False,
            "prompt_segment_count": 0,
            "prompt_token_ids_error": "",
            "prefix_caching_enabled": False,
        })
    history_json = json.dumps(history, indent=2, ensure_ascii=False)

    print("\nFinal Result")
    print(f"  steps={len(final_result['steps'])}")
    print(f"  total_reward={final_result['total_reward']:.4f}")
    print(f"  final_answer={final_result['final']}")
    total_prompt = sum(step.prompt_build_seconds for step in final_result["steps"])
    total_mm = sum(step.multimodal_prep_seconds for step in final_result["steps"])
    total_gen = sum(step.generation_seconds for step in final_result["steps"])
    total_env = sum(step.env_step_seconds for step in final_result["steps"])
    total_wall = sum(step.step_wall_seconds for step in final_result["steps"])
    total_prompt_tokens = sum(step.prompt_total_tokens for step in final_result["steps"])
    print(f"  model_only_seconds={total_gen:.4f}")
    print(f"  model_pipeline_seconds={total_prompt + total_mm + total_gen:.4f}")
    print(f"  env_only_seconds={total_env:.4f}")
    print(f"  step_wall_seconds={total_wall:.4f}")
    print("  total_common_prefix_tokens=0")
    print(f"  total_effective_new_prompt_tokens={total_prompt_tokens}")
    print("  prefix_reuse_ratio=0.0000")
    print(f"  prompt_token_ids_steps=0/{len(final_result['steps'])}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(history_json, encoding="utf-8")
        print(f"Saved JSON to {output_path}")
    else:
        print(history_json)


if __name__ == "__main__":
    main()
