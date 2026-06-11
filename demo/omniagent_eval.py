#!/usr/bin/env python3
"""
Asynchronous multi-GPU evaluator for OmniAgent.

One worker process is pinned to one GPU and evaluates samples sequentially on
that GPU. The main process feeds tasks asynchronously through a queue and writes
per-sample results as soon as they arrive.
"""

import argparse
import copy
import csv
import json
import multiprocessing as mp
import os
import queue
import statistics
import traceback
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: str, limit: int = -1) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit > 0 and len(records) >= limit:
                break
    return records


def normalize_sample(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    answer = sample.get("answer", "")
    if isinstance(answer, list) and len(answer) == 1:
        answer = answer[0]
    options = sample.get("options", [])
    if isinstance(options, list):
        options_text = "\n".join(str(x) for x in options)
    else:
        options_text = str(options or "")
    return {
        "sample_index": index,
        "video": sample.get("video", ""),
        "question_type": str(sample.get("question_type", "MCQ")).split("_", 1)[0].upper(),
        "question": sample.get("question", ""),
        "answer": answer,
        "options_text": options_text,
        "ability": sample.get("ability", ""),
        "data_source": sample.get("data_source", ""),
        "extra_info": sample.get("extra_info", {}) if isinstance(sample.get("extra_info"), dict) else {},
        "raw_sample": sample,
    }


def parse_sample_indices_arg(text: str) -> Optional[List[int]]:
    if not text or not str(text).strip():
        return None
    indices: List[int] = []
    for raw_part in str(text).split(","):
        part = raw_part.strip()
        if part:
            indices.append(int(part))
    return indices or None


def common_prefix_token_count(prev_ids: Optional[List[int]], curr_ids: Optional[List[int]]) -> int:
    if not prev_ids or not curr_ids:
        return 0
    limit = min(len(prev_ids), len(curr_ids))
    idx = 0
    while idx < limit and int(prev_ids[idx]) == int(curr_ids[idx]):
        idx += 1
    return idx


def make_json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def clone_token_segments_for_trace(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: clone_token_segments_for_trace(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_token_segments_for_trace(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_token_segments_for_trace(v) for v in value)
    return copy.deepcopy(value)


def build_runtime_payload(config, key_runtime_envs: List[str]) -> Dict[str, Any]:
    return {
        "runtime_config": {
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
        },
        "env_snapshot": {key: os.environ.get(key, "") for key in key_runtime_envs},
    }


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def build_media_usage_summary(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    clip_durations = [
        float(step["requested_duration_seconds"])
        for step in steps
        if step.get("action_type") == "get_clip" and step.get("requested_duration_seconds") is not None
    ]
    audio_durations = [
        float(step["requested_duration_seconds"])
        for step in steps
        if step.get("action_type") == "get_audio" and step.get("requested_duration_seconds") is not None
    ]
    requested_frames = [
        int(step["requested_num_frames"])
        for step in steps
        if step.get("action_type") == "get_frames" and step.get("requested_num_frames") is not None
    ]
    returned_frames = [
        int(step.get("frame_count", 0))
        for step in steps
        if step.get("action_type") == "get_frames"
    ]

    return {
        "clip_calls": len(clip_durations),
        "audio_calls": len(audio_durations),
        "frame_calls": len(returned_frames),
        "total_clip_seconds": sum(clip_durations),
        "total_audio_seconds": sum(audio_durations),
        "total_requested_frames": sum(requested_frames),
        "total_returned_frames": sum(returned_frames),
        "avg_clip_seconds_per_call": (sum(clip_durations) / len(clip_durations)) if clip_durations else 0.0,
        "avg_audio_seconds_per_call": (sum(audio_durations) / len(audio_durations)) if audio_durations else 0.0,
        "avg_requested_frames_per_call": (sum(requested_frames) / len(requested_frames)) if requested_frames else 0.0,
        "avg_returned_frames_per_call": (sum(returned_frames) / len(returned_frames)) if returned_frames else 0.0,
    }


def mean_or_zero(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def build_latency_summary(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt_build = [float(step.get("prompt_build_seconds", 0.0)) for step in steps]
    multimodal_prep = [float(step.get("multimodal_prep_seconds", 0.0)) for step in steps]
    generation = [float(step.get("generation_seconds", 0.0)) for step in steps]
    env_step = [float(step.get("env_step_seconds", 0.0)) for step in steps]
    postprocess = [float(step.get("postprocess_seconds", 0.0)) for step in steps]
    overhead = [float(step.get("overhead_seconds", 0.0)) for step in steps]
    wall = [float(step.get("step_wall_seconds", 0.0)) for step in steps]
    vllm_prefill_ttft = [float(step.get("vllm_prefill_ttft_seconds", 0.0)) for step in steps]
    vllm_decode_after_first = [float(step.get("vllm_decode_after_first_token_seconds", 0.0)) for step in steps]
    vllm_scheduled_to_finish = [float(step.get("vllm_scheduled_to_finish_seconds", 0.0)) for step in steps]
    vllm_generate_to_first = [float(step.get("vllm_generate_to_first_token_seconds", 0.0)) for step in steps]
    vllm_generate_after_first = [float(step.get("vllm_generate_after_first_token_seconds", 0.0)) for step in steps]
    vllm_queue = [float(step.get("vllm_queue_seconds", 0.0)) for step in steps]
    vllm_scheduler = [float(step.get("vllm_scheduler_seconds", 0.0)) for step in steps]
    vllm_model_forward = [float(step.get("vllm_model_forward_seconds", 0.0)) for step in steps]
    vllm_model_execute = [float(step.get("vllm_model_execute_seconds", 0.0)) for step in steps]

    tool_env_totals: Dict[str, float] = {}
    tool_env_counts: Dict[str, int] = {}
    for step in steps:
        action_type = str(step.get("action_type", "unknown"))
        tool_env_totals[action_type] = tool_env_totals.get(action_type, 0.0) + float(step.get("env_step_seconds", 0.0))
        tool_env_counts[action_type] = tool_env_counts.get(action_type, 0) + 1

    tool_env_avg = {
        action_type: (tool_env_totals[action_type] / tool_env_counts[action_type]) if tool_env_counts[action_type] else 0.0
        for action_type in tool_env_totals
    }

    return {
        "total_prompt_build_seconds": sum(prompt_build),
        "total_multimodal_prep_seconds": sum(multimodal_prep),
        "total_generation_seconds": sum(generation),
        "total_env_step_seconds": sum(env_step),
        "total_postprocess_seconds": sum(postprocess),
        "total_overhead_seconds": sum(overhead),
        "total_step_wall_seconds": sum(wall),
        "total_vllm_prefill_ttft_seconds": sum(vllm_prefill_ttft),
        "total_vllm_decode_after_first_token_seconds": sum(vllm_decode_after_first),
        "total_vllm_scheduled_to_finish_seconds": sum(vllm_scheduled_to_finish),
        "total_vllm_generate_to_first_token_seconds": sum(vllm_generate_to_first),
        "total_vllm_generate_after_first_token_seconds": sum(vllm_generate_after_first),
        "total_vllm_queue_seconds": sum(vllm_queue),
        "total_vllm_scheduler_seconds": sum(vllm_scheduler),
        "total_vllm_model_forward_seconds": sum(vllm_model_forward),
        "total_vllm_model_execute_seconds": sum(vllm_model_execute),
        "pure_model_latency_seconds": sum(generation),
        "model_only_seconds": sum(generation),
        "model_pipeline_seconds": sum(prompt_build) + sum(multimodal_prep) + sum(generation),
        "pure_env_latency_seconds": sum(env_step),
        "env_only_seconds": sum(env_step),
        "agent_compute_seconds": sum(generation) + sum(env_step),
        "avg_prompt_build_seconds_per_step": mean_or_zero(prompt_build),
        "avg_multimodal_prep_seconds_per_step": mean_or_zero(multimodal_prep),
        "avg_generation_seconds_per_step": mean_or_zero(generation),
        "avg_env_step_seconds_per_step": mean_or_zero(env_step),
        "avg_model_only_seconds_per_step": mean_or_zero(generation),
        "avg_model_pipeline_seconds_per_step": mean_or_zero(
            [prompt_build[i] + multimodal_prep[i] + generation[i] for i in range(len(steps))]
        ),
        "avg_env_only_seconds_per_step": mean_or_zero(env_step),
        "avg_postprocess_seconds_per_step": mean_or_zero(postprocess),
        "avg_overhead_seconds_per_step": mean_or_zero(overhead),
        "avg_step_wall_seconds": mean_or_zero(wall),
        "avg_vllm_prefill_ttft_seconds_per_step": mean_or_zero(vllm_prefill_ttft),
        "avg_vllm_decode_after_first_token_seconds_per_step": mean_or_zero(vllm_decode_after_first),
        "avg_vllm_scheduled_to_finish_seconds_per_step": mean_or_zero(vllm_scheduled_to_finish),
        "avg_vllm_generate_to_first_token_seconds_per_step": mean_or_zero(vllm_generate_to_first),
        "avg_vllm_generate_after_first_token_seconds_per_step": mean_or_zero(vllm_generate_after_first),
        "avg_vllm_queue_seconds_per_step": mean_or_zero(vllm_queue),
        "avg_vllm_scheduler_seconds_per_step": mean_or_zero(vllm_scheduler),
        "avg_vllm_model_forward_seconds_per_step": mean_or_zero(vllm_model_forward),
        "avg_vllm_model_execute_seconds_per_step": mean_or_zero(vllm_model_execute),
        "p50_generation_seconds_per_step": percentile(generation, 0.50),
        "p90_generation_seconds_per_step": percentile(generation, 0.90),
        "p95_generation_seconds_per_step": percentile(generation, 0.95),
        "p50_env_step_seconds_per_step": percentile(env_step, 0.50),
        "p90_env_step_seconds_per_step": percentile(env_step, 0.90),
        "p95_env_step_seconds_per_step": percentile(env_step, 0.95),
        "p50_vllm_prefill_ttft_seconds_per_step": percentile(vllm_prefill_ttft, 0.50),
        "p90_vllm_prefill_ttft_seconds_per_step": percentile(vllm_prefill_ttft, 0.90),
        "p95_vllm_prefill_ttft_seconds_per_step": percentile(vllm_prefill_ttft, 0.95),
        "p50_vllm_decode_after_first_token_seconds_per_step": percentile(vllm_decode_after_first, 0.50),
        "p90_vllm_decode_after_first_token_seconds_per_step": percentile(vllm_decode_after_first, 0.90),
        "p95_vllm_decode_after_first_token_seconds_per_step": percentile(vllm_decode_after_first, 0.95),
        "p50_vllm_generate_to_first_token_seconds_per_step": percentile(vllm_generate_to_first, 0.50),
        "p90_vllm_generate_to_first_token_seconds_per_step": percentile(vllm_generate_to_first, 0.90),
        "p95_vllm_generate_to_first_token_seconds_per_step": percentile(vllm_generate_to_first, 0.95),
        "p50_vllm_generate_after_first_token_seconds_per_step": percentile(vllm_generate_after_first, 0.50),
        "p90_vllm_generate_after_first_token_seconds_per_step": percentile(vllm_generate_after_first, 0.90),
        "p95_vllm_generate_after_first_token_seconds_per_step": percentile(vllm_generate_after_first, 0.95),
        "tool_env_seconds_total": tool_env_totals,
        "tool_env_seconds_avg": tool_env_avg,
    }


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def make_slug(text: str) -> str:
    cleaned = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_.")
    return slug or "run"


def resolve_output_dir(output_dir: Path, model_path: str, dataset_jsonl: str) -> Path:
    if any(output_dir.iterdir()) if output_dir.exists() else False:
        return output_dir
    if output_dir.name != "auto":
        return output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = make_slug(Path(model_path).name)
    dataset_name = make_slug(Path(dataset_jsonl).stem)
    return output_dir.parent / f"{dataset_name}__{model_name}__{timestamp}"


def compute_group_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    rewards = [float(r.get("total_reward", 0.0)) for r in results if r.get("status") == "ok"]
    steps = [int(r.get("steps_used", 0)) for r in results if r.get("status") == "ok"]
    durations = [float(r.get("episode_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    response_tokens = [int(r.get("total_response_tokens", 0)) for r in results if r.get("status") == "ok"]
    prompt_tokens = [int(r.get("total_prompt_text_tokens", 0)) for r in results if r.get("status") == "ok"]
    prompt_total_tokens = [int(r.get("total_prompt_total_tokens", 0)) for r in results if r.get("status") == "ok"]
    prompt_nontext_tokens = [int(r.get("total_prompt_nontext_tokens", 0)) for r in results if r.get("status") == "ok"]
    full_sequence_tokens = [int(r.get("total_full_sequence_tokens", 0)) for r in results if r.get("status") == "ok"]
    valid_steps = [int(r.get("valid_step_count", 0)) for r in results if r.get("status") == "ok"]
    invalid_steps = [int(r.get("invalid_step_count", 0)) for r in results if r.get("status") == "ok"]
    valid_prompt_total_tokens = [int(r.get("valid_total_prompt_total_tokens", 0)) for r in results if r.get("status") == "ok"]
    valid_prompt_nontext_tokens = [int(r.get("valid_total_prompt_nontext_tokens", 0)) for r in results if r.get("status") == "ok"]
    valid_full_sequence_tokens = [int(r.get("valid_total_full_sequence_tokens", 0)) for r in results if r.get("status") == "ok"]
    invalid_prompt_total_tokens = [int(r.get("invalid_total_prompt_total_tokens", 0)) for r in results if r.get("status") == "ok"]
    invalid_prompt_nontext_tokens = [int(r.get("invalid_total_prompt_nontext_tokens", 0)) for r in results if r.get("status") == "ok"]
    invalid_full_sequence_tokens = [int(r.get("invalid_total_full_sequence_tokens", 0)) for r in results if r.get("status") == "ok"]
    valid_generation_seconds = [float(r.get("valid_total_generation_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    valid_env_step_seconds = [float(r.get("valid_total_env_step_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    valid_step_wall_seconds = [float(r.get("valid_total_step_wall_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    invalid_generation_seconds = [float(r.get("invalid_total_generation_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    invalid_env_step_seconds = [float(r.get("invalid_total_env_step_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    invalid_step_wall_seconds = [float(r.get("invalid_total_step_wall_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    clip_seconds = [float(r.get("media_usage", {}).get("total_clip_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    audio_seconds = [float(r.get("media_usage", {}).get("total_audio_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    requested_frames = [int(r.get("media_usage", {}).get("total_requested_frames", 0)) for r in results if r.get("status") == "ok"]
    returned_frames = [int(r.get("media_usage", {}).get("total_returned_frames", 0)) for r in results if r.get("status") == "ok"]
    clip_calls = [int(r.get("media_usage", {}).get("clip_calls", 0)) for r in results if r.get("status") == "ok"]
    audio_calls = [int(r.get("media_usage", {}).get("audio_calls", 0)) for r in results if r.get("status") == "ok"]
    frame_calls = [int(r.get("media_usage", {}).get("frame_calls", 0)) for r in results if r.get("status") == "ok"]
    total_prompt_build = [float(r.get("latency", {}).get("total_prompt_build_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_multimodal_prep = [float(r.get("latency", {}).get("total_multimodal_prep_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_generation = [float(r.get("latency", {}).get("total_generation_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_env_step = [float(r.get("latency", {}).get("total_env_step_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_model_only = [float(r.get("latency", {}).get("model_only_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_model_pipeline = [float(r.get("latency", {}).get("model_pipeline_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_env_only = [float(r.get("latency", {}).get("env_only_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_postprocess = [float(r.get("latency", {}).get("total_postprocess_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_overhead = [float(r.get("latency", {}).get("total_overhead_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    avg_gen_per_step = [float(r.get("latency", {}).get("avg_generation_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    avg_env_per_step = [float(r.get("latency", {}).get("avg_env_step_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    avg_model_only_per_step = [float(r.get("latency", {}).get("avg_model_only_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    avg_model_pipeline_per_step = [float(r.get("latency", {}).get("avg_model_pipeline_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    avg_env_only_per_step = [float(r.get("latency", {}).get("avg_env_only_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    avg_prompt_per_step = [float(r.get("latency", {}).get("avg_prompt_build_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    avg_mm_per_step = [float(r.get("latency", {}).get("avg_multimodal_prep_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    avg_overhead_per_step = [float(r.get("latency", {}).get("avg_overhead_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    p50_gen = [float(r.get("latency", {}).get("p50_generation_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    p90_gen = [float(r.get("latency", {}).get("p90_generation_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    p95_gen = [float(r.get("latency", {}).get("p95_generation_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    p50_env = [float(r.get("latency", {}).get("p50_env_step_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    p90_env = [float(r.get("latency", {}).get("p90_env_step_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    p95_env = [float(r.get("latency", {}).get("p95_env_step_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    total_common_prefix_tokens = [int(r.get("total_common_prefix_tokens", 0)) for r in results if r.get("status") == "ok"]
    total_effective_new_prompt_tokens = [int(r.get("total_effective_new_prompt_tokens", 0)) for r in results if r.get("status") == "ok"]
    prompt_token_ids_step_count = [int(r.get("prompt_token_ids_step_count", 0)) for r in results if r.get("status") == "ok"]
    prompt_token_ids_requested_step_count = [int(r.get("prompt_token_ids_requested_step_count", 0)) for r in results if r.get("status") == "ok"]
    prompt_token_ids_success_rate = [float(r.get("prompt_token_ids_success_rate", 0.0)) for r in results if r.get("status") == "ok"]
    prefix_reuse_ratio = [float(r.get("prefix_reuse_ratio", 0.0)) for r in results if r.get("status") == "ok"]
    avg_common_prefix_ratio_per_step = [float(r.get("avg_common_prefix_ratio_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    common_prefix_step_count = [int(r.get("common_prefix_step_count", 0)) for r in results if r.get("status") == "ok"]
    avg_common_prefix_tokens_per_step = [float(r.get("avg_common_prefix_tokens_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    total_vllm_cached_tokens = [int(r.get("total_vllm_cached_tokens", 0)) for r in results if r.get("status") == "ok"]
    avg_vllm_cached_ratio_per_step = [float(r.get("avg_vllm_cached_ratio_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    total_vllm_prefill_ttft = [float(r.get("latency", {}).get("total_vllm_prefill_ttft_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_vllm_decode_after_first = [float(r.get("latency", {}).get("total_vllm_decode_after_first_token_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_vllm_generate_to_first = [float(r.get("latency", {}).get("total_vllm_generate_to_first_token_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_vllm_generate_after_first = [float(r.get("latency", {}).get("total_vllm_generate_after_first_token_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_vllm_queue = [float(r.get("latency", {}).get("total_vllm_queue_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_vllm_model_forward = [float(r.get("latency", {}).get("total_vllm_model_forward_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    total_vllm_model_execute = [float(r.get("latency", {}).get("total_vllm_model_execute_seconds", 0.0)) for r in results if r.get("status") == "ok"]
    avg_vllm_prefill_ttft = [float(r.get("latency", {}).get("avg_vllm_prefill_ttft_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    avg_vllm_decode_after_first = [float(r.get("latency", {}).get("avg_vllm_decode_after_first_token_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    avg_vllm_generate_to_first = [float(r.get("latency", {}).get("avg_vllm_generate_to_first_token_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    avg_vllm_generate_after_first = [float(r.get("latency", {}).get("avg_vllm_generate_after_first_token_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    p50_vllm_prefill_ttft = [float(r.get("latency", {}).get("p50_vllm_prefill_ttft_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    p90_vllm_prefill_ttft = [float(r.get("latency", {}).get("p90_vllm_prefill_ttft_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    p95_vllm_prefill_ttft = [float(r.get("latency", {}).get("p95_vllm_prefill_ttft_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    p50_vllm_decode_after_first = [float(r.get("latency", {}).get("p50_vllm_decode_after_first_token_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    p90_vllm_decode_after_first = [float(r.get("latency", {}).get("p90_vllm_decode_after_first_token_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]
    p95_vllm_decode_after_first = [float(r.get("latency", {}).get("p95_vllm_decode_after_first_token_seconds_per_step", 0.0)) for r in results if r.get("status") == "ok"]

    tool_counter = Counter()
    for result in results:
        if result.get("status") == "ok":
            tool_counter.update(result.get("tool_usage_counts", {}))

    ok_count = len([r for r in results if r.get("status") == "ok"])
    return {
        "num_samples": len(results),
        "num_success": ok_count,
        "num_error": len(results) - ok_count,
        "success_rate": (ok_count / len(results)) if results else 0.0,
        "accuracy_reward_gt_0_5": (sum(1 for r in results if r.get("status") == "ok" and r.get("correct")) / ok_count) if ok_count else 0.0,
        "avg_reward": statistics.mean(rewards) if rewards else 0.0,
        "avg_steps": statistics.mean(steps) if steps else 0.0,
        "avg_episode_seconds": statistics.mean(durations) if durations else 0.0,
        "avg_response_tokens": statistics.mean(response_tokens) if response_tokens else 0.0,
        "avg_prompt_text_tokens": statistics.mean(prompt_tokens) if prompt_tokens else 0.0,
        "avg_prompt_total_tokens": statistics.mean(prompt_total_tokens) if prompt_total_tokens else 0.0,
        "avg_prompt_nontext_tokens": statistics.mean(prompt_nontext_tokens) if prompt_nontext_tokens else 0.0,
        "avg_full_sequence_tokens": statistics.mean(full_sequence_tokens) if full_sequence_tokens else 0.0,
        "avg_valid_step_count": statistics.mean(valid_steps) if valid_steps else 0.0,
        "avg_invalid_step_count": statistics.mean(invalid_steps) if invalid_steps else 0.0,
        "avg_valid_prompt_total_tokens": statistics.mean(valid_prompt_total_tokens) if valid_prompt_total_tokens else 0.0,
        "avg_valid_prompt_nontext_tokens": statistics.mean(valid_prompt_nontext_tokens) if valid_prompt_nontext_tokens else 0.0,
        "avg_valid_full_sequence_tokens": statistics.mean(valid_full_sequence_tokens) if valid_full_sequence_tokens else 0.0,
        "avg_invalid_prompt_total_tokens": statistics.mean(invalid_prompt_total_tokens) if invalid_prompt_total_tokens else 0.0,
        "avg_invalid_prompt_nontext_tokens": statistics.mean(invalid_prompt_nontext_tokens) if invalid_prompt_nontext_tokens else 0.0,
        "avg_invalid_full_sequence_tokens": statistics.mean(invalid_full_sequence_tokens) if invalid_full_sequence_tokens else 0.0,
        "avg_valid_generation_seconds": statistics.mean(valid_generation_seconds) if valid_generation_seconds else 0.0,
        "avg_valid_env_step_seconds": statistics.mean(valid_env_step_seconds) if valid_env_step_seconds else 0.0,
        "avg_valid_step_wall_seconds": statistics.mean(valid_step_wall_seconds) if valid_step_wall_seconds else 0.0,
        "avg_invalid_generation_seconds": statistics.mean(invalid_generation_seconds) if invalid_generation_seconds else 0.0,
        "avg_invalid_env_step_seconds": statistics.mean(invalid_env_step_seconds) if invalid_env_step_seconds else 0.0,
        "avg_invalid_step_wall_seconds": statistics.mean(invalid_step_wall_seconds) if invalid_step_wall_seconds else 0.0,
        "avg_total_clip_seconds_per_sample": statistics.mean(clip_seconds) if clip_seconds else 0.0,
        "avg_total_audio_seconds_per_sample": statistics.mean(audio_seconds) if audio_seconds else 0.0,
        "avg_total_requested_frames_per_sample": statistics.mean(requested_frames) if requested_frames else 0.0,
        "avg_total_returned_frames_per_sample": statistics.mean(returned_frames) if returned_frames else 0.0,
        "avg_clip_calls_per_sample": statistics.mean(clip_calls) if clip_calls else 0.0,
        "avg_audio_calls_per_sample": statistics.mean(audio_calls) if audio_calls else 0.0,
        "avg_frame_calls_per_sample": statistics.mean(frame_calls) if frame_calls else 0.0,
        "avg_total_prompt_build_seconds_per_sample": mean_or_zero(total_prompt_build),
        "avg_total_multimodal_prep_seconds_per_sample": mean_or_zero(total_multimodal_prep),
        "avg_total_generation_seconds_per_sample": mean_or_zero(total_generation),
        "avg_total_env_step_seconds_per_sample": mean_or_zero(total_env_step),
        "avg_model_only_seconds_per_sample": mean_or_zero(total_model_only),
        "avg_model_pipeline_seconds_per_sample": mean_or_zero(total_model_pipeline),
        "avg_env_only_seconds_per_sample": mean_or_zero(total_env_only),
        "avg_total_postprocess_seconds_per_sample": mean_or_zero(total_postprocess),
        "avg_total_overhead_seconds_per_sample": mean_or_zero(total_overhead),
        "avg_total_common_prefix_tokens": mean_or_zero(total_common_prefix_tokens),
        "avg_total_effective_new_prompt_tokens": mean_or_zero(total_effective_new_prompt_tokens),
        "avg_prompt_token_ids_step_count": mean_or_zero(prompt_token_ids_step_count),
        "avg_prompt_token_ids_requested_step_count": mean_or_zero(prompt_token_ids_requested_step_count),
        "avg_prompt_token_ids_success_rate": mean_or_zero(prompt_token_ids_success_rate),
        "avg_common_prefix_step_count": mean_or_zero(common_prefix_step_count),
        "avg_common_prefix_tokens_per_step": mean_or_zero(avg_common_prefix_tokens_per_step),
        "avg_common_prefix_ratio_per_step": mean_or_zero(avg_common_prefix_ratio_per_step),
        "avg_prefix_reuse_ratio": mean_or_zero(prefix_reuse_ratio),
        "avg_total_vllm_cached_tokens": mean_or_zero(total_vllm_cached_tokens),
        "avg_vllm_cached_ratio_per_step": mean_or_zero(avg_vllm_cached_ratio_per_step),
        "avg_total_vllm_prefill_ttft_seconds_per_sample": mean_or_zero(total_vllm_prefill_ttft),
        "avg_total_vllm_decode_after_first_token_seconds_per_sample": mean_or_zero(total_vllm_decode_after_first),
        "avg_total_vllm_generate_to_first_token_seconds_per_sample": mean_or_zero(total_vllm_generate_to_first),
        "avg_total_vllm_generate_after_first_token_seconds_per_sample": mean_or_zero(total_vllm_generate_after_first),
        "avg_total_vllm_queue_seconds_per_sample": mean_or_zero(total_vllm_queue),
        "avg_total_vllm_model_forward_seconds_per_sample": mean_or_zero(total_vllm_model_forward),
        "avg_total_vllm_model_execute_seconds_per_sample": mean_or_zero(total_vllm_model_execute),
        "avg_prompt_build_seconds_per_step": mean_or_zero(avg_prompt_per_step),
        "avg_multimodal_prep_seconds_per_step": mean_or_zero(avg_mm_per_step),
        "avg_generation_seconds_per_step": mean_or_zero(avg_gen_per_step),
        "avg_env_step_seconds_per_step": mean_or_zero(avg_env_per_step),
        "avg_model_only_seconds_per_step": mean_or_zero(avg_model_only_per_step),
        "avg_model_pipeline_seconds_per_step": mean_or_zero(avg_model_pipeline_per_step),
        "avg_env_only_seconds_per_step": mean_or_zero(avg_env_only_per_step),
        "avg_overhead_seconds_per_step": mean_or_zero(avg_overhead_per_step),
        "avg_vllm_prefill_ttft_seconds_per_step": mean_or_zero(avg_vllm_prefill_ttft),
        "avg_vllm_decode_after_first_token_seconds_per_step": mean_or_zero(avg_vllm_decode_after_first),
        "avg_vllm_generate_to_first_token_seconds_per_step": mean_or_zero(avg_vllm_generate_to_first),
        "avg_vllm_generate_after_first_token_seconds_per_step": mean_or_zero(avg_vllm_generate_after_first),
        "p50_generation_seconds_per_step": mean_or_zero(p50_gen),
        "p90_generation_seconds_per_step": mean_or_zero(p90_gen),
        "p95_generation_seconds_per_step": mean_or_zero(p95_gen),
        "p50_env_step_seconds_per_step": mean_or_zero(p50_env),
        "p90_env_step_seconds_per_step": mean_or_zero(p90_env),
        "p95_env_step_seconds_per_step": mean_or_zero(p95_env),
        "p50_vllm_prefill_ttft_seconds_per_step": mean_or_zero(p50_vllm_prefill_ttft),
        "p90_vllm_prefill_ttft_seconds_per_step": mean_or_zero(p90_vllm_prefill_ttft),
        "p95_vllm_prefill_ttft_seconds_per_step": mean_or_zero(p95_vllm_prefill_ttft),
        "p50_vllm_decode_after_first_token_seconds_per_step": mean_or_zero(p50_vllm_decode_after_first),
        "p90_vllm_decode_after_first_token_seconds_per_step": mean_or_zero(p90_vllm_decode_after_first),
        "p95_vllm_decode_after_first_token_seconds_per_step": mean_or_zero(p95_vllm_decode_after_first),
        "tool_usage_counts": dict(tool_counter),
    }


def build_breakdowns(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    group_fields = {
        "question_type": lambda r: r.get("question_type", ""),
        "ability": lambda r: r.get("ability", ""),
        "data_source": lambda r: r.get("data_source", ""),
        "split": lambda r: (r.get("extra_info") or {}).get("split", ""),
    }
    breakdowns: Dict[str, Dict[str, Any]] = {}
    for field_name, getter in group_fields.items():
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for result in results:
            key = str(getter(result) or "UNKNOWN")
            buckets.setdefault(key, []).append(result)
        breakdowns[field_name] = {
            key: compute_group_metrics(bucket)
            for key, bucket in sorted(buckets.items(), key=lambda x: x[0])
        }
    return breakdowns


def write_breakdown_csv(path: Path, field_name: str, breakdown: Dict[str, Dict[str, Any]]) -> None:
    fieldnames: List[str] = [field_name]
    seen = {field_name}
    for _key, metrics in breakdown.items():
        for metric_key in metrics.keys():
            if metric_key not in seen:
                seen.add(metric_key)
                fieldnames.append(metric_key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, metrics in breakdown.items():
            row = {field_name: key, **metrics}
            row["tool_usage_counts"] = json.dumps(metrics.get("tool_usage_counts", {}), ensure_ascii=False)
            writer.writerow(row)


def write_sample_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "sample_index",
        "status",
        "correct",
        "question_type",
        "ability",
        "data_source",
        "split",
        "extra_index",
        "gpu_id",
        "video",
        "ground_truth",
        "final_answer",
        "total_reward",
        "steps_used",
        "valid_step_count",
        "invalid_step_count",
        "common_prefix_step_count",
        "prompt_token_ids_step_count",
        "prompt_token_ids_requested_step_count",
        "prompt_token_ids_success_rate",
        "total_common_prefix_tokens",
        "total_effective_new_prompt_tokens",
        "avg_common_prefix_tokens_per_step",
        "avg_common_prefix_ratio_per_step",
        "prefix_reuse_ratio",
        "total_vllm_cached_tokens",
        "avg_vllm_cached_tokens_per_step",
        "avg_vllm_cached_ratio_per_step",
        "episode_seconds",
        "total_response_tokens",
        "total_prompt_text_tokens",
        "total_prompt_total_tokens",
        "total_prompt_nontext_tokens",
        "total_full_sequence_tokens",
        "valid_total_prompt_total_tokens",
        "valid_total_prompt_nontext_tokens",
        "valid_total_full_sequence_tokens",
        "invalid_total_prompt_total_tokens",
        "invalid_total_prompt_nontext_tokens",
        "invalid_total_full_sequence_tokens",
        "total_prompt_build_seconds",
        "total_multimodal_prep_seconds",
        "total_generation_seconds",
        "total_env_step_seconds",
        "model_only_seconds",
        "model_pipeline_seconds",
        "env_only_seconds",
        "total_vllm_prefill_ttft_seconds",
        "total_vllm_decode_after_first_token_seconds",
        "total_vllm_generate_to_first_token_seconds",
        "total_vllm_generate_after_first_token_seconds",
        "total_vllm_queue_seconds",
        "total_vllm_model_forward_seconds",
        "total_vllm_model_execute_seconds",
        "total_postprocess_seconds",
        "total_overhead_seconds",
        "avg_generation_seconds_per_step",
        "avg_env_step_seconds_per_step",
        "avg_prompt_build_seconds_per_step",
        "avg_model_only_seconds_per_step",
        "avg_model_pipeline_seconds_per_step",
        "avg_env_only_seconds_per_step",
        "avg_multimodal_prep_seconds_per_step",
        "avg_overhead_seconds_per_step",
        "avg_vllm_prefill_ttft_seconds_per_step",
        "avg_vllm_decode_after_first_token_seconds_per_step",
        "avg_vllm_generate_to_first_token_seconds_per_step",
        "avg_vllm_generate_after_first_token_seconds_per_step",
        "avg_vllm_queue_seconds_per_step",
        "avg_vllm_model_forward_seconds_per_step",
        "avg_vllm_model_execute_seconds_per_step",
        "p50_generation_seconds_per_step",
        "p90_generation_seconds_per_step",
        "p95_generation_seconds_per_step",
        "p50_env_step_seconds_per_step",
        "p90_env_step_seconds_per_step",
        "p95_env_step_seconds_per_step",
        "p50_vllm_prefill_ttft_seconds_per_step",
        "p90_vllm_prefill_ttft_seconds_per_step",
        "p95_vllm_prefill_ttft_seconds_per_step",
        "p50_vllm_decode_after_first_token_seconds_per_step",
        "p90_vllm_decode_after_first_token_seconds_per_step",
        "p95_vllm_decode_after_first_token_seconds_per_step",
        "clip_calls",
        "audio_calls",
        "frame_calls",
        "total_clip_seconds",
        "total_audio_seconds",
        "total_requested_frames",
        "total_returned_frames",
        "tool_sequence",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            media = result.get("media_usage", {})
            latency = result.get("latency", {})
            extra_info = result.get("extra_info", {}) or {}
            writer.writerow({
                "sample_index": result.get("sample_index"),
                "status": result.get("status"),
                "correct": result.get("correct"),
                "question_type": result.get("question_type"),
                "ability": result.get("ability"),
                "data_source": result.get("data_source"),
                "split": extra_info.get("split", ""),
                "extra_index": extra_info.get("index", ""),
                "gpu_id": result.get("gpu_id"),
                "video": result.get("video"),
                "ground_truth": result.get("ground_truth"),
                "final_answer": result.get("final_answer"),
                "total_reward": result.get("total_reward"),
                "steps_used": result.get("steps_used"),
                "valid_step_count": result.get("valid_step_count"),
                "invalid_step_count": result.get("invalid_step_count"),
                "common_prefix_step_count": result.get("common_prefix_step_count"),
                "prompt_token_ids_step_count": result.get("prompt_token_ids_step_count"),
                "prompt_token_ids_requested_step_count": result.get("prompt_token_ids_requested_step_count"),
                "prompt_token_ids_success_rate": result.get("prompt_token_ids_success_rate"),
                "total_common_prefix_tokens": result.get("total_common_prefix_tokens"),
                "total_effective_new_prompt_tokens": result.get("total_effective_new_prompt_tokens"),
                "avg_common_prefix_tokens_per_step": result.get("avg_common_prefix_tokens_per_step"),
                "avg_common_prefix_ratio_per_step": result.get("avg_common_prefix_ratio_per_step"),
                "prefix_reuse_ratio": result.get("prefix_reuse_ratio"),
                "total_vllm_cached_tokens": result.get("total_vllm_cached_tokens"),
                "avg_vllm_cached_tokens_per_step": result.get("avg_vllm_cached_tokens_per_step"),
                "avg_vllm_cached_ratio_per_step": result.get("avg_vllm_cached_ratio_per_step"),
                "episode_seconds": result.get("episode_seconds"),
                "total_response_tokens": result.get("total_response_tokens"),
                "total_prompt_text_tokens": result.get("total_prompt_text_tokens"),
                "total_prompt_total_tokens": result.get("total_prompt_total_tokens"),
                "total_prompt_nontext_tokens": result.get("total_prompt_nontext_tokens"),
                "total_full_sequence_tokens": result.get("total_full_sequence_tokens"),
                "valid_total_prompt_total_tokens": result.get("valid_total_prompt_total_tokens"),
                "valid_total_prompt_nontext_tokens": result.get("valid_total_prompt_nontext_tokens"),
                "valid_total_full_sequence_tokens": result.get("valid_total_full_sequence_tokens"),
                "invalid_total_prompt_total_tokens": result.get("invalid_total_prompt_total_tokens"),
                "invalid_total_prompt_nontext_tokens": result.get("invalid_total_prompt_nontext_tokens"),
                "invalid_total_full_sequence_tokens": result.get("invalid_total_full_sequence_tokens"),
                "total_prompt_build_seconds": latency.get("total_prompt_build_seconds", 0.0),
                "total_multimodal_prep_seconds": latency.get("total_multimodal_prep_seconds", 0.0),
                "total_generation_seconds": latency.get("total_generation_seconds", 0.0),
                "total_env_step_seconds": latency.get("total_env_step_seconds", 0.0),
                "model_only_seconds": latency.get("model_only_seconds", 0.0),
                "model_pipeline_seconds": latency.get("model_pipeline_seconds", 0.0),
                "env_only_seconds": latency.get("env_only_seconds", 0.0),
                "total_vllm_prefill_ttft_seconds": latency.get("total_vllm_prefill_ttft_seconds", 0.0),
                "total_vllm_decode_after_first_token_seconds": latency.get("total_vllm_decode_after_first_token_seconds", 0.0),
                "total_vllm_generate_to_first_token_seconds": latency.get("total_vllm_generate_to_first_token_seconds", 0.0),
                "total_vllm_generate_after_first_token_seconds": latency.get("total_vllm_generate_after_first_token_seconds", 0.0),
                "total_vllm_queue_seconds": latency.get("total_vllm_queue_seconds", 0.0),
                "total_vllm_model_forward_seconds": latency.get("total_vllm_model_forward_seconds", 0.0),
                "total_vllm_model_execute_seconds": latency.get("total_vllm_model_execute_seconds", 0.0),
                "total_postprocess_seconds": latency.get("total_postprocess_seconds", 0.0),
                "total_overhead_seconds": latency.get("total_overhead_seconds", 0.0),
                "avg_generation_seconds_per_step": latency.get("avg_generation_seconds_per_step", 0.0),
                "avg_env_step_seconds_per_step": latency.get("avg_env_step_seconds_per_step", 0.0),
                "avg_prompt_build_seconds_per_step": latency.get("avg_prompt_build_seconds_per_step", 0.0),
                "avg_model_only_seconds_per_step": latency.get("avg_model_only_seconds_per_step", 0.0),
                "avg_model_pipeline_seconds_per_step": latency.get("avg_model_pipeline_seconds_per_step", 0.0),
                "avg_env_only_seconds_per_step": latency.get("avg_env_only_seconds_per_step", 0.0),
                "avg_multimodal_prep_seconds_per_step": latency.get("avg_multimodal_prep_seconds_per_step", 0.0),
                "avg_overhead_seconds_per_step": latency.get("avg_overhead_seconds_per_step", 0.0),
                "avg_vllm_prefill_ttft_seconds_per_step": latency.get("avg_vllm_prefill_ttft_seconds_per_step", 0.0),
                "avg_vllm_decode_after_first_token_seconds_per_step": latency.get("avg_vllm_decode_after_first_token_seconds_per_step", 0.0),
                "avg_vllm_generate_to_first_token_seconds_per_step": latency.get("avg_vllm_generate_to_first_token_seconds_per_step", 0.0),
                "avg_vllm_generate_after_first_token_seconds_per_step": latency.get("avg_vllm_generate_after_first_token_seconds_per_step", 0.0),
                "avg_vllm_queue_seconds_per_step": latency.get("avg_vllm_queue_seconds_per_step", 0.0),
                "avg_vllm_model_forward_seconds_per_step": latency.get("avg_vllm_model_forward_seconds_per_step", 0.0),
                "avg_vllm_model_execute_seconds_per_step": latency.get("avg_vllm_model_execute_seconds_per_step", 0.0),
                "p50_generation_seconds_per_step": latency.get("p50_generation_seconds_per_step", 0.0),
                "p90_generation_seconds_per_step": latency.get("p90_generation_seconds_per_step", 0.0),
                "p95_generation_seconds_per_step": latency.get("p95_generation_seconds_per_step", 0.0),
                "p50_env_step_seconds_per_step": latency.get("p50_env_step_seconds_per_step", 0.0),
                "p90_env_step_seconds_per_step": latency.get("p90_env_step_seconds_per_step", 0.0),
                "p95_env_step_seconds_per_step": latency.get("p95_env_step_seconds_per_step", 0.0),
                "p50_vllm_prefill_ttft_seconds_per_step": latency.get("p50_vllm_prefill_ttft_seconds_per_step", 0.0),
                "p90_vllm_prefill_ttft_seconds_per_step": latency.get("p90_vllm_prefill_ttft_seconds_per_step", 0.0),
                "p95_vllm_prefill_ttft_seconds_per_step": latency.get("p95_vllm_prefill_ttft_seconds_per_step", 0.0),
                "p50_vllm_decode_after_first_token_seconds_per_step": latency.get("p50_vllm_decode_after_first_token_seconds_per_step", 0.0),
                "p90_vllm_decode_after_first_token_seconds_per_step": latency.get("p90_vllm_decode_after_first_token_seconds_per_step", 0.0),
                "p95_vllm_decode_after_first_token_seconds_per_step": latency.get("p95_vllm_decode_after_first_token_seconds_per_step", 0.0),
                "clip_calls": media.get("clip_calls", 0),
                "audio_calls": media.get("audio_calls", 0),
                "frame_calls": media.get("frame_calls", 0),
                "total_clip_seconds": media.get("total_clip_seconds", 0.0),
                "total_audio_seconds": media.get("total_audio_seconds", 0.0),
                "total_requested_frames": media.get("total_requested_frames", 0),
                "total_returned_frames": media.get("total_returned_frames", 0),
                "tool_sequence": ",".join(result.get("tool_sequence", [])),
                "error": result.get("error", ""),
            })


def worker_main(
    worker_id: int,
    gpu_id: str,
    config_dict: Dict[str, Any],
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    trace_export_dir: str = "",
):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["OMP_NUM_THREADS"] = "1"

        from omniagent_demo_pro import DemoConfig, KEY_RUNTIME_ENVS, ModelManager, generate_history_json

        config = DemoConfig(**config_dict)
        os.environ["USE_DYNAMIC_STEP"] = "True" if config.use_dynamic_step else "False"
        os.environ["USE_TITO"] = "True" if config.use_tito else "False"

        model = ModelManager()
        load_start = time.time()
        success, load_msg = model.load(config)
        load_time = time.time() - load_start
        if not success:
            result_queue.put({
                "type": "worker_error",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "error": load_msg,
            })
            return

        runtime_payload = build_runtime_payload(config, KEY_RUNTIME_ENVS)

        while True:
            task = task_queue.get()
            if task is None:
                task_queue.task_done()
                break

            sample = task["sample"]
            sample_index = sample["sample_index"]
            started_at = utc_now()
            episode_start = time.time()
            sample_trace_dir: Optional[Path] = None
            trace_steps: List[Dict[str, Any]] = []
            if trace_export_dir:
                sample_trace_dir = Path(trace_export_dir) / f"sample_{sample_index:06d}"
                sample_trace_dir.mkdir(parents=True, exist_ok=True)
            result_record: Dict[str, Any] = {
                "type": "sample_result",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "model_path": config.model_path,
                "sample_index": sample_index,
                "video": sample["video"],
                "question_type": sample["question_type"],
                "ability": sample.get("ability", ""),
                "data_source": sample.get("data_source", ""),
                "extra_info": sample.get("extra_info", {}),
                "question": sample["question"],
                "ground_truth": sample["answer"],
                "options": sample["options_text"].splitlines() if sample["options_text"] else [],
                "started_at": started_at,
                "model_load_seconds": load_time,
                "runtime_config": runtime_payload["runtime_config"],
                "env_snapshot": runtime_payload["env_snapshot"],
                "tool_usage_counts": {},
                "tool_sequence": [],
                "steps": [],
                "status": "ok",
                "error": "",
            }

            try:
                ffprobe_start = time.time()
                try:
                    import subprocess
                    r = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "default=nw=1", sample["video"]],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    duration = float(r.stdout.strip().split("=")[1])
                    r = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-show_entries", "stream=r_frame_rate", "-select_streams", "v:0", "-of", "default=nw=1", sample["video"]],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    fps = eval(r.stdout.strip().split("=")[1])
                    r = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-show_entries", "stream=codec_type", "-of", "default=nw=1", sample["video"]],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    has_audio = "audio" in r.stdout
                except Exception:
                    duration, fps, has_audio = 120.0, 30.0, True
                ffprobe_time = time.time() - ffprobe_start

                data = {
                    "video": [sample["video"]],
                    "question_type": [sample["question_type"]],
                    "question": [sample["question"]],
                    "answer": [sample["answer"]],
                    "options": [[o.strip() for o in sample["options_text"].splitlines() if o.strip()]] if sample["question_type"] == "MCQ" else [None],
                    "fps": [fps],
                    "duration_seconds": [duration],
                    "has_audio": [has_audio],
                }

                model.build_env(config)
                reset_start = time.time()
                obs, infos = model.reset(data)
                reset_time = time.time() - reset_start
                full_messages = list(obs["text"][0]) if obs and obs.get("text") else []

                done = False
                step_n = 0
                total_reward = 0.0
                tool_counter: Counter = Counter()
                tool_sequence: List[str] = []
                total_response_tokens = 0
                total_prompt_text_tokens = 0
                total_prompt_total_tokens = 0
                total_prompt_nontext_tokens = 0
                total_full_sequence_tokens = 0
                valid_step_count = 0
                invalid_step_count = 0
                valid_total_response_tokens = 0
                valid_total_prompt_text_tokens = 0
                valid_total_prompt_total_tokens = 0
                valid_total_prompt_nontext_tokens = 0
                valid_total_full_sequence_tokens = 0
                valid_total_prompt_build_seconds = 0.0
                valid_total_multimodal_prep_seconds = 0.0
                valid_total_generation_seconds = 0.0
                valid_total_env_step_seconds = 0.0
                valid_total_postprocess_seconds = 0.0
                valid_total_overhead_seconds = 0.0
                valid_total_step_wall_seconds = 0.0
                invalid_total_response_tokens = 0
                invalid_total_prompt_text_tokens = 0
                invalid_total_prompt_total_tokens = 0
                invalid_total_prompt_nontext_tokens = 0
                invalid_total_full_sequence_tokens = 0
                invalid_total_prompt_build_seconds = 0.0
                invalid_total_multimodal_prep_seconds = 0.0
                invalid_total_generation_seconds = 0.0
                invalid_total_env_step_seconds = 0.0
                invalid_total_postprocess_seconds = 0.0
                invalid_total_overhead_seconds = 0.0
                invalid_total_step_wall_seconds = 0.0
                total_common_prefix_tokens = 0
                total_effective_new_prompt_tokens = 0
                common_prefix_step_count = 0
                prompt_token_ids_step_count = 0
                prompt_token_ids_requested_step_count = 0
                total_vllm_cached_tokens = 0
                vllm_cached_step_count = 0
                total_vllm_prefill_ttft_seconds = 0.0
                total_vllm_decode_after_first_token_seconds = 0.0
                total_vllm_generate_to_first_token_seconds = 0.0
                total_vllm_generate_after_first_token_seconds = 0.0
                total_vllm_queue_seconds = 0.0
                total_vllm_model_forward_seconds = 0.0
                total_vllm_model_execute_seconds = 0.0
                prev_prompt_token_ids: Optional[List[int]] = None
                final_answer = ""

                while not done and step_n < config.max_steps:
                    step_n += 1
                    messages = obs["text"][0]
                    token_segments_before = None
                    if config.use_tito and isinstance(obs, dict):
                        token_segments_before = obs.get("token_segments")
                        if isinstance(token_segments_before, list) and token_segments_before and isinstance(token_segments_before[0], list):
                            token_segments_before = token_segments_before[0]

                    step_started = time.time()
                    prompt_build_started = time.time()
                    prompt_text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if isinstance(prompt_text, list):
                        prompt_text = prompt_text[0] if prompt_text else ""
                    prompt_text_tokens = len(model.processor.tokenizer.encode(prompt_text, add_special_tokens=False))
                    prompt_preview_seconds = time.time() - prompt_build_started
                    total_prompt_text_tokens += prompt_text_tokens

                    response_text, response_ids, gen_metrics = model.generate(
                        messages,
                        return_metrics=True,
                        sample_has_audio=has_audio,
                    )
                    prompt_build_seconds = float(gen_metrics.get("prompt_build_seconds", 0.0))
                    multimodal_prep_seconds = float(gen_metrics.get("multimodal_prep_seconds", 0.0))
                    generation_seconds = float(gen_metrics.get("llm_generate_seconds", 0.0))
                    prompt_text_tokens = int(gen_metrics.get("prompt_text_tokens", prompt_text_tokens))
                    prompt_total_tokens = int(gen_metrics.get("prompt_total_tokens", prompt_text_tokens))
                    prompt_nontext_tokens = int(gen_metrics.get("prompt_nontext_tokens", max(0, prompt_total_tokens - prompt_text_tokens)))
                    response_token_count = int(gen_metrics.get("response_tokens", len(response_ids)))
                    full_sequence_tokens = int(gen_metrics.get("full_sequence_tokens", prompt_total_tokens + response_token_count))
                    prompt_ids = gen_metrics.pop("prompt_token_ids_internal", None)
                    prompt_token_ids_requested = bool(gen_metrics.get("prompt_token_ids_requested", True))
                    prompt_token_ids_available = bool(gen_metrics.get("prompt_token_ids_available", bool(prompt_ids)))
                    common_prefix_tokens = common_prefix_token_count(prev_prompt_token_ids, prompt_ids)
                    common_prefix_ratio = (float(common_prefix_tokens) / float(prompt_total_tokens)) if prompt_total_tokens else 0.0
                    effective_new_prompt_tokens = max(0, int(prompt_total_tokens) - int(common_prefix_tokens))
                    if prompt_ids:
                        prev_prompt_token_ids = prompt_ids
                    if prompt_token_ids_requested:
                        prompt_token_ids_requested_step_count += 1
                    if prompt_token_ids_available:
                        prompt_token_ids_step_count += 1
                    total_common_prefix_tokens += common_prefix_tokens
                    total_effective_new_prompt_tokens += effective_new_prompt_tokens
                    if common_prefix_tokens > 0:
                        common_prefix_step_count += 1

                    vllm_cached_tokens = int(gen_metrics.get("vllm_cached_tokens", 0))
                    vllm_cached_ratio = (float(vllm_cached_tokens) / float(prompt_total_tokens)) if prompt_total_tokens else 0.0
                    total_vllm_cached_tokens += vllm_cached_tokens
                    if vllm_cached_tokens > 0:
                        vllm_cached_step_count += 1
                    vllm_prefill_ttft_seconds = float(gen_metrics.get("vllm_prefill_ttft_seconds", 0.0))
                    vllm_decode_after_first_token_seconds = float(gen_metrics.get("vllm_decode_after_first_token_seconds", 0.0))
                    vllm_generate_to_first_token_seconds = float(gen_metrics.get("vllm_generate_to_first_token_seconds", 0.0))
                    vllm_generate_after_first_token_seconds = float(gen_metrics.get("vllm_generate_after_first_token_seconds", 0.0))
                    vllm_queue_seconds = float(gen_metrics.get("vllm_queue_seconds", 0.0))
                    vllm_scheduler_seconds = float(gen_metrics.get("vllm_scheduler_seconds", 0.0))
                    vllm_model_forward_seconds = float(gen_metrics.get("vllm_model_forward_seconds", 0.0))
                    vllm_model_execute_seconds = float(gen_metrics.get("vllm_model_execute_seconds", 0.0))
                    total_vllm_prefill_ttft_seconds += vllm_prefill_ttft_seconds
                    total_vllm_decode_after_first_token_seconds += vllm_decode_after_first_token_seconds
                    total_vllm_generate_to_first_token_seconds += vllm_generate_to_first_token_seconds
                    total_vllm_generate_after_first_token_seconds += vllm_generate_after_first_token_seconds
                    total_vllm_queue_seconds += vllm_queue_seconds
                    total_vllm_model_forward_seconds += vllm_model_forward_seconds
                    total_vllm_model_execute_seconds += vllm_model_execute_seconds
                    total_response_tokens += response_token_count
                    total_prompt_total_tokens += prompt_total_tokens
                    total_prompt_nontext_tokens += prompt_nontext_tokens
                    total_full_sequence_tokens += full_sequence_tokens

                    postprocess_started = time.time()
                    try:
                        parsed = json.loads(response_text)
                        action_payload = parsed.get("action", {}) or {}
                        action_type = action_payload.get("type", "unknown")
                        think = parsed.get("think", "")
                        observation = parsed.get("observation", "")
                        confidence_raw = parsed.get("confidence")
                        confidence = float(confidence_raw) if confidence_raw is not None else None
                        action_start = safe_float(action_payload.get("start"))
                        action_end = safe_float(action_payload.get("end"))
                        requested_num_frames = safe_int(action_payload.get("num"))
                        requested_duration_seconds = (
                            max(0.0, action_end - action_start)
                            if action_start is not None and action_end is not None
                            else None
                        )
                        final_answer = action_payload.get("content", final_answer) if action_type == "answer" else final_answer
                    except Exception:
                        action_type = "unknown"
                        think = ""
                        observation = ""
                        confidence = None
                        action_start = None
                        action_end = None
                        requested_num_frames = None
                        requested_duration_seconds = None
                    postprocess_seconds = time.time() - postprocess_started

                    env_started = time.time()
                    next_obs, rewards, dones, extra = model.step([response_text], [response_ids] if config.use_tito else None)
                    env_step_seconds = time.time() - env_started

                    if next_obs and "text" in next_obs and next_obs["text"]:
                        full_messages = list(next_obs["text"][0])

                    clip_path = None
                    audio_path = None
                    frame_paths: List[str] = []
                    extra_info = extra[0] if isinstance(extra, list) and extra and isinstance(extra[0], dict) else {}
                    if extra_info:
                        clip_path = extra_info.get("clip_path")
                        audio_path = extra_info.get("audio_path")
                        frame_paths = extra_info.get("frame_paths", []) or []
                    is_action_valid = bool(extra_info.get("is_action_valid", True))
                    error_code = extra_info.get("error_code")
                    error_message = extra_info.get("error_message")

                    reward = float(rewards[0]) if isinstance(rewards, (list, tuple)) else float(rewards)
                    done = bool(dones[0]) if isinstance(dones, (list, tuple)) else bool(dones)
                    total_reward += reward

                    tool_counter[action_type] += 1
                    tool_sequence.append(action_type)
                    step_wall_seconds = time.time() - step_started
                    overhead_seconds = max(0.0, step_wall_seconds - prompt_build_seconds - generation_seconds - env_step_seconds - postprocess_seconds)

                    if is_action_valid:
                        valid_step_count += 1
                        valid_total_response_tokens += response_token_count
                        valid_total_prompt_text_tokens += prompt_text_tokens
                        valid_total_prompt_total_tokens += prompt_total_tokens
                        valid_total_prompt_nontext_tokens += prompt_nontext_tokens
                        valid_total_full_sequence_tokens += full_sequence_tokens
                        valid_total_prompt_build_seconds += prompt_build_seconds
                        valid_total_multimodal_prep_seconds += multimodal_prep_seconds
                        valid_total_generation_seconds += generation_seconds
                        valid_total_env_step_seconds += env_step_seconds
                        valid_total_postprocess_seconds += postprocess_seconds
                        valid_total_overhead_seconds += overhead_seconds
                        valid_total_step_wall_seconds += step_wall_seconds
                    else:
                        invalid_step_count += 1
                        invalid_total_response_tokens += response_token_count
                        invalid_total_prompt_text_tokens += prompt_text_tokens
                        invalid_total_prompt_total_tokens += prompt_total_tokens
                        invalid_total_prompt_nontext_tokens += prompt_nontext_tokens
                        invalid_total_full_sequence_tokens += full_sequence_tokens
                        invalid_total_prompt_build_seconds += prompt_build_seconds
                        invalid_total_multimodal_prep_seconds += multimodal_prep_seconds
                        invalid_total_generation_seconds += generation_seconds
                        invalid_total_env_step_seconds += env_step_seconds
                        invalid_total_postprocess_seconds += postprocess_seconds
                        invalid_total_overhead_seconds += overhead_seconds
                        invalid_total_step_wall_seconds += step_wall_seconds

                    result_record["steps"].append({
                        "step": step_n,
                        "action_type": action_type,
                        "is_action_valid": is_action_valid,
                        "error_code": error_code,
                        "error_message": error_message,
                        "action_start": action_start,
                        "action_end": action_end,
                        "requested_duration_seconds": requested_duration_seconds,
                        "requested_num_frames": requested_num_frames,
                        "confidence": confidence,
                        "reward": reward,
                        "cumulative_reward": total_reward,
                        "prompt_build_preview_seconds": prompt_preview_seconds,
                        "prompt_build_seconds": prompt_build_seconds,
                        "multimodal_prep_seconds": multimodal_prep_seconds,
                        "generation_seconds": generation_seconds,
                        "env_step_seconds": env_step_seconds,
                        "postprocess_seconds": postprocess_seconds,
                        "overhead_seconds": overhead_seconds,
                        "step_wall_seconds": step_wall_seconds,
                        "prompt_text_tokens": prompt_text_tokens,
                        "prompt_total_tokens": prompt_total_tokens,
                        "prompt_nontext_tokens": prompt_nontext_tokens,
                        "response_tokens": response_token_count,
                        "full_sequence_tokens": full_sequence_tokens,
                        "common_prefix_tokens": common_prefix_tokens,
                        "common_prefix_ratio": common_prefix_ratio,
                        "effective_new_prompt_tokens": effective_new_prompt_tokens,
                        "prompt_token_ids_requested": prompt_token_ids_requested,
                        "prompt_token_ids_available": prompt_token_ids_available,
                        "vllm_cached_tokens": vllm_cached_tokens,
                        "vllm_cached_ratio": vllm_cached_ratio,
                        "vllm_prefill_ttft_seconds": vllm_prefill_ttft_seconds,
                        "vllm_decode_after_first_token_seconds": vllm_decode_after_first_token_seconds,
                        "vllm_scheduled_to_finish_seconds": float(gen_metrics.get("vllm_scheduled_to_finish_seconds", 0.0)),
                        "vllm_generate_to_first_token_seconds": vllm_generate_to_first_token_seconds,
                        "vllm_generate_after_first_token_seconds": vllm_generate_after_first_token_seconds,
                        "vllm_queue_seconds": vllm_queue_seconds,
                        "vllm_scheduler_seconds": vllm_scheduler_seconds,
                        "vllm_model_forward_seconds": vllm_model_forward_seconds,
                        "vllm_model_execute_seconds": vllm_model_execute_seconds,
                        "clip_path": clip_path,
                        "audio_path": audio_path,
                        "frame_paths": frame_paths,
                        "frame_count": len(frame_paths),
                        "response": response_text,
                        "think": think,
                        "observation": observation,
                    })

                    if sample_trace_dir is not None:
                        token_segments_file = ""
                        if token_segments_before:
                            token_segments_file = f"step_{step_n:03d}_token_segments.pt"
                            torch.save(
                                clone_token_segments_for_trace(token_segments_before),
                                sample_trace_dir / token_segments_file,
                            )
                        trace_steps.append({
                            "step": step_n,
                            "messages_before_step": make_json_safe(messages),
                            "token_segments_file": token_segments_file,
                            "response_text": response_text,
                            "response_ids": [int(x) for x in response_ids],
                            "action_type": action_type,
                            "think": think,
                            "observation": observation,
                            "confidence": confidence,
                            "reward": reward,
                            "cumulative_reward": total_reward,
                            "is_action_valid": is_action_valid,
                            "error_code": error_code,
                            "error_message": error_message,
                            "action_start": action_start,
                            "action_end": action_end,
                            "requested_duration_seconds": requested_duration_seconds,
                            "requested_num_frames": requested_num_frames,
                            "prompt_text_tokens": prompt_text_tokens,
                            "prompt_total_tokens": prompt_total_tokens,
                            "prompt_nontext_tokens": prompt_nontext_tokens,
                            "response_tokens": response_token_count,
                            "full_sequence_tokens": full_sequence_tokens,
                            "common_prefix_tokens": common_prefix_tokens,
                            "common_prefix_ratio": common_prefix_ratio,
                            "effective_new_prompt_tokens": effective_new_prompt_tokens,
                            "prompt_build_seconds": prompt_build_seconds,
                            "multimodal_prep_seconds": multimodal_prep_seconds,
                            "generation_seconds_vllm": generation_seconds,
                            "vllm_prefill_ttft_seconds": vllm_prefill_ttft_seconds,
                            "vllm_decode_after_first_token_seconds": vllm_decode_after_first_token_seconds,
                            "vllm_generate_to_first_token_seconds": vllm_generate_to_first_token_seconds,
                            "vllm_generate_after_first_token_seconds": vllm_generate_after_first_token_seconds,
                            "vllm_queue_seconds": vllm_queue_seconds,
                            "env_step_seconds": env_step_seconds,
                            "postprocess_seconds": postprocess_seconds,
                            "overhead_seconds": overhead_seconds,
                            "step_wall_seconds": step_wall_seconds,
                        })

                    obs = next_obs

                ended_at = utc_now()
                episode_seconds = time.time() - episode_start

                result_record.update({
                    "ended_at": ended_at,
                    "episode_seconds": episode_seconds,
                    "ffprobe_seconds": ffprobe_time,
                    "reset_seconds": reset_time,
                    "final_answer": final_answer,
                    "total_reward": total_reward,
                    "steps_used": len(result_record["steps"]),
                    "tool_usage_counts": dict(tool_counter),
                    "tool_sequence": tool_sequence,
                    "valid_step_count": valid_step_count,
                    "invalid_step_count": invalid_step_count,
                    "common_prefix_step_count": common_prefix_step_count,
                    "prompt_token_ids_step_count": prompt_token_ids_step_count,
                    "prompt_token_ids_requested_step_count": prompt_token_ids_requested_step_count,
                    "prompt_token_ids_success_rate": (float(prompt_token_ids_step_count) / float(prompt_token_ids_requested_step_count)) if prompt_token_ids_requested_step_count else 0.0,
                    "total_common_prefix_tokens": total_common_prefix_tokens,
                    "total_effective_new_prompt_tokens": total_effective_new_prompt_tokens,
                    "avg_common_prefix_tokens_per_step": (float(total_common_prefix_tokens) / float(len(result_record["steps"]))) if result_record["steps"] else 0.0,
                    "avg_common_prefix_ratio_per_step": mean_or_zero([float(step.get("common_prefix_ratio", 0.0)) for step in result_record["steps"]]),
                    "prefix_reuse_ratio": (float(total_common_prefix_tokens) / float(total_prompt_total_tokens)) if total_prompt_total_tokens else 0.0,
                    "total_vllm_cached_tokens": total_vllm_cached_tokens,
                    "avg_vllm_cached_tokens_per_step": (float(total_vllm_cached_tokens) / float(len(result_record["steps"]))) if result_record["steps"] else 0.0,
                    "avg_vllm_cached_ratio_per_step": mean_or_zero([float(step.get("vllm_cached_ratio", 0.0)) for step in result_record["steps"]]),
                    "vllm_cached_step_count": vllm_cached_step_count,
                    "total_vllm_prefill_ttft_seconds": total_vllm_prefill_ttft_seconds,
                    "total_vllm_decode_after_first_token_seconds": total_vllm_decode_after_first_token_seconds,
                    "total_vllm_generate_to_first_token_seconds": total_vllm_generate_to_first_token_seconds,
                    "total_vllm_generate_after_first_token_seconds": total_vllm_generate_after_first_token_seconds,
                    "total_vllm_queue_seconds": total_vllm_queue_seconds,
                    "total_vllm_model_forward_seconds": total_vllm_model_forward_seconds,
                    "total_vllm_model_execute_seconds": total_vllm_model_execute_seconds,
                    "total_response_tokens": total_response_tokens,
                    "total_prompt_text_tokens": total_prompt_text_tokens,
                    "total_prompt_total_tokens": total_prompt_total_tokens,
                    "total_prompt_nontext_tokens": total_prompt_nontext_tokens,
                    "total_full_sequence_tokens": total_full_sequence_tokens,
                    "valid_total_response_tokens": valid_total_response_tokens,
                    "valid_total_prompt_text_tokens": valid_total_prompt_text_tokens,
                    "valid_total_prompt_total_tokens": valid_total_prompt_total_tokens,
                    "valid_total_prompt_nontext_tokens": valid_total_prompt_nontext_tokens,
                    "valid_total_full_sequence_tokens": valid_total_full_sequence_tokens,
                    "invalid_total_response_tokens": invalid_total_response_tokens,
                    "invalid_total_prompt_text_tokens": invalid_total_prompt_text_tokens,
                    "invalid_total_prompt_total_tokens": invalid_total_prompt_total_tokens,
                    "invalid_total_prompt_nontext_tokens": invalid_total_prompt_nontext_tokens,
                    "invalid_total_full_sequence_tokens": invalid_total_full_sequence_tokens,
                    "valid_total_prompt_build_seconds": valid_total_prompt_build_seconds,
                    "valid_total_multimodal_prep_seconds": valid_total_multimodal_prep_seconds,
                    "valid_total_generation_seconds": valid_total_generation_seconds,
                    "valid_total_env_step_seconds": valid_total_env_step_seconds,
                    "valid_total_postprocess_seconds": valid_total_postprocess_seconds,
                    "valid_total_overhead_seconds": valid_total_overhead_seconds,
                    "valid_total_step_wall_seconds": valid_total_step_wall_seconds,
                    "invalid_total_prompt_build_seconds": invalid_total_prompt_build_seconds,
                    "invalid_total_multimodal_prep_seconds": invalid_total_multimodal_prep_seconds,
                    "invalid_total_generation_seconds": invalid_total_generation_seconds,
                    "invalid_total_env_step_seconds": invalid_total_env_step_seconds,
                    "invalid_total_postprocess_seconds": invalid_total_postprocess_seconds,
                    "invalid_total_overhead_seconds": invalid_total_overhead_seconds,
                    "invalid_total_step_wall_seconds": invalid_total_step_wall_seconds,
                    "avg_response_tokens_per_step": (total_response_tokens / len(result_record["steps"])) if result_record["steps"] else 0.0,
                    "avg_prompt_text_tokens_per_step": (total_prompt_text_tokens / len(result_record["steps"])) if result_record["steps"] else 0.0,
                    "avg_prompt_total_tokens_per_step": (total_prompt_total_tokens / len(result_record["steps"])) if result_record["steps"] else 0.0,
                    "avg_prompt_nontext_tokens_per_step": (total_prompt_nontext_tokens / len(result_record["steps"])) if result_record["steps"] else 0.0,
                    "avg_full_sequence_tokens_per_step": (total_full_sequence_tokens / len(result_record["steps"])) if result_record["steps"] else 0.0,
                    "correct": bool(total_reward > 0.5),
                    "env_info": infos[0] if infos else {},
                })
                result_record["media_usage"] = build_media_usage_summary(result_record["steps"])
                result_record["latency"] = build_latency_summary(result_record["steps"])

                history_json = generate_history_json(
                    [],
                    full_messages,
                    sample["video"],
                    sample["question"],
                    runtime_config=runtime_payload["runtime_config"],
                    env_snapshot=runtime_payload["env_snapshot"],
                    include_messages=True,
                )
                result_record["final_history"] = json.loads(history_json)
                if sample_trace_dir is not None:
                    trace_manifest = {
                        "source_backend": "vllm",
                        "sample_index": sample_index,
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "model_path": config.model_path,
                        "video": sample["video"],
                        "question_type": sample["question_type"],
                        "question": sample["question"],
                        "ground_truth": sample["answer"],
                        "options": sample["options_text"].splitlines() if sample["options_text"] else [],
                        "runtime_config": runtime_payload["runtime_config"],
                        "env_snapshot": runtime_payload["env_snapshot"],
                        "steps": trace_steps,
                    }
                    trace_manifest_path = sample_trace_dir / "trace_manifest.json"
                    trace_manifest_path.write_text(
                        json.dumps(trace_manifest, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    result_record["trace_manifest"] = str(trace_manifest_path)
            except Exception as e:
                result_record.update({
                    "status": "error",
                    "error": str(e),
                    "ended_at": utc_now(),
                    "episode_seconds": time.time() - episode_start,
                })

            result_queue.put(result_record)
            task_queue.task_done()
    except Exception:
        result_queue.put({
            "type": "worker_fatal",
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "error": traceback.format_exc(),
        })
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_ids", default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--max_samples", type=int, default=-1)
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
    parser.add_argument("--sample_indices", default="", help="Optional comma-separated original dataset sample indices.")
    parser.add_argument("--trace_export_dir", default="", help="Optional directory to export frozen per-step traces.")
    return parser.parse_args()


def build_config_dict(args) -> Dict[str, Any]:
    use_dynamic_step = True
    if args.no_dynamic_step:
        use_dynamic_step = False
    elif args.dynamic_step:
        use_dynamic_step = True
    return {
        "model_path": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_steps": args.max_steps,
        "max_frames_len": args.max_frames_len,
        "max_audio_len": args.max_audio_len,
        "max_clip_len": args.max_clip_len,
        "mode": args.mode,
        "use_dynamic_step": use_dynamic_step,
        "use_tito": bool(args.use_tito),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }


def main():
    args = parse_args()
    output_dir = resolve_output_dir(Path(args.output_dir), args.model_path, args.dataset_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    sample_csv_path = output_dir / "samples.csv"
    summary_csv_path = output_dir / "summary.csv"
    breakdown_json_path = output_dir / "breakdowns.json"
    metadata_path = output_dir / "run_metadata.json"

    selected_sample_indices = parse_sample_indices_arg(args.sample_indices)
    load_limit = -1 if selected_sample_indices else args.max_samples
    samples = [normalize_sample(s, i) for i, s in enumerate(load_jsonl(args.dataset_jsonl, load_limit))]
    if selected_sample_indices:
        sample_map = {sample["sample_index"]: sample for sample in samples}
        missing = [idx for idx in selected_sample_indices if idx not in sample_map]
        if missing:
            raise ValueError(f"sample_indices not found in dataset: {missing}")
        samples = [sample_map[idx] for idx in selected_sample_indices]

    total_samples = len(samples)
    completed_indices: set = set()
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    completed_indices.add(rec["sample_index"])
                except (json.JSONDecodeError, KeyError):
                    continue
        if completed_indices:
            samples = [s for s in samples if s["sample_index"] not in completed_indices]
            print(f"[resume] found {len(completed_indices)} completed samples in {results_path}, "
                  f"remaining {len(samples)}/{total_samples}")
            if not samples:
                print("[resume] all samples already processed, nothing to do.")
                return

    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids provided.")
    if args.tensor_parallel_size != 1:
        raise ValueError("This evaluator currently requires tensor_parallel_size=1 for one-worker-per-GPU scheduling.")

    ctx = mp.get_context("spawn")
    task_queue: mp.JoinableQueue = ctx.JoinableQueue()
    result_queue: mp.Queue = ctx.Queue()
    config_dict = build_config_dict(args)

    workers: List[mp.Process] = []
    for worker_id, gpu_id in enumerate(gpu_ids):
        proc = ctx.Process(
            target=worker_main,
            args=(worker_id, gpu_id, config_dict, task_queue, result_queue, args.trace_export_dir),
            daemon=False,
        )
        proc.start()
        workers.append(proc)

    for sample in samples:
        task_queue.put({"sample": sample})
    for _ in workers:
        task_queue.put(None)

    collected: List[Dict[str, Any]] = []
    worker_errors: List[Dict[str, Any]] = []
    expected_results = len(samples)
    run_started_wall = time.time()
    error_count = 0

    metadata = {
        "created_at": utc_now(),
        "dataset_jsonl": args.dataset_jsonl,
        "model_path": args.model_path,
        "output_dir": str(output_dir),
        "gpu_ids": gpu_ids,
        "num_samples": total_samples,
        "num_resumed": len(completed_indices),
        "sample_indices": selected_sample_indices or [],
        "trace_export_dir": args.trace_export_dir,
        "runtime_config": build_config_dict(args),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(results_path, "a" if completed_indices else "w", encoding="utf-8") as out_f:
        last_heartbeat = 0.0
        while True:
            try:
                item = result_queue.get(timeout=5)
            except queue.Empty:
                alive_workers = [p for p in workers if p.is_alive()]
                if not alive_workers:
                    break
                now = time.time()
                if now - last_heartbeat >= 10:
                    elapsed = now - run_started_wall
                    print(
                        f"[progress] ok={len(collected)} errors={error_count} "
                        f"alive_workers={len(alive_workers)} "
                        f"elapsed={format_seconds(elapsed)}"
                    )
                    last_heartbeat = now
                continue

            if item.get("type") == "worker_error":
                worker_errors.append(item)
                raise RuntimeError(f"Worker {item['worker_id']} failed to initialize: {item['error']}")
            if item.get("type") == "worker_fatal":
                raise RuntimeError(f"Worker {item['worker_id']} crashed:\n{item['error']}")

            if item.get("status") != "ok":
                error_count += 1
                print(f"[error] sample={item.get('sample_index')} gpu={item.get('gpu_id')} "
                      f"error={str(item.get('error', 'unknown'))[:200]}")
                continue

            collected.append(item)
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_f.flush()
            elapsed = time.time() - run_started_wall
            avg_per_sample = elapsed / len(collected) if collected else 0.0
            eta_seconds = avg_per_sample * (expected_results - len(collected))
            print(
                f"[{len(collected)}/{expected_results}] "
                f"sample={item['sample_index']} gpu={item['gpu_id']} "
                f"reward={item.get('total_reward', 0.0):.4f} "
                f"gen={item.get('latency', {}).get('total_generation_seconds', 0.0):.2f}s "
                f"env={item.get('latency', {}).get('total_env_step_seconds', 0.0):.2f}s "
                f"wall={item.get('episode_seconds', 0.0):.2f}s "
                f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta_seconds)}"
            )
            last_heartbeat = time.time()

    if error_count:
        print(f"[warning] {error_count} samples failed, will be retried on next resume")

    task_queue.join()
    for proc in workers:
        proc.join(timeout=5)

    if completed_indices:
        all_results = []
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        print(f"[resume] statistics computed on all {len(all_results)} samples "
              f"({len(completed_indices)} resumed + {len(collected)} new)")
    else:
        all_results = collected

    ok_results = [r for r in all_results if r.get("status") == "ok"]
    error_results = [r for r in all_results if r.get("status") != "ok"]
    rewards = [float(r.get("total_reward", 0.0)) for r in ok_results]
    steps = [int(r.get("steps_used", 0)) for r in ok_results]
    durations = [float(r.get("episode_seconds", 0.0)) for r in ok_results]
    response_tokens = [int(r.get("total_response_tokens", 0)) for r in ok_results]
    clip_seconds = [float(r.get("media_usage", {}).get("total_clip_seconds", 0.0)) for r in ok_results]
    audio_seconds = [float(r.get("media_usage", {}).get("total_audio_seconds", 0.0)) for r in ok_results]
    requested_frames = [int(r.get("media_usage", {}).get("total_requested_frames", 0)) for r in ok_results]
    returned_frames = [int(r.get("media_usage", {}).get("total_returned_frames", 0)) for r in ok_results]
    clip_calls = [int(r.get("media_usage", {}).get("clip_calls", 0)) for r in ok_results]
    audio_calls = [int(r.get("media_usage", {}).get("audio_calls", 0)) for r in ok_results]
    frame_calls = [int(r.get("media_usage", {}).get("frame_calls", 0)) for r in ok_results]
    tool_counter = Counter()
    for r in ok_results:
        tool_counter.update(r.get("tool_usage_counts", {}))

    total_clip_calls = sum(clip_calls)
    total_audio_calls = sum(audio_calls)
    total_frame_calls = sum(frame_calls)
    total_clip_seconds = sum(clip_seconds)
    total_audio_seconds = sum(audio_seconds)
    total_requested_frames = sum(requested_frames)
    total_returned_frames = sum(returned_frames)

    summary = {
        "started_at": metadata["created_at"],
        "finished_at": utc_now(),
        "dataset_jsonl": args.dataset_jsonl,
        "output_dir": str(output_dir),
        "model_path": args.model_path,
        "total_episode_seconds": sum(r.get("episode_seconds", 0.0) for r in ok_results),
        "media_usage_summary": {
            "total_clip_calls": total_clip_calls,
            "total_audio_calls": total_audio_calls,
            "total_frame_calls": total_frame_calls,
            "total_clip_seconds": total_clip_seconds,
            "total_audio_seconds": total_audio_seconds,
            "total_requested_frames": total_requested_frames,
            "total_returned_frames": total_returned_frames,
            "avg_clip_seconds_per_call": (total_clip_seconds / total_clip_calls) if total_clip_calls else 0.0,
            "avg_audio_seconds_per_call": (total_audio_seconds / total_audio_calls) if total_audio_calls else 0.0,
            "avg_requested_frames_per_call": (total_requested_frames / total_frame_calls) if total_frame_calls else 0.0,
            "avg_returned_frames_per_call": (total_returned_frames / total_frame_calls) if total_frame_calls else 0.0,
            "avg_clip_calls_per_sample": statistics.mean(clip_calls) if clip_calls else 0.0,
            "avg_audio_calls_per_sample": statistics.mean(audio_calls) if audio_calls else 0.0,
            "avg_frame_calls_per_sample": statistics.mean(frame_calls) if frame_calls else 0.0,
        },
        "gpu_ids": gpu_ids,
        "runtime_config": build_config_dict(args),
    }
    summary.update(compute_group_metrics(all_results))

    breakdowns = build_breakdowns(all_results)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    breakdown_json_path.write_text(json.dumps(breakdowns, indent=2, ensure_ascii=False), encoding="utf-8")

    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "num_samples",
            "num_success",
            "num_error",
            "success_rate",
            "accuracy_reward_gt_0_5",
            "avg_reward",
            "avg_steps",
            "avg_episode_seconds",
            "avg_response_tokens",
            "avg_prompt_text_tokens",
            "avg_prompt_total_tokens",
            "avg_prompt_nontext_tokens",
            "avg_full_sequence_tokens",
            "avg_valid_step_count",
            "avg_invalid_step_count",
            "avg_valid_prompt_total_tokens",
            "avg_valid_prompt_nontext_tokens",
            "avg_valid_full_sequence_tokens",
            "avg_invalid_prompt_total_tokens",
            "avg_invalid_prompt_nontext_tokens",
            "avg_invalid_full_sequence_tokens",
            "avg_valid_generation_seconds",
            "avg_valid_env_step_seconds",
            "avg_valid_step_wall_seconds",
            "avg_invalid_generation_seconds",
            "avg_invalid_env_step_seconds",
            "avg_invalid_step_wall_seconds",
            "avg_total_clip_seconds_per_sample",
            "avg_total_audio_seconds_per_sample",
            "avg_total_requested_frames_per_sample",
            "avg_total_returned_frames_per_sample",
            "avg_clip_calls_per_sample",
            "avg_audio_calls_per_sample",
            "avg_frame_calls_per_sample",
            "avg_total_prompt_build_seconds_per_sample",
            "avg_total_multimodal_prep_seconds_per_sample",
            "avg_total_generation_seconds_per_sample",
            "avg_total_env_step_seconds_per_sample",
            "avg_model_only_seconds_per_sample",
            "avg_model_pipeline_seconds_per_sample",
            "avg_env_only_seconds_per_sample",
            "avg_total_postprocess_seconds_per_sample",
            "avg_total_overhead_seconds_per_sample",
            "avg_total_common_prefix_tokens",
            "avg_total_effective_new_prompt_tokens",
            "avg_prompt_token_ids_step_count",
            "avg_prompt_token_ids_requested_step_count",
            "avg_prompt_token_ids_success_rate",
            "avg_common_prefix_step_count",
            "avg_common_prefix_tokens_per_step",
            "avg_common_prefix_ratio_per_step",
            "avg_prefix_reuse_ratio",
            "avg_total_vllm_cached_tokens",
            "avg_vllm_cached_ratio_per_step",
            "avg_total_vllm_prefill_ttft_seconds_per_sample",
            "avg_total_vllm_decode_after_first_token_seconds_per_sample",
            "avg_total_vllm_generate_to_first_token_seconds_per_sample",
            "avg_total_vllm_generate_after_first_token_seconds_per_sample",
            "avg_total_vllm_queue_seconds_per_sample",
            "avg_total_vllm_model_forward_seconds_per_sample",
            "avg_total_vllm_model_execute_seconds_per_sample",
            "avg_prompt_build_seconds_per_step",
            "avg_multimodal_prep_seconds_per_step",
            "avg_generation_seconds_per_step",
            "avg_env_step_seconds_per_step",
            "avg_model_only_seconds_per_step",
            "avg_model_pipeline_seconds_per_step",
            "avg_env_only_seconds_per_step",
            "avg_overhead_seconds_per_step",
            "avg_vllm_prefill_ttft_seconds_per_step",
            "avg_vllm_decode_after_first_token_seconds_per_step",
            "avg_vllm_generate_to_first_token_seconds_per_step",
            "avg_vllm_generate_after_first_token_seconds_per_step",
            "p50_generation_seconds_per_step",
            "p90_generation_seconds_per_step",
            "p95_generation_seconds_per_step",
            "p50_env_step_seconds_per_step",
            "p90_env_step_seconds_per_step",
            "p95_env_step_seconds_per_step",
            "p50_vllm_prefill_ttft_seconds_per_step",
            "p90_vllm_prefill_ttft_seconds_per_step",
            "p95_vllm_prefill_ttft_seconds_per_step",
            "p50_vllm_decode_after_first_token_seconds_per_step",
            "p90_vllm_decode_after_first_token_seconds_per_step",
            "p95_vllm_decode_after_first_token_seconds_per_step",
            "tool_usage_counts",
            "total_episode_seconds",
            "output_dir",
            "dataset_jsonl",
            "model_path",
        ])
        writer.writeheader()
        row = {k: summary.get(k) for k in writer.fieldnames}
        row["tool_usage_counts"] = json.dumps(summary.get("tool_usage_counts", {}), ensure_ascii=False)
        writer.writerow(row)

    write_sample_csv(sample_csv_path, all_results)
    for field_name, breakdown in breakdowns.items():
        write_breakdown_csv(output_dir / f"summary_by_{field_name}.csv", field_name, breakdown)

    print("\nSummary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nArtifacts")
    print(f"  results_jsonl: {results_path}")
    print(f"  summary_json: {summary_path}")
    print(f"  summary_csv: {summary_csv_path}")
    print(f"  sample_csv: {sample_csv_path}")
    print(f"  breakdowns_json: {breakdown_json_path}")


if __name__ == "__main__":
    main()
