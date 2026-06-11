#!/usr/bin/env python3
"""
OmniAgent Demo Pro - one media box per step, scrollable
"""

import os
import sys
import json
import argparse
import subprocess
import traceback
import shutil
import html
import re
import tempfile
import time
import warnings
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import quote

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.update({
    "VLLM_USE_V1": "0",
    "TOKENIZERS_PARALLELISM": "False",
    "USE_OSS_IN_VIDEOENV": "False",
    "USE_DYNAMIC_STEP": "True",
    "USE_TITO": "False",
    "DELETE_MEDIA_WHEN_ERROR": "True",
    "STRICT_CONFIDENCE_CHECK": "False",
    "MIN_MAX_STEPS": "5",
    "FORCE_QWENVL_VIDEO_READER": "True",
    "BYPASS_DURATION_CHECK": "True",
    "RAY_DEDUP_LOGS": "0",
    # Ray timeout configuration - prevent GCS disconnection
    "RAY_gcs_rpc_server_reconnect_timeout_s": "3000",  # 5-minute reconnection timeout
    "RAY_gcs_server_request_timeout_seconds": "3000",
    "RAY_timeout_ms": "3000000",  # 5 minutes
    "OMNIAGENT_QUIET_LOGS": os.environ.get("OMNIAGENT_QUIET_LOGS", "true"),
})

warnings.filterwarnings("ignore", message=".*System prompt modified, audio output may not work as expected.*")
warnings.filterwarnings("ignore", message=".*video decoding and encoding capabilities of torchvision are deprecated.*")
if os.environ.get("OMNIAGENT_QUIET_LOGS", "true").lower() in ("1", "true", "t", "yes", "y"):
    logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("qwen_vl_utils").setLevel(logging.ERROR)
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("vllm").setLevel(logging.ERROR)

KEY_RUNTIME_ENVS = [
    "USE_DYNAMIC_STEP",
    "USE_TITO",
    "STRICT_CONFIDENCE_CHECK",
    "MIN_MAX_STEPS",
    "FORCE_QWENVL_VIDEO_READER",
    "DELETE_MEDIA_WHEN_ERROR",
    "BYPASS_DURATION_CHECK",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOMME_ROOT = os.environ.get("OMNIAGENT_VIDEOMME_ROOT", "")
LVBENCH_ROOT = os.environ.get("OMNIAGENT_LVBENCH_ROOT", "")
VIDI_ROOT = os.environ.get("OMNIAGENT_VIDI_ROOT", "")


def demo_data_path(root: str, relative_path: str) -> str:
    if not root:
        return ""
    return str(Path(root) / relative_path)


def build_allowed_paths() -> List[str]:
    paths = {
        str(PROJECT_ROOT),
        str(Path(tempfile.gettempdir())),
    }
    video_env_tmp_dir = os.environ.get("VIDEO_ENV_TMP_DIR", "./video_env_tmp")
    try:
        paths.add(str((PROJECT_ROOT / video_env_tmp_dir).resolve()))
    except Exception:
        pass
    for root in [VIDEOMME_ROOT, LVBENCH_ROOT, VIDI_ROOT]:
        if root:
            paths.add(str(Path(root).resolve()))
    extra = os.environ.get("OMNIAGENT_ALLOWED_PATHS", "")
    for item in extra.split(":"):
        item = item.strip()
        if item:
            paths.add(str(Path(item).resolve()))
    return sorted(paths)

import gradio as gr

from agent_system.environments.env_manager import VideoEnvironmentManager
from agent_system.environments.env_package.video_env import build_video_envs
from agent_system.environments.env_package.oss_reader import OssReader

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from qwen_omni_utils import process_audio_info

oss_reader = OssReader()

MAX_STEPS = 50
STATUS_READY = "READY"
STATUS_LOADING_MODEL = "LOADING_MODEL"
STATUS_INITIALIZING_RUNTIME = "INITIALIZING_RUNTIME"
STATUS_RUNNING = "RUNNING"
STATUS_STOPPING = "STOPPING"
STATUS_DONE = "DONE"
STATUS_ERROR = "ERROR"
STATUS_BLOCKED = "BLOCKED"


@dataclass
class DemoConfig:
    model_path: str = ""
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.6
    max_model_len: int = 131072
    max_prompt_length: int = 65536
    max_response_length: int = 4096
    max_num_batched_tokens: int = 131072
    max_steps: int = 32  # test-time scaling: try 12, 22, 32, 42, 52
    max_frames_len: int = 60
    max_audio_len: float = 300.0
    max_clip_len: float = 60.0
    mode: str = "OmniAgent"
    use_dynamic_step: bool = True
    use_tito: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20


@dataclass
class StepInfo:
    step: int
    action_type: str
    think: str
    observation: str
    confidence: Optional[float]
    response: str
    reward: float
    prompt_build_seconds: float = 0.0
    multimodal_prep_seconds: float = 0.0
    generation_seconds: float = 0.0
    env_step_seconds: float = 0.0
    postprocess_seconds: float = 0.0
    overhead_seconds: float = 0.0
    step_wall_seconds: float = 0.0
    prompt_text_tokens: int = 0
    prompt_total_tokens: int = 0
    prompt_nontext_tokens: int = 0
    response_tokens: int = 0
    full_sequence_tokens: int = 0
    clip_path: Optional[str] = None
    audio_path: Optional[str] = None
    frame_paths: List[str] = field(default_factory=list)


BUILTIN_EXAMPLES = [
    {
        "name": "Immigrant Diaries (MCQ)",
        "video": str(PROJECT_ROOT / "assets/example_video_mcq.mp4"),
        "question": 'Who or what lauds "Immigrant Diaries" as "A SURE FIRE HIT", according to the video?',
        "answer": "A",
        "type": "MCQ",
        "options": "A. Remote Goat.\nB. The New York Times.\nC. Variety.\nD. IndieWire.",
    },
    {
        "name": "Interview Grounding (TR)",
        "video": str(PROJECT_ROOT / "assets/example_video_tr.mp4"),
        "question": "What are all the time ranges corresponding to the text query: \"A man with tousled dark hair and a beaded necklace thoughtfully shares his perspective, the subtle floral pattern of his light green shirt contrasting against the light-colored wall behind him as he speaks about challenges and roles.\"?",
        "answer": "[51.72, 62.92]",
        "type": "TR",
        "options": "",
    },
    {
        "name": "White Horse (FF)",
        "video": str(PROJECT_ROOT / "assets/example_video_ff.mp4"),
        "question": "During the montage, what color was the horse that the boy in yellow is riding?",
        "answer": "White",
        "type": "FF",
        "options": "",
    },
]


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.llm = None
        self.processor = None
        self.sampling_params = None
        self.config = None
        self.is_loaded = False
        self._env_manager = None

    def load(self, config: DemoConfig) -> Tuple[bool, str]:
        if self.is_loaded:
            return True, "✅ Already loaded"

        try:
            if not VLLM_AVAILABLE:
                return False, "❌ vLLM not available"

            self.llm = LLM(
                model=config.model_path,
                tensor_parallel_size=config.tensor_parallel_size,
                gpu_memory_utilization=config.gpu_memory_utilization,
                max_model_len=config.max_model_len,
                max_num_batched_tokens=config.max_num_batched_tokens,
                trust_remote_code=True,
                dtype="bfloat16",
                limit_mm_per_prompt={"image": config.max_frames_len, "video": 1, "audio": 1}
            )
            self.processor = AutoProcessor.from_pretrained(config.model_path, trust_remote_code=True)
            self.update_sampling_params(config)
            self.config = config
            self.is_loaded = True
            return True, "✅ Loaded!"
        except Exception as e:
            return False, f"❌ {str(e)}"

    def unload(self):
        if self._env_manager:
            try: self._env_manager.close()
            except: pass
        self._env_manager = None
        self.llm = None
        self.processor = None
        self.sampling_params = None
        self.is_loaded = False
        return "✅ Unloaded"

    def update_sampling_params(self, config: DemoConfig):
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_response_length,
            stop=["<|im_end|>", "```"],
        )

    def build_env(self, config: DemoConfig):
        if self._env_manager is not None:
            try:
                self._env_manager.close()
            except Exception:
                pass
            self._env_manager = None

        class C: pass
        ec = C()
        ec.env = C()
        ec.env.env_name, ec.env.seed, ec.env.max_steps = "video_env", 42, config.max_steps
        ec.env.rollout = C()
        ec.env.rollout.n = 1
        ec.env.video_star = C()
        ec.env.video_star.max_frames_len = config.max_frames_len
        ec.env.video_star.max_audio_len = config.max_audio_len
        ec.env.video_star.max_clip_len = config.max_clip_len
        ec.env.video_star.mode = config.mode

        envs = build_video_envs(
            seed=142, env_num=1, group_n=1,
            max_frames_len=config.max_frames_len, max_audio_len=config.max_audio_len,
            max_clip_len=config.max_clip_len, max_steps=config.max_steps,
            processor_path=config.model_path, max_prompt_len=config.max_prompt_length,
            max_response_len=config.max_response_length, is_train=False
        )
        self._env_manager = VideoEnvironmentManager(envs, lambda a, *args, **kwargs: (a, [True] * len(a)), ec)

    def generate(self, messages, return_metrics: bool = False, sample_has_audio: bool = True):
        prompt_started = time.time()
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(prompt, list): prompt = prompt[0] if prompt else ""
        prompt_build_seconds = time.time() - prompt_started
        prompt_text_tokens = len(self.processor.tokenizer.encode(prompt, add_special_tokens=False))

        mm_started = time.time()
        imgs, vids, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        has_audio = bool(sample_has_audio)
        audios = None
        if has_audio:
            try: audios = process_audio_info(messages, use_audio_in_video=True)
            except: pass

        mm_data, mm_kwargs = {}, {}
        if imgs: mm_data["image"] = imgs
        if vids: mm_data["video"] = vids
        if video_kwargs:
            for key, val in video_kwargs.items():
                if isinstance(val, list) and len(val) == 1:
                    mm_kwargs[key] = val[0]
                else:
                    mm_kwargs[key] = val
        if audios: mm_data["audio"], mm_kwargs["use_audio_in_video"] = audios, True

        inp = {"prompt": prompt}
        if mm_data: inp["multi_modal_data"] = mm_data
        if mm_kwargs: inp["mm_processor_kwargs"] = mm_kwargs
        multimodal_prep_seconds = time.time() - mm_started

        token_started = time.time()
        prompt_total_tokens = prompt_text_tokens
        token_count_error = ""
        prompt_token_ids_internal = None
        prompt_token_ids_requested = True
        prompt_token_ids_available = False
        try:
            processor_kwargs = {"text": [prompt], "return_tensors": "pt"}
            if imgs:
                processor_kwargs["images"] = imgs
            if vids:
                processor_kwargs["videos"] = vids
            if video_kwargs:
                processor_kwargs.update(video_kwargs)
            if audios:
                processor_kwargs["audio"] = audios
                processor_kwargs["use_audio_in_video"] = True
            else:
                processor_kwargs["use_audio_in_video"] = False
            proc_out = self.processor(**processor_kwargs)
            input_ids = proc_out.get("input_ids")
            if input_ids is not None:
                prompt_total_tokens = int(input_ids[0].numel())
                try:
                    prompt_token_ids_internal = [int(x) for x in input_ids[0].detach().cpu().tolist()]
                    prompt_token_ids_available = True
                except Exception as e:
                    prompt_token_ids_internal = None
                    prompt_token_ids_available = False
                    token_count_error = f"{token_count_error} | prompt_token_ids_extract_failed: {e}" if token_count_error else f"prompt_token_ids_extract_failed: {e}"
        except Exception as e:
            token_count_error = str(e)
        token_count_seconds = time.time() - token_started
        prompt_nontext_tokens = max(0, int(prompt_total_tokens) - int(prompt_text_tokens))

        llm_started = time.time()
        out = self.llm.generate(inp, self.sampling_params, use_tqdm=False)
        llm_generate_seconds = time.time() - llm_started
        request_metrics = out[0].metrics if out else None
        num_cached_tokens_raw = out[0].num_cached_tokens if out else None

        def _optional_float(value):
            try:
                return float(value) if value is not None else None
            except Exception:
                return None

        def _nonneg_delta(start, end):
            if start is None or end is None:
                return 0.0, False
            delta = float(end) - float(start)
            if delta < 0.0:
                return 0.0, False
            return delta, True

        first_scheduled_time = _optional_float(getattr(request_metrics, "first_scheduled_time", None))
        first_token_time = _optional_float(getattr(request_metrics, "first_token_time", None))
        finished_time = _optional_float(getattr(request_metrics, "finished_time", None))
        time_in_queue = _optional_float(getattr(request_metrics, "time_in_queue", None))
        scheduler_time = _optional_float(getattr(request_metrics, "scheduler_time", None))
        model_forward_time = _optional_float(getattr(request_metrics, "model_forward_time", None))
        model_execute_time = _optional_float(getattr(request_metrics, "model_execute_time", None))

        vllm_prefill_ttft_seconds, vllm_prefill_ttft_available = _nonneg_delta(first_scheduled_time, first_token_time)
        vllm_decode_after_first_token_seconds, vllm_decode_after_first_token_available = _nonneg_delta(first_token_time, finished_time)
        vllm_scheduled_to_finish_seconds, vllm_scheduled_to_finish_available = _nonneg_delta(first_scheduled_time, finished_time)
        vllm_generate_to_first_token_seconds, vllm_generate_to_first_token_available = _nonneg_delta(llm_started, first_token_time)
        vllm_generate_after_first_token_seconds = 0.0
        vllm_generate_after_first_token_available = False
        if vllm_generate_to_first_token_available:
            residual = llm_generate_seconds - vllm_generate_to_first_token_seconds
            if residual >= 0.0:
                vllm_generate_after_first_token_seconds = residual
                vllm_generate_after_first_token_available = True

        vllm_queue_seconds = max(0.0, float(time_in_queue or 0.0))
        vllm_queue_seconds_available = time_in_queue is not None
        metrics = {
            "prompt_build_seconds": prompt_build_seconds,
            "multimodal_prep_seconds": multimodal_prep_seconds,
            "token_count_seconds": token_count_seconds,
            "llm_generate_seconds": llm_generate_seconds,
            "generate_call_seconds": prompt_build_seconds + multimodal_prep_seconds + llm_generate_seconds,
            "prompt_text_tokens": int(prompt_text_tokens),
            "prompt_total_tokens": int(prompt_total_tokens),
            "prompt_nontext_tokens": int(prompt_nontext_tokens),
            "token_count_error": token_count_error,
            "prompt_token_ids_requested": prompt_token_ids_requested,
            "prompt_token_ids_available": prompt_token_ids_available,
            "prompt_token_ids_internal": prompt_token_ids_internal,
            "vllm_metrics_available": bool(request_metrics is not None),
            "vllm_queue_seconds": vllm_queue_seconds,
            "vllm_queue_seconds_available": vllm_queue_seconds_available,
            "vllm_scheduler_seconds": max(0.0, float(scheduler_time or 0.0)),
            "vllm_scheduler_seconds_available": scheduler_time is not None,
            "vllm_model_forward_seconds": max(0.0, float(model_forward_time or 0.0)),
            "vllm_model_forward_seconds_available": model_forward_time is not None,
            "vllm_model_execute_seconds": max(0.0, float(model_execute_time or 0.0)),
            "vllm_model_execute_seconds_available": model_execute_time is not None,
            "vllm_prefill_ttft_seconds": vllm_prefill_ttft_seconds,
            "vllm_prefill_ttft_available": vllm_prefill_ttft_available,
            "vllm_decode_after_first_token_seconds": vllm_decode_after_first_token_seconds,
            "vllm_decode_after_first_token_available": vllm_decode_after_first_token_available,
            "vllm_scheduled_to_finish_seconds": vllm_scheduled_to_finish_seconds,
            "vllm_scheduled_to_finish_available": vllm_scheduled_to_finish_available,
            "vllm_generate_to_first_token_seconds": vllm_generate_to_first_token_seconds,
            "vllm_generate_to_first_token_available": vllm_generate_to_first_token_available,
            "vllm_generate_after_first_token_seconds": vllm_generate_after_first_token_seconds,
            "vllm_generate_after_first_token_available": vllm_generate_after_first_token_available,
            "vllm_cached_tokens": int(num_cached_tokens_raw) if num_cached_tokens_raw is not None else 0,
            "vllm_cached_tokens_available": num_cached_tokens_raw is not None,
        }
        if not out or not out[0].outputs:
            return ("", [], metrics) if return_metrics else ("", [])
        sample = out[0].outputs[0]
        response_ids = list(sample.token_ids)
        metrics["response_tokens"] = int(len(response_ids))
        metrics["full_sequence_tokens"] = int(prompt_total_tokens + len(response_ids))
        return (sample.text.strip(), response_ids, metrics) if return_metrics else (sample.text.strip(), response_ids)

    def reset(self, data):
        return self._env_manager.reset(data)

    def step(self, actions, action_ids=None):
        return self._env_manager.step(actions, action_ids)


class InferenceEngine:
    def __init__(self, model: ModelManager, config: DemoConfig):
        self.model = model
        self.config = config
        self.steps: List[StepInfo] = []
        self.full_messages: List = []  # full message history

    def run(self, video, question, answer, q_type, options):
        self.steps = []
        self.full_messages = []

        try:
            try:
                r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "default=nw=1", video], capture_output=True, text=True, timeout=10)
                duration = float(r.stdout.strip().split('=')[1])
                r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "stream=r_frame_rate", "-select_streams", "v:0", "-of", "default=nw=1", video], capture_output=True, text=True, timeout=10)
                fps = eval(r.stdout.strip().split('=')[1])
                r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "stream=codec_type", "-of", "default=nw=1", video], capture_output=True, text=True, timeout=10)
                has_audio = "audio" in r.stdout
            except:
                duration, fps, has_audio = 120.0, 30.0, True

            data = {
                'video': [video], 'question_type': [q_type], 'question': [question],
                'answer': [answer], 'options': [options], 'fps': [fps],
                'duration_seconds': [duration], 'has_audio': [has_audio]
            }

            self.model.build_env(self.config)
            obs, _ = self.model.reset(data)

            # Save initial messages
            initial_msgs = obs['text'][0]
            self.full_messages = list(initial_msgs) if initial_msgs else []

            done, step_n, reward_sum = False, 0, 0.0

            while not done and step_n < self.config.max_steps:
                step_n += 1
                msgs = obs['text'][0]

                step_started = time.time()
                try:
                    resp, resp_ids, gen_metrics = self.model.generate(
                        msgs,
                        return_metrics=True,
                        sample_has_audio=has_audio,
                    )
                except Exception:
                    resp = '{"think":"Error","observation":"","action":{"type":"answer","content":"Error"}}'
                    resp_ids = []
                    gen_metrics = {
                        "prompt_build_seconds": 0.0,
                        "multimodal_prep_seconds": 0.0,
                        "token_count_seconds": 0.0,
                        "llm_generate_seconds": 0.0,
                        "generate_call_seconds": 0.0,
                        "prompt_text_tokens": 0,
                        "prompt_total_tokens": 0,
                        "prompt_nontext_tokens": 0,
                        "response_tokens": 0,
                        "full_sequence_tokens": 0,
                    }
                prompt_build_seconds = float(gen_metrics.get("prompt_build_seconds", 0.0))
                multimodal_prep_seconds = float(gen_metrics.get("multimodal_prep_seconds", 0.0))
                generation_seconds = float(gen_metrics.get("llm_generate_seconds", 0.0))
                prompt_text_tokens = int(gen_metrics.get("prompt_text_tokens", 0))
                prompt_total_tokens = int(gen_metrics.get("prompt_total_tokens", prompt_text_tokens))
                prompt_nontext_tokens = int(gen_metrics.get("prompt_nontext_tokens", max(0, prompt_total_tokens - prompt_text_tokens)))
                response_tokens = int(gen_metrics.get("response_tokens", len(resp_ids)))
                full_sequence_tokens = int(gen_metrics.get("full_sequence_tokens", prompt_total_tokens + response_tokens))

                postprocess_started = time.time()
                try:
                    r = json.loads(resp)
                    action_type = r.get('action', {}).get('type', 'unknown')
                    think = r.get('think', '')
                    observation = r.get('observation', '')
                    confidence_raw = r.get('confidence', None)
                    confidence = float(confidence_raw) if confidence_raw is not None else None
                except:
                    action_type, think, observation, confidence = 'unknown', '', '', None
                postprocess_seconds = time.time() - postprocess_started

                env_started = time.time()
                next_obs, rewards, dones, extra = self.model.step([resp], [resp_ids] if self.config.use_tito else None)
                env_step_seconds = time.time() - env_started

                # Update full message history
                if next_obs and 'text' in next_obs and next_obs['text']:
                    self.full_messages = list(next_obs['text'][0])

                clip, audio, frames = None, None, []
                if isinstance(extra, list) and extra and isinstance(extra[0], dict):
                    clip = extra[0].get('clip_path')
                    audio = extra[0].get('audio_path')
                    frames = extra[0].get('frame_paths', [])
                clip, audio, frames = filter_step_media_by_action_type(action_type, clip, audio, frames)

                rw = float(rewards[0]) if isinstance(rewards, (list, np.ndarray)) else float(rewards)
                done = bool(dones[0]) if isinstance(dones, (list, np.ndarray)) else bool(dones)
                reward_sum += rw
                step_wall_seconds = time.time() - step_started
                overhead_seconds = max(0.0, step_wall_seconds - prompt_build_seconds - generation_seconds - env_step_seconds - postprocess_seconds)

                si = StepInfo(
                    step_n, action_type, think, observation, confidence, resp, rw,
                    prompt_build_seconds, multimodal_prep_seconds, generation_seconds, env_step_seconds,
                    postprocess_seconds, overhead_seconds, step_wall_seconds,
                    prompt_text_tokens, prompt_total_tokens, prompt_nontext_tokens, response_tokens, full_sequence_tokens,
                    clip, audio, frames
                )
                self.steps.append(si)

                yield {'type': 'step', 'step': si, 'total_reward': reward_sum}
                obs = next_obs

            final = ""
            for s in reversed(self.steps):
                if s.action_type == 'answer':
                    try: final = json.loads(s.response).get('action', {}).get('content', '')
                    except: pass
                    break

            yield {'type': 'done', 'steps': self.steps, 'total_reward': reward_sum, 'final': final}

        except Exception as e:
            yield {'type': 'error', 'error': str(e)}



def generate_history_json(
    steps: List,
    full_messages: List = None,
    video: str = "",
    question: str = "",
    gt_answer: str = "",
    final_answer: str = "",
    question_type: str = "",
    options_text: str = "",
    source_label: str = "",
    runtime_config: Dict | None = None,
    env_snapshot: Dict | None = None,
    include_messages: bool = True,
) -> str:
    """Generate full history in JSON format"""
    video_identity = extract_video_identity(video, source_label)
    history = {
        "video": video,
        "question": question,
        "gt_answer": gt_answer,
        "final_answer": final_answer,
        "question_type": question_type,
        "options": options_text,
        "source_label": video_identity["source_label"],
        "dataset_name": video_identity["dataset_name"],
        "video_name": video_identity["video_name"],
        "total_steps": len(steps),
        "runtime_config": runtime_config or {},
        "env_snapshot": env_snapshot or {},
        "messages": [],
        "steps": []
    }
    
    # Add full message history
    if include_messages and full_messages:
        for msg in full_messages:
            msg_data = {"role": msg.get("role", "unknown")}
            content = msg.get("content", "")
            # Handle content that may be a list
            if isinstance(content, list):
                msg_data["content"] = []
                for item in content:
                    if isinstance(item, dict):
                        item_data = {"type": item.get("type", "text")}
                        if item.get("type") == "text":
                            item_data["text"] = item.get("text", "")
                        elif item.get("type") == "image":
                            item_data["image"] = item.get("image", "")
                        elif item.get("type") == "video":
                            item_data["video"] = item.get("video", "")
                        elif item.get("type") == "audio":
                            item_data["audio"] = item.get("audio", "")
                        msg_data["content"].append(item_data)
                    else:
                        msg_data["content"].append(str(item))
            else:
                msg_data["content"] = content
            history["messages"].append(msg_data)
    
    # Add step details
    for step in steps:
        step_data = {
            "step": step.step,
            "action_type": step.action_type,
            "think": step.think,
            "observation": step.observation,
            "confidence": step.confidence,
            "response": step.response,
            "reward": step.reward,
            "prompt_build_seconds": step.prompt_build_seconds,
            "multimodal_prep_seconds": step.multimodal_prep_seconds,
            "generation_seconds": step.generation_seconds,
            "env_step_seconds": step.env_step_seconds,
            "postprocess_seconds": step.postprocess_seconds,
            "overhead_seconds": step.overhead_seconds,
            "step_wall_seconds": step.step_wall_seconds,
            "prompt_text_tokens": step.prompt_text_tokens,
            "prompt_total_tokens": step.prompt_total_tokens,
            "prompt_nontext_tokens": step.prompt_nontext_tokens,
            "response_tokens": step.response_tokens,
            "full_sequence_tokens": step.full_sequence_tokens,
        }
        if step.clip_path:
            step_data["clip_path"] = step.clip_path
        if step.audio_path:
            step_data["audio_path"] = step.audio_path
        if step.frame_paths:
            step_data["frame_paths"] = step.frame_paths
        history["steps"].append(step_data)
    
    return json.dumps(history, indent=2, ensure_ascii=False)

def get_step_html(step: StepInfo) -> str:
    """Generate step card HTML - display all text completely"""
    colors = {'get_frames': '#2196F3', 'get_clip': '#4CAF50', 'get_audio': '#FF9800', 'answer': '#F44336'}
    icons = {'get_frames': '🖼️', 'get_clip': '🎥', 'get_audio': '🔊', 'answer': '✅'}
    c = colors.get(step.action_type, '#607D8B')
    i = icons.get(step.action_type, '📌')
    
    # Escape HTML special characters for complete display
    think_escaped = html.escape(step.think) if step.think else ""
    observation_escaped = html.escape(step.observation) if step.observation else ""
    
    # Convert newlines to <br> to preserve formatting
    think_formatted = think_escaped.replace('\n', '<br>')
    observation_formatted = observation_escaped.replace('\n', '<br>')
    
    # Parse action info
    action_info = ""
    try:
        resp_data = json.loads(step.response) if step.response else {}
        action = resp_data.get('action', {})
        action_type = action.get('type', '')
        
        # Format action parameters
        if action_type:
            action_params = []
            if action_type == 'get_frames':
                start = action.get('start', '')
                end = action.get('end', '')
                num = action.get('num', '')
                if start is not None:
                    action_params.append(f"start: {start}")
                if end is not None:
                    action_params.append(f"end: {end}")
                if num is not None:
                    action_params.append(f"num: {num}")
            elif action_type == 'get_clip':
                start = action.get('start', '')
                end = action.get('end', '')
                if start is not None:
                    action_params.append(f"start: {start}")
                if end is not None:
                    action_params.append(f"end: {end}")
            elif action_type == 'get_audio':
                start = action.get('start', '')
                end = action.get('end', '')
                if start is not None:
                    action_params.append(f"start: {start}")
                if end is not None:
                    action_params.append(f"end: {end}")
            elif action_type == 'answer':
                content = action.get('content', '')
                if content:
                    action_params.append(f"content: {content}")
            
            action_info = f"type: {action_type}"
            if action_params:
                action_info += ", " + ", ".join(action_params)
    except:
        action_info = step.action_type
    
    action_escaped = html.escape(str(action_info)) if action_info else ""
    action_formatted = action_escaped.replace('\n', '<br>')
    confidence_text = f"{step.confidence:.3f}" if isinstance(step.confidence, (int, float)) else "N/A"

    return f"""
    <div style="background: #ffffff; border: 1px solid #d7e3f0; border-radius: 16px; margin-bottom: 16px; overflow: hidden; box-shadow: 0 10px 24px rgba(28, 52, 84, 0.08);">
        <div style="background: linear-gradient(135deg, rgba(24, 66, 122, 0.96) 0%, rgba(37, 99, 171, 0.92) 100%); padding: 14px 18px; border-left: 5px solid {c}; display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 22px;">{i}</span>
                <div>
                    <div style="font-weight: 700; font-size: 16px; color: #ffffff;">Step {step.step}</div>
                    <div style="font-size: 12px; color: rgba(255, 255, 255, 0.78);">Agent decision trace</div>
                </div>
            </div>
            <span style="background: rgba(255,255,255,0.14); color: white; padding: 5px 12px; border-radius: 999px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;">{step.action_type}</span>
        </div>
        <div style="padding: 16px;">
            <div style="background: #f8fbff; border: 1px solid #dce9f7; border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;">
                <div style="color: #144b8b; font-weight: 700; font-size: 12px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.04em;">Observation</div>
                <div style="color: #22313f; font-size: 13px; line-height: 1.65; white-space: pre-wrap; word-wrap: break-word;">{observation_formatted}</div>
            </div>
            <div style="background: #f6faf6; border: 1px solid #dcebdc; border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;">
                <div style="color: #2f6b3b; font-weight: 700; font-size: 12px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.04em;">Thinking</div>
                <div style="color: #22313f; font-size: 13px; line-height: 1.65; white-space: pre-wrap; word-wrap: break-word;">{think_formatted}</div>
            </div>
            <div style="background: #fff8f1; border: 1px solid #f1dec6; border-radius: 12px; padding: 12px 14px; margin-bottom: 12px;">
                <div style="color: #9a5710; font-weight: 700; font-size: 12px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.04em;">Action</div>
                <div style="color: #22313f; font-size: 13px; line-height: 1.65; font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{action_formatted}</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; gap: 12px; padding-top: 10px; border-top: 1px solid #edf2f7;">
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <span style="background: #eef3fb; color: #18427a; padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 700;">Confidence {confidence_text}</span>
                    <span style="background: #eef8f1; color: #207a46; padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 700;">Reward {step.reward:.4f}</span>
                </div>
            </div>
        </div>
    </div>
    """


def render_frame_strip(frame_paths: List[str]) -> str:
    items = []
    for idx, frame_path in enumerate(frame_paths, start=1):
        quoted_path = quote(frame_path)
        image_url = f"/gradio_api/file={quoted_path}"
        time_label = extract_frame_time_label(frame_path)
        label = f"#{idx}"
        if time_label:
            label += f" · {time_label}"
        items.append(
            f"""
            <div class="frame-thumb">
                <a class="frame-thumb-button" href="{image_url}" target="_blank" rel="noopener noreferrer">
                    <img src="{image_url}" alt="frame-{idx}" />
                </a>
                <div class="frame-thumb-label">{label}</div>
            </div>
            """
        )
    return f"""
    <div class="frame-strip-wrap">
        <div class="frame-strip-hint">Frames</div>
        <div class="frame-strip">{''.join(items)}</div>
    </div>
    """


def status_updates(state: str, detail: str):
    return [gr.update(value=state), gr.update(value=detail)]


def probe_video_metadata(video_path: str) -> Dict[str, object]:
    info = {
        "path": video_path or "",
        "exists": False,
        "duration": None,
        "fps": None,
        "size_mb": None,
    }
    if not video_path:
        return info

    path = Path(video_path)
    if not path.exists():
        return info

    info["exists"] = True
    try:
        info["size_mb"] = path.stat().st_size / (1024 * 1024)
    except Exception:
        pass

    try:
        duration_cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=nw=1", str(path)
        ]
        duration_out = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=10)
        duration_text = duration_out.stdout.strip()
        if "=" in duration_text:
            info["duration"] = float(duration_text.split("=")[1])
    except Exception:
        pass

    try:
        fps_cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "stream=r_frame_rate",
            "-select_streams", "v:0", "-of", "default=nw=1", str(path)
        ]
        fps_out = subprocess.run(fps_cmd, capture_output=True, text=True, timeout=10)
        fps_text = fps_out.stdout.strip()
        if "=" in fps_text:
            info["fps"] = eval(fps_text.split("=")[1])
    except Exception:
        pass

    return info


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "Unknown"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def detect_dataset_name(video_path: str) -> str:
    if not video_path:
        return "Unknown"

    path_obj = Path(video_path)
    try:
        resolved_video = path_obj.resolve(strict=False)
    except Exception:
        resolved_video = path_obj

    for dataset_name, root in [
        ("Video-MME", VIDEOMME_ROOT),
        ("LVBench", LVBENCH_ROOT),
        ("VIDI", VIDI_ROOT),
    ]:
        if not root:
            continue
        try:
            resolved_root = Path(root).resolve(strict=False)
            resolved_video.relative_to(resolved_root)
            return dataset_name
        except Exception:
            continue

    return "Uploaded" if path_obj.exists() else "Custom"


def extract_video_identity(video_path: str, source_label: str = "") -> Dict[str, str]:
    video_name = Path(video_path).name if video_path else ""
    return {
        "dataset_name": detect_dataset_name(video_path),
        "video_name": video_name or (source_label or "Unknown"),
        "source_label": source_label or video_name or "Current Video",
    }


def extract_frame_time_label(frame_path: str) -> str:
    frame_name = Path(frame_path).name
    match = re.search(r"_frame_(\d+(?:\.\d+)?)\.", frame_name)
    return f"{float(match.group(1)):.2f}s" if match else ""


def filter_step_media_by_action_type(
    action_type: str,
    clip_path: Optional[str],
    audio_path: Optional[str],
    frame_paths: Optional[List[str]],
) -> Tuple[Optional[str], Optional[str], List[str]]:
    frames = list(frame_paths or [])
    if action_type == "get_clip":
        return clip_path, None, []
    if action_type == "get_audio":
        return None, audio_path, []
    if action_type == "get_frames":
        return None, None, frames
    return None, None, []


def render_video_card(video_path: str, source_label: str) -> str:
    if not video_path:
        return """
        <div class="video-card video-card-empty">
            <div class="video-card-title">No video selected</div>
            <div class="video-card-note">Choose a built-in example or upload a local video.</div>
        </div>
        """

    meta = probe_video_metadata(video_path)
    path_text = html.escape(video_path)
    duration_text = format_duration(meta.get("duration"))
    fps = meta.get("fps")
    fps_text = f"{fps:.2f}" if isinstance(fps, (int, float)) else "Unknown"
    size_mb = meta.get("size_mb")
    size_text = f"{size_mb:.1f} MB" if isinstance(size_mb, (int, float)) else "Unknown"
    preview_note = (
        "Preview is loaded on demand to avoid the browser eagerly pulling long videos."
        if meta.get("exists")
        else "Preview unavailable because the file was not found."
    )

    return f"""
    <div class="video-card">
        <div class="video-card-header">
            <div>
                <div class="video-card-eyebrow">Current Video</div>
                <div class="video-card-title">{html.escape(source_label)}</div>
            </div>
            <span class="video-card-badge">
                <span class="video-card-badge-label">Duration</span>
                <span class="video-card-badge-value">{duration_text}</span>
            </span>
        </div>
        <div class="video-card-grid">
            <div><span>FPS</span><strong>{fps_text}</strong></div>
            <div><span>Size</span><strong>{size_text}</strong></div>
        </div>
        <div class="video-card-path" title="{path_text}">{path_text}</div>
        <div class="video-card-note">{html.escape(preview_note)}</div>
    </div>
    """


def upload_original_video_public_url(video_path: str) -> Optional[str]:
    if not video_path:
        return None
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        return None

    bucket_name = os.environ.get("OMNIAGENT_EXPORT_VIDEO_BUCKET", os.environ.get("OSS_BUCKET", ""))
    object_prefix = os.environ.get("OMNIAGENT_EXPORT_VIDEO_PREFIX", "omniagent_demo_source_video/")
    expire_seconds = int(os.environ.get("OMNIAGENT_EXPORT_VIDEO_EXPIRE_SECONDS", str(365 * 24 * 3600)))

    if not object_prefix.endswith("/"):
        object_prefix += "/"

    date_dir = datetime.now().strftime("%Y%m%d")
    size_tag = path.stat().st_size
    key_name = f"{slugify_filename(path.stem, max_len=48)}_{size_tag}{path.suffix}"
    object_key = f"{object_prefix}{date_dir}/{key_name}"

    try:
        bucket = oss_reader.bucket[bucket_name]
        if not bucket.object_exists(object_key):
            bucket.put_object_from_file(object_key, str(path))
        return bucket.sign_url("GET", object_key, expire_seconds, slash_safe=True)
    except Exception as e:
        print(f"[Export] Failed to create public video URL: {e}")
        return None


def slugify_filename(text: str, max_len: int = 64) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if not text:
        text = "result"
    return text[:max_len].rstrip("-") or "result"


def format_export_action(step: Dict) -> str:
    response_text = step.get("response", "")
    try:
        resp_data = json.loads(response_text) if response_text else {}
        action = resp_data.get("action", {})
    except Exception:
        action = {}

    action_type = action.get("type") or step.get("action_type", "unknown")
    parts = [f"type: {action_type}"]
    for key in ("start", "end", "num", "content"):
        if key in action and action.get(key) is not None:
            parts.append(f"{key}: {action.get(key)}")
    return ", ".join(parts)


def copy_export_media(src: str, media_dir: Path, export_name: str) -> Optional[str]:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists() or not src_path.is_file():
        return None
    suffix = src_path.suffix or ""
    target_name = f"{export_name}{suffix}"
    target_path = media_dir / target_name
    shutil.copy2(src_path, target_path)
    return str(Path("media") / target_name)


def build_export_html(
    history: Dict,
    state_text: str,
    status_text: str,
    reward_text: str,
    answer_text: str,
    gt_answer_text: str,
    question_type_text: str,
    options_text: str,
    source_label: str,
    video_meta: Dict[str, object],
    public_video_url: Optional[str],
) -> str:
    question = html.escape(str(history.get("question", "")))
    source_video_raw = str(history.get("video_name", "")) or Path(str(history.get("video", ""))).name
    source_video = html.escape(source_video_raw)
    dataset_name = html.escape(str(history.get("dataset_name", "")) or detect_dataset_name(str(history.get("video", ""))))
    question_type_raw = question_type_text or str(history.get("question_type", "")) or str(history.get("runtime_config", {}).get("question_type", "")) or "Unknown"
    gt_answer_raw = gt_answer_text or str(history.get("gt_answer", ""))
    final_answer_raw = answer_text or str(history.get("final_answer", ""))
    question_type = html.escape(question_type_raw)
    gt_answer = html.escape(gt_answer_raw)
    final_answer = html.escape(final_answer_raw)
    options_html = html.escape(options_text or str(history.get("options", ""))).replace("\n", "<br>")
    runtime_json = html.escape(json.dumps(history.get("runtime_config", {}), ensure_ascii=False, indent=2))
    duration_text = format_duration(video_meta.get("duration"))
    fps = video_meta.get("fps")
    fps_text = f"{fps:.2f}" if isinstance(fps, (int, float)) else "Unknown"
    size_mb = video_meta.get("size_mb")
    size_text = f"{size_mb:.1f} MB" if isinstance(size_mb, (int, float)) else "Unknown"
    source_title = html.escape(str(history.get("source_label", "")) or source_label or source_video_raw or "Current Video")
    video_link_html = ""
    if public_video_url:
        safe_url = html.escape(public_video_url)
        video_link_html = f"""
        <div class="video-link-row">
          <a class="video-link" href="{safe_url}" target="_blank" rel="noopener noreferrer">Open Original Video</a>
        </div>
        <div class="media-block" style="margin-top: 10px;">
          <div class="media-label">Original Video via Public URL</div>
          <video controls preload="metadata" src="{safe_url}"></video>
        </div>
        """

    step_blocks = []
    for step in history.get("steps", []):
        observation = html.escape(step.get("observation", "")).replace("\n", "<br>")
        think = html.escape(step.get("think", "")).replace("\n", "<br>")
        action = html.escape(format_export_action(step)).replace("\n", "<br>")
        confidence = step.get("confidence")
        confidence_text = f"{float(confidence):.3f}" if isinstance(confidence, (int, float)) else "N/A"
        media_html = ""

        action_type = str(step.get("action_type", "unknown"))
        clip_rel = step.get("export_clip_path") if action_type == "get_clip" else None
        audio_rel = step.get("export_audio_path") if action_type == "get_audio" else None
        frame_entries = step.get("export_frames", []) if action_type == "get_frames" else []
        frame_rels = step.get("export_frame_paths", []) if action_type == "get_frames" else []
        if clip_rel:
            media_html += f"""
            <div class="media-block">
                <div class="media-label">Retrieved Clip</div>
                <video controls preload="metadata" src="{html.escape(clip_rel)}"></video>
            </div>
            """
        if audio_rel:
            media_html += f"""
            <div class="media-block">
                <div class="media-label">Retrieved Audio</div>
                <audio controls preload="metadata" src="{html.escape(audio_rel)}"></audio>
            </div>
            """
        if frame_entries or frame_rels:
            frame_items = []
            if frame_entries:
                normalized_frames = []
                for frame_entry in frame_entries:
                    if isinstance(frame_entry, dict):
                        normalized_frames.append(frame_entry)
                    else:
                        normalized_frames.append({"path": str(frame_entry), "time_label": ""})
            else:
                normalized_frames = [{"path": frame_rel, "time_label": ""} for frame_rel in frame_rels]
            for idx, frame_entry in enumerate(normalized_frames, start=1):
                frame_rel = str(frame_entry.get("path", ""))
                time_label = str(frame_entry.get("time_label", "")).strip()
                label = f"Frame {idx}"
                if time_label:
                    label += f" · {time_label}"
                frame_items.append(
                    f"""
                    <a class="frame-item" href="{html.escape(frame_rel)}" target="_blank" rel="noopener noreferrer">
                        <img src="{html.escape(frame_rel)}" alt="frame-{idx}" />
                        <span>{html.escape(label)}</span>
                    </a>
                    """
                )
            media_html += f"""
            <div class="media-block">
                <div class="media-label">Retrieved Frames</div>
                <div class="frame-grid">{''.join(frame_items)}</div>
            </div>
            """

        step_blocks.append(
            f"""
            <section class="step-card">
                <div class="step-head">
                    <div>
                        <div class="step-title">Step {step.get("step", "?")}</div>
                        <div class="step-meta">{html.escape(step.get("action_type", "unknown"))}</div>
                    </div>
                    <div class="pill-row">
                        <span class="pill">Confidence {confidence_text}</span>
                        <span class="pill">Reward {float(step.get("reward", 0.0)):.4f}</span>
                    </div>
                </div>
                <div class="info-block">
                    <div class="info-label">Observation</div>
                    <div>{observation}</div>
                </div>
                <div class="info-block">
                    <div class="info-label">Thinking</div>
                    <div>{think}</div>
                </div>
                <div class="info-block">
                    <div class="info-label">Action</div>
                    <div class="mono">{action}</div>
                </div>
                {media_html}
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OmniAgent Result Export</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f3f7fb 0%, #eef3f8 100%);
      color: #1d2d3d;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero, .step-card {{
      background: #fff;
      border: 1px solid #dce6f0;
      border-radius: 18px;
      box-shadow: 0 12px 30px rgba(21, 50, 80, 0.08);
    }}
    .hero {{
      padding: 22px;
      margin-bottom: 18px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      color: #143d6b;
    }}
    .hero-note {{
      color: #5b7288;
      margin-bottom: 16px;
    }}
    .video-card {{
      background: linear-gradient(180deg, #f8fbff 0%, #f2f7fc 100%);
      border: 1px solid #d8e4ef;
      border-radius: 16px;
      padding: 14px;
      margin-bottom: 14px;
    }}
    .video-card-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .video-card-eyebrow {{
      color: #65809b;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }}
    .video-card-title {{
      color: #143d6b;
      font-size: 16px;
      font-weight: 700;
      line-height: 1.3;
    }}
    .video-card-badge {{
      background: linear-gradient(135deg, #163f71 0%, #2f7abf 100%);
      color: #ffffff;
      border-radius: 16px;
      padding: 10px 14px;
      min-width: 118px;
      text-align: center;
      box-shadow: 0 10px 20px rgba(21, 61, 107, 0.18);
    }}
    .video-card-badge-label {{
      display: block;
      color: #FDFDFD;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      opacity: 0.82;
      margin-bottom: 4px;
    }}
    .video-card-badge-value {{
      display: block;
      color: #FDFDFD;
      font-size: 20px;
      line-height: 1.1;
      font-weight: 800;
      white-space: nowrap;
    }}
    .video-card-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .video-card-grid div {{
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid #dce6ef;
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .video-card-grid span {{
      display: block;
      color: #6a8197;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 4px;
    }}
    .video-card-grid strong {{
      color: #1c2f44;
      font-size: 14px;
    }}
    .video-card-path {{
      background: #ffffff;
      border: 1px solid #dce6ef;
      border-radius: 12px;
      color: #395066;
      font-family: monospace;
      font-size: 12px;
      line-height: 1.5;
      padding: 10px 12px;
      margin-bottom: 10px;
      white-space: nowrap;
      overflow-x: auto;
    }}
    .video-link-row {{
      margin-bottom: 8px;
    }}
    .video-link {{
      display: inline-block;
      background: #143d6b;
      color: #fff;
      text-decoration: none;
      padding: 8px 12px;
      border-radius: 10px;
      font-weight: 700;
      font-size: 13px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .summary-card {{
      background: #f8fbff;
      border: 1px solid #dce7f2;
      border-radius: 14px;
      padding: 12px 14px;
    }}
    .summary-card span {{
      display: block;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #637f99;
      margin-bottom: 4px;
    }}
    .summary-card strong {{
      display: block;
      color: #163b64;
      font-size: 15px;
      line-height: 1.5;
      word-break: break-word;
    }}
    .runtime-box {{
      background: #0f2741;
      color: #e7eef6;
      border-radius: 14px;
      padding: 14px;
      overflow: auto;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }}
    .section-title {{
      margin: 22px 0 12px;
      font-size: 18px;
      font-weight: 700;
      color: #163b64;
    }}
    .step-card {{
      padding: 18px;
      margin-bottom: 16px;
    }}
    .step-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .step-title {{
      font-size: 18px;
      font-weight: 700;
      color: #143d6b;
    }}
    .step-meta {{
      color: #67819a;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-top: 4px;
    }}
    .pill-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .pill {{
      background: #eef4fb;
      color: #18427a;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .info-block {{
      background: #f8fbff;
      border: 1px solid #dbe7f2;
      border-radius: 12px;
      padding: 12px 14px;
      margin-bottom: 10px;
      line-height: 1.6;
    }}
    .info-label {{
      color: #58728b;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      word-break: break-word;
    }}
    .media-block {{
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid #e8eef5;
    }}
    .media-label {{
      color: #58728b;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    video, audio {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #0f1720;
    }}
    .frame-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 10px;
    }}
    .frame-item {{
      display: block;
      text-decoration: none;
      color: #395066;
      font-size: 12px;
      font-weight: 700;
    }}
    .frame-item img {{
      width: 100%;
      height: 100px;
      object-fit: cover;
      border-radius: 10px;
      border: 1px solid #dce6ef;
      display: block;
      margin-bottom: 6px;
      background: #eef4f9;
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>OmniAgent Offline Result</h1>
      <div class="hero-note">This bundle contains the execution trace and retrieved media only. The original source video is intentionally omitted.</div>
      <div class="video-card">
        <div class="video-card-header">
          <div>
            <div class="video-card-eyebrow">Current Video</div>
            <div class="video-card-title">{source_title}</div>
          </div>
          <span class="video-card-badge">
            <span class="video-card-badge-label">Duration</span>
            <span class="video-card-badge-value">{duration_text}</span>
          </span>
        </div>
        <div class="video-card-grid">
          <div><span>FPS</span><strong>{fps_text}</strong></div>
          <div><span>Size</span><strong>{size_text}</strong></div>
        </div>
        <div class="video-card-path">{html.escape(str(history.get("video", "")))}</div>
        {video_link_html}
      </div>
      <div class="summary-grid">
        <div class="summary-card"><span>Dataset</span><strong>{dataset_name or "Unknown"}</strong></div>
        <div class="summary-card"><span>Video Name</span><strong>{source_video or "Unknown"}</strong></div>
        <div class="summary-card"><span>Question</span><strong>{question}</strong></div>
        <div class="summary-card"><span>Question Type</span><strong>{question_type}</strong></div>
        <div class="summary-card"><span>GT Answer</span><strong>{gt_answer or "N/A"}</strong></div>
        <div class="summary-card"><span>Final Answer</span><strong>{final_answer or "N/A"}</strong></div>
        <div class="summary-card"><span>State</span><strong>{html.escape(state_text or "")}</strong></div>
        <div class="summary-card"><span>Status</span><strong>{html.escape(status_text or "")}</strong></div>
        <div class="summary-card"><span>Reward</span><strong>{html.escape(reward_text or "")}</strong></div>
      </div>
      {f'<div class="info-block"><div class="info-label">Options</div><div>{options_html}</div></div>' if options_html and question_type_raw.upper() == "MCQ" else ''}
      <div class="info-label">Runtime Config</div>
      <div class="runtime-box">{runtime_json}</div>
    </section>
    <div class="section-title">Execution Trace</div>
    {''.join(step_blocks)}
  </main>
</body>
</html>
"""


def export_result_bundle(
    history_json_text: str,
    state_text: str,
    status_text: str,
    reward_text: str,
    answer_text: str,
    gt_answer_text: str,
    question_type_text: str,
    options_text: str,
    source_label: str,
):
    if not history_json_text or history_json_text.strip() in ("", "{}"):
        return "No result available to export yet.", None

    try:
        history = json.loads(history_json_text)
    except Exception as e:
        return f"Failed to parse history JSON: {e}", None

    export_root = Path(tempfile.gettempdir()) / "omniagent_demo_exports"
    export_root.mkdir(parents=True, exist_ok=True)

    video_identity = extract_video_identity(str(history.get("video", "")), str(history.get("source_label", "")) or source_label)
    if not history.get("source_label"):
        history["source_label"] = video_identity["source_label"]
    if not history.get("dataset_name"):
        history["dataset_name"] = video_identity["dataset_name"]
    if not history.get("video_name"):
        history["video_name"] = video_identity["video_name"]
    if not history.get("gt_answer"):
        history["gt_answer"] = gt_answer_text
    if not history.get("final_answer"):
        history["final_answer"] = answer_text
    if not history.get("question_type"):
        history["question_type"] = question_type_text or str(history.get("runtime_config", {}).get("question_type", ""))
    if not history.get("options"):
        history["options"] = options_text

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = slugify_filename(Path(str(history.get("video", ""))).stem, max_len=24)
    question_type_value = (
        str(history.get("question_type", ""))
        or question_type_text
        or str(history.get("runtime_config", {}).get("question_type", ""))
    )
    question_type = slugify_filename(question_type_value, max_len=12)
    if not question_type or question_type == "result":
        question_text = str(history.get("question", ""))
        question_type = "tr" if "time ranges" in question_text.lower() else "ff"
        if (options_text or str(history.get("options", ""))) or "options:" in question_text.lower():
            question_type = "mcq"
    question_slug = slugify_filename(history.get("question", "omniagent-result"), max_len=28)
    bundle_stem = f"omniagent_result_{timestamp}_{video_name}_{question_type}_{question_slug}"
    work_dir = export_root / bundle_stem
    media_dir = work_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    try:
        video_path = str(history.get("video", ""))
        video_meta = probe_video_metadata(video_path)
        public_video_url = upload_original_video_public_url(video_path)

        for step in history.get("steps", []):
            step_no = int(step.get("step", 0))
            action_type = str(step.get("action_type", ""))
            clip_path, audio_path, frame_paths = filter_step_media_by_action_type(
                action_type,
                step.get("clip_path"),
                step.get("audio_path"),
                step.get("frame_paths", []),
            )
            step["clip_path"] = clip_path
            step["audio_path"] = audio_path
            step["frame_paths"] = frame_paths
            step.pop("export_clip_path", None)
            step.pop("export_audio_path", None)
            step.pop("export_frame_paths", None)
            step.pop("export_frames", None)

            if clip_path:
                step["export_clip_path"] = copy_export_media(
                    clip_path,
                    media_dir,
                    f"step_{step_no:02d}_clip",
                )
            elif audio_path:
                step["export_audio_path"] = copy_export_media(
                    audio_path,
                    media_dir,
                    f"step_{step_no:02d}_audio",
                )
            elif frame_paths:
                export_frames = []
                export_frame_entries = []
                for idx, frame_path in enumerate(frame_paths, start=1):
                    rel = copy_export_media(frame_path, media_dir, f"step_{step_no:02d}_frame_{idx:02d}")
                    if rel:
                        export_frames.append(rel)
                        export_frame_entries.append({
                            "path": rel,
                            "time_label": extract_frame_time_label(frame_path),
                        })
                if export_frames:
                    step["export_frame_paths"] = export_frames
                    step["export_frames"] = export_frame_entries

        history_path = work_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        html_text = build_export_html(
            history,
            state_text,
            status_text,
            reward_text,
            answer_text or str(history.get("final_answer", "")),
            gt_answer_text or str(history.get("gt_answer", "")),
            question_type_text or str(history.get("question_type", "")),
            options_text or str(history.get("options", "")),
            source_label,
            video_meta,
            public_video_url,
        )
        (work_dir / "index.html").write_text(html_text, encoding="utf-8")

        zip_path = export_root / f"{bundle_stem}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in work_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, arcname=str(file_path.relative_to(work_dir)))
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        return f"Export failed: {e}", None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    media_count = sum(
        len(step.get("export_frame_paths", []))
        + (1 if step.get("export_clip_path") else 0)
        + (1 if step.get("export_audio_path") else 0)
        for step in history.get("steps", [])
    )
    return f"Exported offline bundle with {len(history.get('steps', []))} steps and {media_count} retrieved media files.", str(zip_path)


def create_demo(config: DemoConfig):
    model = ModelManager()
    current_engine = None
    is_running = False
    next_run_id = 0
    active_run_id = 0

    css = """
    body { background: linear-gradient(180deg, #f3f7fb 0%, #eef3f8 100%); }
    #steps-container { max-height: 65vh; overflow-y: auto; padding-right: 8px; }
    #steps-container::-webkit-scrollbar { width: 6px; }
    #steps-container::-webkit-scrollbar-thumb { background: #9fb4c8; border-radius: 3px; }
    .app-shell {
        background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(246,250,253,0.98) 100%);
        border: 1px solid #dce6f0;
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 18px 40px rgba(20, 47, 76, 0.08);
    }
    .main-title { 
        text-align: center;
        background: linear-gradient(135deg, #143d6b 0%, #2f7abf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 30px !important;
        font-weight: 700 !important;
        margin-bottom: 6px !important;
    }
    .subtitle { 
        text-align: center; 
        color: #58708a;
        font-size: 14px;
        margin-bottom: 18px;
    }
    .section-header {
        border-bottom: 2px solid #2f7abf;
        padding-bottom: 8px;
        margin-bottom: 12px;
        color: #163b64;
        font-weight: 700;
    }
    .panel-card {
        background: #ffffff;
        border: 1px solid #dbe6ef;
        border-radius: 16px;
        padding: 14px;
        margin-bottom: 14px;
        box-shadow: 0 8px 18px rgba(21, 50, 80, 0.05);
    }
    .control-note {
        background: #f5f9fd;
        border: 1px solid #dce8f4;
        color: #4a6178;
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 12px;
        font-size: 13px;
    }
    .status-box textarea, .status-box input {
        font-weight: 600 !important;
        color: #143d6b !important;
    }
    .state-box textarea, .state-box input {
        font-weight: 800 !important;
        color: #18427a !important;
        letter-spacing: 0.04em;
    }
    .metric-box textarea, .metric-box input {
        font-weight: 700 !important;
    }
    .button-row { gap: 10px; }
    .steps-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        color: #163b64;
        font-weight: 700;
    }
    .frame-strip-wrap {
        background: #ffffff;
        border: 1px solid #d7e3f0;
        border-radius: 14px;
        padding: 12px;
        overflow: hidden;
    }
    .frame-strip-hint {
        color: #48627d;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .frame-strip {
        display: flex;
        gap: 10px;
        overflow-x: auto;
        padding-bottom: 4px;
        scrollbar-width: thin;
    }
    .frame-strip::-webkit-scrollbar {
        height: 8px;
    }
    .frame-strip::-webkit-scrollbar-thumb {
        background: #a6b8ca;
        border-radius: 999px;
    }
    .frame-thumb {
        flex: 0 0 auto;
        width: 132px;
    }
    .frame-thumb-button {
        padding: 0;
        border: 0;
        background: transparent;
        cursor: zoom-in;
        width: 132px;
    }
    .frame-thumb img {
        width: 132px;
        height: 88px;
        object-fit: cover;
        border-radius: 10px;
        border: 1px solid #dce6ef;
        display: block;
        background: #eff4f9;
        transition: transform 0.16s ease, box-shadow 0.16s ease;
    }
    .frame-thumb-button:hover img {
        transform: translateY(-1px);
        box-shadow: 0 10px 18px rgba(26, 57, 92, 0.14);
    }
    .frame-thumb-label {
        color: #5c738a;
        font-size: 11px;
        font-weight: 700;
        text-align: center;
        margin-top: 6px;
    }
    .video-card {
        background: linear-gradient(180deg, #f8fbff 0%, #f2f7fc 100%);
        border: 1px solid #d8e4ef;
        border-radius: 16px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
    }
    .video-card-empty {
        background: linear-gradient(180deg, #fafcfe 0%, #f4f7fa 100%);
    }
    .video-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 12px;
    }
    .video-card-eyebrow {
        color: #65809b;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .video-card-title {
        color: #143d6b;
        font-size: 16px;
        font-weight: 700;
        line-height: 1.3;
    }
    .video-card-badge {
        background: linear-gradient(135deg, #163f71 0%, #2f7abf 100%);
        color: #ffffff;
        border-radius: 16px;
        padding: 10px 14px;
        min-width: 118px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(21, 61, 107, 0.18);
    }
    .video-card-badge-label {
        display: block;
        color: #FDFDFD;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.82;
        margin-bottom: 4px;
    }
    .video-card-badge-value {
        display: block;
        color: #FDFDFD;
        font-size: 20px;
        line-height: 1.1;
        font-weight: 800;
        white-space: nowrap;
    }
    .video-card-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 12px;
    }
    .video-card-grid div {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid #dce6ef;
        border-radius: 12px;
        padding: 10px 12px;
    }
    .video-card-grid span {
        display: block;
        color: #6a8197;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .video-card-grid strong {
        color: #1c2f44;
        font-size: 14px;
    }
    .video-card-path {
        background: #ffffff;
        border: 1px solid #dce6ef;
        border-radius: 12px;
        color: #395066;
        font-family: monospace;
        font-size: 12px;
        line-height: 1.5;
        padding: 10px 12px;
        margin-bottom: 10px;
        white-space: nowrap;
        overflow-x: auto;
    }
    .video-card-note {
        color: #5e768e;
        font-size: 12px;
        line-height: 1.5;
    }
    .video-upload-panel {
    }
    .video-upload-panel .wrap {
    }
    .video-upload-panel .video-container,
    .video-upload-panel .empty-container,
    .video-upload-panel [data-testid="video"] {
    }
    .video-upload-panel .controls,
    .video-upload-panel .footer,
    .video-upload-panel .source-selection {
    }
    .video-upload-panel button[aria-label="Upload"],
    .video-upload-panel button[aria-label="Webcam"] {
    }
    """

    with gr.Blocks(title="OmniAgent Pro", css=css, theme=gr.themes.Soft(primary_hue="indigo")) as demo:

        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(scale=1, min_width=380):
                gr.HTML("<div class='main-title'>🎬 OmniAgent Pro</div>")
                gr.HTML("<div class='subtitle'>Video Understanding with Multi-Modal Agent</div>")
                _model_name = Path(config.model_path).name or config.model_path
                gr.HTML(
                    f"<div style='text-align:center; margin:-6px 0 8px; font-size:13px; color:#68829c;'>"
                    f"Model: <b>{_model_name}</b> &nbsp;|&nbsp; "
                    f"Built-in examples use short clips due to repo size limits. Upload your own videos of any length to explore!"
                    f"</div>"
                )
                with gr.Group(elem_classes=["panel-card"]):
                    example_dd = gr.Dropdown(choices=[(e["name"], i) for i, e in enumerate(BUILTIN_EXAMPLES)], label="📚 Select Example", value=None)
                    selected_video_path = gr.State("")
                    selected_video_label = gr.State("No video selected")
                    video_card = gr.HTML(render_video_card("", "No video selected"))
                    video_upload = gr.Video(
                        label="📤 Upload Video / Webcam",
                        sources=["upload", "webcam"],
                        elem_classes=["video-upload-panel"],
                    )
                    with gr.Row():
                        preview_btn = gr.Button("▶ Load Preview", variant="secondary")
                        clear_preview_btn = gr.Button("Hide Preview", variant="secondary")
                    video_preview = gr.Video(label="🎥 Video Preview", height=180, interactive=False, visible=False)
                    question_in = gr.Textbox(label="❓ Question", lines=2, placeholder="Enter your question about the video...")
                    with gr.Row():
                        answer_in = gr.Textbox(label="✅ Answer (GT)", placeholder="Ground truth")
                        q_type = gr.Dropdown(
                        choices=[
                            ("MCQ - Multiple Choice", "MCQ"),
                            ("TR - Temporal Grounding", "TR"),
                            ("FF - Free Form", "FF"),
                            ("NUM - Numerical", "NUM"),
                            ("SIZE - Size Estimation", "SIZE")
                        ],
                        value="MCQ", label="📋 Question Type"
                    )
                    options_in = gr.Textbox(label="📝 Options (MCQ)", lines=2, placeholder="A. ...\nB. ...\nC. ...")

                with gr.Accordion("⚙️ Runtime Configuration", open=False, elem_classes=["panel-card"]):
                    gr.HTML("<div class='control-note'>Applies on the next inference run. Sampling parameters update without reloading the model.</div>")
                    gr.HTML("<div class='section-header'>Environment</div>")
                    max_steps = gr.Slider(1, 50, value=config.max_steps, step=1, label="Max Steps")
                    max_frames = gr.Slider(10, 100, value=config.max_frames_len, step=5, label="Max Frames")
                    max_audio = gr.Slider(60, 600, value=int(config.max_audio_len), step=30, label="Max Audio (s)")
                    max_clip = gr.Slider(10, 120, value=int(config.max_clip_len), step=5, label="Max Clip (s)")
                    mode_input = gr.Textbox(value=config.mode, label="Mode")
                    dynamic_step_info = gr.Checkbox(
                        value=config.use_dynamic_step,
                        label="Dynamic Step Enabled",
                        interactive=True,
                    )
                    use_tito_info = gr.Checkbox(
                        value=config.use_tito,
                        label="TITO Enabled",
                        interactive=True,
                    )
                    gr.HTML(
                        "<div class='control-note'>"
                        "When enabled, the actual step limit is computed as "
                        "min(user_max_steps, MIN_MAX_STEPS + floor(video_duration / max_clip_len)). "
                        "MIN_MAX_STEPS defaults to 5 unless overridden by the runtime environment. "
                        "If disabled, effective_max_steps = user_max_steps. "
                        "The environment terminates when step_count >= effective_max_steps, and if the final allowed step "
                        "is not an `answer` action, the run fails with a step-limit error. "
                        "This is a runtime option and applies on the next inference run."
                        "</div>"
                    )
                    gr.HTML(
                        "<div class='control-note'>"
                        "TITO (Token-In-Token-Out) passes raw assistant output token IDs directly into `video_env` to prevent retokenization drift and preserve exact token-level history. "
                        "This is a runtime option and applies on the next inference run."
                        "</div>"
                    )

                    gr.HTML("<div class='section-header'>Sampling</div>")
                    with gr.Row():
                        temp_slider = gr.Slider(0.1, 2.0, value=config.temperature, step=0.1, label="Temperature")
                        top_p_slider = gr.Slider(0.1, 1.0, value=config.top_p, step=0.05, label="Top-p")
                        top_k_slider = gr.Slider(1, 100, value=config.top_k, step=1, label="Top-k")

                with gr.Accordion("🧠 Model Initialization", open=False, elem_classes=["panel-card"]):
                    gr.HTML("<div class='control-note'>These values are fixed when the model is loaded. If the model is unloaded, the next inference run will initialize it again automatically.</div>")
                    model_path_info = gr.Textbox(value=config.model_path, label="Model Path", interactive=False)
                    with gr.Row():
                        tensor_parallel_info = gr.Number(
                            value=config.tensor_parallel_size, label="Tensor Parallel Size", interactive=False
                        )
                        gpu_memory_info = gr.Number(
                            value=config.gpu_memory_utilization, label="GPU Memory Utilization", interactive=False
                        )
                    with gr.Row():
                        max_model_len_info = gr.Number(
                            value=config.max_model_len, label="Max Model Len", interactive=False
                        )
                        max_batched_tokens_info = gr.Number(
                            value=config.max_num_batched_tokens, label="Max Batched Tokens", interactive=False
                        )

                with gr.Row(elem_classes=["button-row"]):
                    start_btn = gr.Button("🚀 Start Inference", variant="primary", size="lg")
                    force_stop_btn = gr.Button("💥 Force Stop", variant="stop")

                with gr.Group(elem_classes=["panel-card"]):
                    state_out = gr.Textbox(label="🧭 State", value=STATUS_READY, interactive=False, elem_classes=["state-box"])
                    status = gr.Textbox(label="📊 Status", value="Model will auto-load on start", interactive=False, elem_classes=["status-box"])
                    with gr.Row():
                        reward_out = gr.Textbox("0.0000", label="💰 Reward", interactive=False, elem_classes=["metric-box"])
                        answer_out = gr.Textbox(label="🎯 Answer", interactive=False)

            # Right side result area
            with gr.Column(scale=2, min_width=750):
                gr.HTML("<div class='steps-title'><span>📊 Inference Steps</span><span style='font-size:12px; color:#68829c; font-weight:600;'>Scrollable execution trace</span></div>")
                
                # Scrollable container
                with gr.Column(elem_id="steps-container"):
                    # Pre-create step components
                    step_containers = []
                    step_htmls = []
                    step_videos = []
                    step_audios = []
                    step_frame_strips = []
                    
                    for i in range(MAX_STEPS):
                        with gr.Column(visible=False) as container:
                            step_htmls.append(gr.HTML(value="", visible=True))
                            with gr.Row():
                                step_videos.append(gr.Video(height=150, show_label=False, visible=False))
                                step_audios.append(gr.Audio(show_label=False, visible=False))
                                step_frame_strips.append(gr.HTML(value="", visible=False))
                        step_containers.append(container)

                # JSON History display area
                with gr.Accordion("📜 Full History (JSON)", open=False):
                    history_json = gr.Code(
                        label="",
                        value="{}",
                        language="json",
                        lines=15,
                        interactive=False
                    )
                with gr.Accordion("📦 Offline Export", open=False):
                    gr.HTML("<div class='control-note'>Export the current result page, JSON trace, and only the media retrieved by the agent. The original source video is not included.</div>")
                    export_btn = gr.Button("Export Result ZIP", variant="secondary")
                    export_status = gr.Textbox(label="Export Status", interactive=False)
                    export_file = gr.File(label="Download ZIP", interactive=False)

        # Event handling
        def clear_step_outputs():
            steps_update = []
            for i in range(MAX_STEPS):
                steps_update.extend([
                    gr.update(visible=False),
                    gr.update(value="", visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                ])
            return steps_update

        def cleanup_ray_runtime():
            try:
                import ray
                if ray.is_initialized():
                    ray.shutdown()
            except Exception as e:
                print(f"[Ray] Shutdown warning: {e}")

            for ray_dir in ["/tmp/ray", "/tmp/ray_spill"]:
                if os.path.exists(ray_dir):
                    try:
                        shutil.rmtree(ray_dir, ignore_errors=True)
                    except Exception as e:
                        print(f"[Ray] Cleanup warning for {ray_dir}: {e}")

        def is_run_cancelled(run_id: int) -> bool:
            return active_run_id != run_id

        def on_video_change(video):
            if not video:
                return "", "No video selected", render_video_card("", "No video selected")
            return video, "Uploaded Video", render_video_card(video, "Uploaded Video")

        def on_example(idx):
            if idx is None or idx >= len(BUILTIN_EXAMPLES):
                return "", "No video selected", render_video_card("", "No video selected"), gr.update(value=None), gr.update(value=None, visible=False), "", "", "MCQ", ""
            e = BUILTIN_EXAMPLES[idx]
            return (
                e["video"],
                e["name"],
                render_video_card(e["video"], e["name"]),
                gr.update(value=None),
                gr.update(value=None, visible=False),
                e["question"],
                e["answer"],
                e["type"],
                e.get("options", ""),
            )

        def load_preview(uploaded_video, selected_path):
            target = uploaded_video or selected_path
            if not target:
                return gr.update(value=None, visible=False)
            return gr.update(value=target, visible=True)

        def hide_preview():
            return gr.update(visible=False)

        def on_start(uploaded_video, selected_path, source_label, question, answer, qt, opts, ms, mf, ma, mc, mode, use_dynamic_step, use_tito, temp, top_p, top_k):
            nonlocal current_engine, is_running, next_run_id, active_run_id

            if is_running:
                yield status_updates(STATUS_BLOCKED, "Inference already running. Stop it before starting a new one.") + [gr.update(), gr.update()] + \
                      [gr.update()] * (MAX_STEPS * 5) + [gr.update(), gr.update(value=""), gr.update(value=None)]
                return

            next_run_id += 1
            run_id = next_run_id
            active_run_id = run_id
            is_running = True

            # Update configuration
            config.max_steps = ms
            config.max_frames_len = mf
            config.max_audio_len = float(ma)
            config.max_clip_len = float(mc)
            config.mode = mode
            config.use_dynamic_step = bool(use_dynamic_step)
            config.use_tito = bool(use_tito)
            config.temperature = temp
            config.top_p = top_p
            config.top_k = top_k
            os.environ["USE_DYNAMIC_STEP"] = "True" if config.use_dynamic_step else "False"
            os.environ["USE_TITO"] = "True" if config.use_tito else "False"

            # Sampling params are runtime config, refresh before each inference.
            if model.is_loaded:
                model.update_sampling_params(config)
            
            # Auto-load model if not loaded
            try:
                if not model.is_loaded:
                    if is_run_cancelled(run_id):
                        return
                    yield status_updates(STATUS_LOADING_MODEL, "Loading model (first time may take 1-2 minutes)...") + [gr.update(value="0"), gr.update(value="")] + \
                          [gr.update(visible=False)] * (MAX_STEPS * 5) + [gr.update(value="{}"), gr.update(value=""), gr.update(value=None)]
                    success, msg = model.load(config)
                    if not success:
                        yield status_updates(STATUS_ERROR, msg) + [gr.update(value="0"), gr.update(value="")] + \
                              [gr.update(visible=False)] * (MAX_STEPS * 5) + [gr.update(value="{}"), gr.update(value=""), gr.update(value=None)]
                        return
                
                # Restart Ray before each inference to ensure clean state
                if is_run_cancelled(run_id):
                    return
                yield status_updates(STATUS_INITIALIZING_RUNTIME, "Initializing Ray runtime...") + [gr.update(value="0"), gr.update(value="")] + \
                          [gr.update(visible=False)] * (MAX_STEPS * 5) + [gr.update(value="{}"), gr.update(value=""), gr.update(value=None)]
                
                try:
                    import ray
                    cleanup_ray_runtime()
                    ray.init(ignore_reinit_error=True)
                    print("[Ray] Restarted for new inference")
                except Exception as e:
                    print(f"[Ray] Restart warning: {e}")

                resolved_video = uploaded_video or selected_path

                if not resolved_video or not question:
                    base = status_updates(STATUS_ERROR, "Need video and question") + [gr.update(value="0"), gr.update(value="")]
                    yield base + clear_step_outputs() + [gr.update(value="{}"), gr.update(value=""), gr.update(value=None)]
                    return

                opts_list = [o.strip() for o in opts.split('\n') if o.strip()] if qt == "MCQ" else None
                runtime_config_payload = {
                    "question_type": qt,
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
                }
                env_snapshot_payload = {key: os.environ.get(key, "") for key in KEY_RUNTIME_ENVS}

                current_engine = InferenceEngine(model, config)

                for res in current_engine.run(resolved_video, question, answer, qt, opts_list):
                    if is_run_cancelled(run_id):
                        return
                    if res['type'] == 'step':
                        s = res['step']
                        base = [
                            *status_updates(
                                STATUS_RUNNING,
                                f"Step {s.step}/{ms} | Dynamic Step {'ON' if config.use_dynamic_step else 'OFF'} | TITO {'ON' if config.use_tito else 'OFF'}"
                            ),
                            gr.update(value=f"{res['total_reward']:.4f}"),
                            gr.update(value="")
                        ]
                        
                        steps_update = []
                        steps = current_engine.steps
                        for i in range(MAX_STEPS):
                            if i < len(steps):
                                st = steps[i]
                                steps_update.append(gr.update(visible=True))
                                steps_update.append(gr.update(value=get_step_html(st), visible=True))
                                
                                if st.action_type == 'get_clip' and st.clip_path and os.path.exists(st.clip_path):
                                    steps_update.append(gr.update(value=st.clip_path, visible=True))
                                    steps_update.append(gr.update(visible=False))
                                    steps_update.append(gr.update(visible=False))
                                elif st.action_type == 'get_audio' and st.audio_path and os.path.exists(st.audio_path):
                                    steps_update.append(gr.update(visible=False))
                                    steps_update.append(gr.update(value=st.audio_path, visible=True))
                                    steps_update.append(gr.update(visible=False))
                                elif st.action_type == 'get_frames' and st.frame_paths:
                                    frames = [f for f in st.frame_paths if os.path.exists(f)]
                                    if frames:
                                        steps_update.append(gr.update(visible=False))
                                        steps_update.append(gr.update(visible=False))
                                        steps_update.append(gr.update(value=render_frame_strip(frames), visible=True))
                                    else:
                                        steps_update.extend([gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
                                else:
                                    steps_update.extend([gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
                            else:
                                steps_update.extend([gr.update(visible=False), gr.update(value="", visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
                        
                        history_str = generate_history_json(
                            current_engine.steps,
                            current_engine.full_messages,
                            resolved_video,
                            question,
                            gt_answer=answer,
                            final_answer="",
                            question_type=qt,
                            options_text=opts,
                            source_label=source_label,
                            runtime_config=runtime_config_payload,
                            env_snapshot=env_snapshot_payload,
                            include_messages=False,
                        )
                        if is_run_cancelled(run_id):
                            return
                        yield base + steps_update + [gr.update(value=history_str), gr.update(value=""), gr.update(value=None)]

                    elif res['type'] == 'done':
                        steps = res['steps']
                        done_status = f"Done: {len(steps)} steps"
                        base = [
                            *status_updates(STATUS_DONE, done_status),
                            gr.update(value=f"{res['total_reward']:.4f}"),
                            gr.update(value=res['final'])
                        ]
                        
                        steps_update = []
                        for i in range(MAX_STEPS):
                            if i < len(steps):
                                st = steps[i]
                                steps_update.append(gr.update(visible=True))
                                steps_update.append(gr.update(value=get_step_html(st), visible=True))
                                
                                if st.action_type == 'get_clip' and st.clip_path and os.path.exists(st.clip_path):
                                    steps_update.append(gr.update(value=st.clip_path, visible=True))
                                    steps_update.append(gr.update(visible=False))
                                    steps_update.append(gr.update(visible=False))
                                elif st.action_type == 'get_audio' and st.audio_path and os.path.exists(st.audio_path):
                                    steps_update.append(gr.update(visible=False))
                                    steps_update.append(gr.update(value=st.audio_path, visible=True))
                                    steps_update.append(gr.update(visible=False))
                                elif st.action_type == 'get_frames' and st.frame_paths:
                                    frames = [f for f in st.frame_paths if os.path.exists(f)]
                                    if frames:
                                        steps_update.append(gr.update(visible=False))
                                        steps_update.append(gr.update(visible=False))
                                        steps_update.append(gr.update(value=render_frame_strip(frames), visible=True))
                                    else:
                                        steps_update.extend([gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
                                else:
                                    steps_update.extend([gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
                            else:
                                steps_update.extend([gr.update(visible=False), gr.update(value="", visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
                        
                        history_str = generate_history_json(
                            steps,
                            current_engine.full_messages,
                            resolved_video,
                            question,
                            gt_answer=answer,
                            final_answer=res['final'],
                            question_type=qt,
                            options_text=opts,
                            source_label=source_label,
                            runtime_config=runtime_config_payload,
                            env_snapshot=env_snapshot_payload,
                            include_messages=True,
                        )
                        if is_run_cancelled(run_id):
                            return
                        yield base + steps_update + [gr.update(value=history_str), gr.update(value=""), gr.update(value=None)]

                    elif res['type'] == 'error':
                        base = status_updates(STATUS_ERROR, res['error'][:80]) + [gr.update(value="0"), gr.update(value="")]
                        yield base + clear_step_outputs() + [gr.update(value="{}"), gr.update(value=""), gr.update(value=None)]

            except Exception as e:
                base = status_updates(STATUS_ERROR, str(e)[:80]) + [gr.update(value="0"), gr.update(value="")]
                yield base + clear_step_outputs() + [gr.update(value="{}"), gr.update(value=""), gr.update(value=None)]
            finally:
                if active_run_id == run_id:
                    cleanup_ray_runtime()
                    current_engine = None
                    is_running = False

        def on_force_stop():
            nonlocal current_engine, is_running, active_run_id
            if current_engine is None:
                return status_updates(STATUS_READY, "No inference is running.")
            active_run_id = 0
            current_engine = None
            is_running = False
            return status_updates(STATUS_DONE, "Force stop completed for the current UI run. Model stays loaded; backend work may finish its current step asynchronously.")

        # Build output list
        outputs = [state_out, status, reward_out, answer_out]
        for i in range(MAX_STEPS):
            outputs.append(step_containers[i])
            outputs.append(step_htmls[i])
            outputs.append(step_videos[i])
            outputs.append(step_audios[i])
            outputs.append(step_frame_strips[i])
        outputs.append(history_json)  # Add JSON history output
        outputs.append(export_status)
        outputs.append(export_file)

        # Configuration parameter input list
        config_inputs = [max_steps, max_frames, max_audio, max_clip, mode_input, dynamic_step_info, use_tito_info, temp_slider, top_p_slider, top_k_slider]
        
        example_dd.change(
            on_example,
            [example_dd],
            [selected_video_path, selected_video_label, video_card, video_upload, video_preview, question_in, answer_in, q_type, options_in],
        )
        video_upload.change(on_video_change, [video_upload], [selected_video_path, selected_video_label, video_card], queue=False)
        preview_btn.click(load_preview, [video_upload, selected_video_path], [video_preview], queue=False)
        clear_preview_btn.click(hide_preview, outputs=[video_preview], queue=False)
        start_event = start_btn.click(
            on_start,
            [video_upload, selected_video_path, selected_video_label, question_in, answer_in, q_type, options_in] + config_inputs,
            outputs,
        )
        force_stop_btn.click(on_force_stop, outputs=[state_out, status], cancels=[start_event], queue=False, show_progress="hidden")
        export_btn.click(
            export_result_bundle,
            [history_json, state_out, status, reward_out, answer_out, answer_in, q_type, options_in, selected_video_label],
            [export_status, export_file],
            queue=False,
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("Runtime env summary:")
    for key in KEY_RUNTIME_ENVS:
        print(f"  {key}={os.environ.get(key, '')}")

    # Clean up Ray temp files on startup
    print("🧹 Cleaning up Ray temp files...")
    import subprocess
    try:
        subprocess.run(["ray", "stop", "--force"], capture_output=True, timeout=10)
    except:
        pass
    
    for ray_dir in ["/tmp/ray", "/tmp/ray_spill"]:
        if os.path.exists(ray_dir):
            try:
                shutil.rmtree(ray_dir, ignore_errors=True)
            except:
                pass
    print("✅ Ray temp files cleaned")

    config = DemoConfig(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    demo = create_demo(config)
    demo.queue()
    demo.launch(
        server_name=args.host, server_port=args.port, share=args.share,
        allowed_paths=build_allowed_paths()
    )


if __name__ == "__main__":
    main()
