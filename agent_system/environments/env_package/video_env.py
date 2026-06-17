# ============================================================
#  video_env.py  –  unified error marker / message
#  Version: uses Ray global GlobalProcessor, completely avoids redundant tokenizer loading
# ============================================================

import json, logging, math, os, re, shlex, shutil, subprocess, time, uuid
from datetime import date, datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import ray
import time

from agent_system.environments.env_package.oss_reader import OssReader
from agent_system.environments.prompts import *
from verl.utils.reward_score.omni_agent import (
    compute_score_free_form,
    evaluate_reasoning_quality,
)
from verl.utils import hf_processor
from qwen_omni_utils import process_audio_info
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)
oss_reader = OssReader()
QUIET_LOGS = os.getenv("OMNIAGENT_QUIET_LOGS", "false").lower() in ("1", "true", "t", "yes", "y")


def info_print(*args, **kwargs):
    if not QUIET_LOGS:
        print(*args, **kwargs)

# ======================================================================
# 1.  Global tokenizer / processor -- loaded only once in a single Actor
# ======================================================================
@ray.remote
class GlobalProcessor:
    """Globally unique tokenizer / processor."""

    def __init__(self, path: str):
        self.proc = hf_processor(path, trust_remote_code=False, use_fast=True)
        info_print(f"[GlobalProcessor] loaded from {path}")

    # ---------- Exposed interfaces ----------
    def apply_chat_template(self, messages, add_generation_prompt=True):
        prompt = self.proc.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=False
        )
        # Some implementations may return a list; convert to str
        if isinstance(prompt, list):
            prompt = prompt[0] if len(prompt) == 1 else "".join(prompt)
        return prompt

    def count_tokens(
        self,
        prompt: str,
        img_inputs=None,
        vid_inputs=None,
        audio_inputs=None,
        use_audio_in_video=False,
    ) -> int:
        kwargs = {"text": [prompt], "return_tensors": "pt"}
        if img_inputs:
            kwargs["images"] = img_inputs
        if vid_inputs:
            kwargs["videos"] = vid_inputs
        if audio_inputs is not None and len(audio_inputs) > 0:
            kwargs["audio"] = audio_inputs
            kwargs["use_audio_in_video"] = use_audio_in_video
        else:
            kwargs["use_audio_in_video"] = False
        out = self.proc(**kwargs)
        return int(out["input_ids"][0].numel())


    def encode_message(self, message: Dict, has_audio: bool) -> Dict:
        """
        [Pure TITO upgrade - smart deduplication]
        1. Automatically compute Input IDs, Grid, Lengths.
        2. [Fix] Automatically remove default System Prompt forcefully inserted by apply_chat_template.
        """
        # 1. Media check
        has_media = False
        if "content" in message:
            for part in message["content"]:
                if part.get("type") in ["audio", "video", "image"]:
                    has_media = True; break
        messages = [message]
        img_inputs, vid_inputs, audio_inputs = None, None, None
        if has_media:
            img_inputs, vid_inputs = process_vision_info(messages)
            
            if has_audio:
                audio_inputs = process_audio_info(messages, use_audio_in_video=True)
        
        # 2. Generate Prompt (an unwanted system prompt may be inserted here)
        text_prompt = self.apply_chat_template(messages, add_generation_prompt=False)

        if isinstance(text_prompt, list): text_prompt = text_prompt[0]

        process_kwargs = {"text": [text_prompt], "return_tensors": "pt", "padding": False}
        if img_inputs: process_kwargs["images"] = img_inputs
        if vid_inputs: process_kwargs["videos"] = vid_inputs
        if audio_inputs:
             process_kwargs["audio"] = audio_inputs
             process_kwargs["use_audio_in_video"] = True

        batch_output = self.proc(**process_kwargs)
        
        # Get raw IDs
        raw_ids = batch_output["input_ids"][0].tolist()

        # =================================================================
        # [CRITICAL FIX] Parasitic System Prompt removal surgery
        # =================================================================
        # Only when the current message is *not* System, but the result contains a System header, do we need to remove it.
        # Criterion: if raw_ids contains two occurrences of <|im_start|> (151644),
        # the first is the auto-inserted System, and the second is our actual User/Asst.
        # =================================================================
        
        is_system_msg = (message.get("role") == "system")
        IM_START = 151644
        
        if not is_system_msg:
            # Find positions of <|im_start|>
            starts = [i for i, x in enumerate(raw_ids) if x == IM_START]
            
            if len(starts) > 1:
                # Found extra header! Typically [SystemStart, ..., UserStart, ...]
                # Keep the second Start and everything after it
                real_start_idx = starts[1]
                # print(f"[GlobalProcessor] Detected & Removed auto-inserted System Prompt ({real_start_idx} tokens).")
                raw_ids = raw_ids[real_start_idx:]
                
                # Note: Grid does not need trimming because System Prompt is pure text and doesn't consume Grid.

        # [Fix] Must convert to CPU
        def safe_cpu(t): return t.cpu() if t is not None else None

        ret = {
            "input_ids": raw_ids, # Using trimmed IDs
            "image_grid_thw": safe_cpu(batch_output.get("image_grid_thw")),
            "video_grid_thw": safe_cpu(batch_output.get("video_grid_thw")),
            "pixel_values": safe_cpu(batch_output.get("pixel_values")),
            "pixel_values_videos": safe_cpu(batch_output.get("pixel_values_videos")),
            "video_second_per_grid": safe_cpu(batch_output.get("video_second_per_grid")),
            "token_count": len(raw_ids),
            "mm_data": {
                "image": img_inputs or [],
                "video": vid_inputs or [],
                "audio": audio_inputs or [],
            },
        }
        if "feature_attention_mask" in batch_output:
            ret["feature_attention_mask"] = safe_cpu(batch_output["feature_attention_mask"])
            ret["audio_feature_lengths"] = safe_cpu(torch.sum(batch_output["feature_attention_mask"], dim=1))
        return ret

    # === [New] Expose decode interface ===
    def decode(self, token_ids: List[int]) -> str:
        # Try to decode through the processor's internal tokenizer
        if hasattr(self.proc, "tokenizer"):
            return self.proc.tokenizer.decode(token_ids)
        elif hasattr(self.proc, "decode"):
            return self.proc.decode(token_ids)
        return "[Error] GlobalProcessor cannot find tokenizer.decode"

def get_global_processor(path: str):
    """
    Get (or create) the globally unique processor Actor.
    Supports reuse across Jobs, and handles conflicts when multiple processes create simultaneously.
    """
    actor_name = "GLOBAL_PROCESSOR"
    # Explicitly specify namespace to prevent Actor lookup failures across different Jobs
    namespace = "omni_eval_ws"

    handle = None

    # 1. Try to get the existing Actor
    try:
        handle = ray.get_actor(actor_name, namespace=namespace)
        info_print(f"[Ray] Successfully found existing Actor: {actor_name}")
    except (ValueError, KeyError):
        # 2. If not found, try to create one
        info_print(f"[Ray] Actor {actor_name} not found. Attempting to start a new one...")
        try:
            # Try to start in detached mode so the Actor survives even after the current script ends
            handle = GlobalProcessor.options(
                name=actor_name,
                namespace=namespace,
                lifetime="detached",
                num_cpus=0  # Don't occupy CPU quota; serves only as a logical manager
            ).remote(path)
            info_print(f"[Ray] New GlobalProcessor remote object created.")
        except Exception as e:
            # 3. If we reach here, a concurrent creation conflict occurred (another process just created it first)
            info_print(f"[Ray] Race condition detected or creation failed: {e}")
            info_print(f"[Ray] Performing multi-round lookup retries...")
            
            # Retry lookup 10 times, with 2-second intervals, to allow Ray cluster to sync Actor registration
            for i in range(10):
                try:
                    time.sleep(2)
                    handle = ray.get_actor(actor_name, namespace=namespace)
                    if handle:
                        info_print(f"[Ray] Found Actor on retry attempt {i+1}")
                        break
                except (ValueError, KeyError):
                    continue

    # 4. Final state verification and logging
    if handle:
        info_print(f"[Ray] GlobalProcessor is READY. Handle: {handle}")
    else:
        # If we reach here, the Ray environment may have a serious configuration issue
        print(f"[Ray] CRITICAL: Failed to initialize GlobalProcessor handle after all attempts.")

    return handle



# ------------------ OSS helpers ------------------------------------------------
def _put_with_retry(bucket, object_key, local_path,
                    max_retry: int = 10, base_delay: float = 1.0):
    for n in range(1, max_retry + 1):
        try:
            bucket.put_object_from_file(object_key, local_path)
            return
        except Exception as e:
            if n >= max_retry:
                raise
            print(f"[upload retry] {object_key=} attempt {n}/{max_retry} failed: {e}")
            time.sleep(base_delay)

# ------------------ misc helpers ----------------------------------------------
def _spans_to_ndarray(spans):
    if spans is None:
        return np.empty((0, 2), float)

    if isinstance(spans, np.ndarray):
        a = spans.astype(float)
        if np.isnan(a).any() or np.isinf(a).any():
            raise ValueError("span contains NaN or Inf")
        return a.reshape(-1, 2) if a.ndim == 1 else a

    if isinstance(spans, (list, tuple)):
        if len(spans) == 0:
            return np.empty((0, 2), float)

        if isinstance(spans[0], (list, tuple, np.ndarray)):
            a = np.array([[float(x), float(y)] for x, y in spans], float)
            if np.isnan(a).any() or np.isinf(a).any():
                raise ValueError("span contains NaN or Inf")
            return a

        if len(spans) == 2 and all(isinstance(x, (int, float)) for x in spans):
            a = np.array([[float(spans[0]), float(spans[1])]], float)
            if np.isnan(a).any() or np.isinf(a).any():
                raise ValueError("span contains NaN or Inf")
            return a

    raise ValueError(f"Invalid span format: {spans!r}")


def _merge_spans(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    a = a[np.argsort(a[:, 0])]
    m = [a[0].tolist()]
    for s, e in a[1:]:
        if s <= m[-1][1]:
            m[-1][1] = max(m[-1][1], e)
        else:
            m.append([s, e])
    return np.asarray(m, float)


def overlap_ratio(p: np.ndarray, g: np.ndarray) -> float:
    if p.size == 0 and g.size == 0:
        return 1.0
    if p.size == 0 or g.size == 0:
        return 0.0
    p = _merge_spans(p.copy())
    g = _merge_spans(g.copy())
    inter = 0.0
    for s1, e1 in p:
        for s2, e2 in g:
            inter += max(0, min(e1, e2) - max(s1, s2))
    lp = np.sum(p[:, 1] - p[:, 0])
    lg = np.sum(g[:, 1] - g[:, 0])
    return float(inter / (lp + lg - inter + 1e-12))

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)
    eps = 1e-8
    rel_err = torch.abs(pred - target) / (torch.abs(target) + eps)
    ths = torch.arange(start, end + interval / 2, interval, dtype=torch.float32)
    mra = (rel_err < (1 - ths)).float().mean()
    return mra.item()


# ------------------ Err & helper ----------------------------------------------
class Err(Enum):
    STEP_LIMIT_REACHED    = auto()
    TOKEN_LIMIT_EXCEEDED  = auto()
    FFMPEG_AUDIO_FAIL     = auto()
    FFMPEG_CLIP_FAIL      = auto()
    FFMPEG_FRAME_FAIL     = auto()
    EARLY_ANSWER          = auto()
    ENV_ALREADY_DONE      = auto()
    INVALID_JSON          = auto()
    ANSWER_JSON_PARSE     = auto()
    GT_JSON_PARSE         = auto()
    UNKNOWN_ACTION_TYPE   = auto()
    FRAME_TS_OOB          = auto()
    TOO_MANY_FRAMES       = auto()
    INVALID_FRAME_NUM     = auto()
    AUDIO_ARG_TYPE        = auto()
    NO_AUDIO              = auto()
    AUDIO_RANGE           = auto()
    AUDIO_TOO_SHORT       = auto()
    AUDIO_TOO_LONG        = auto()
    CLIP_ARG_TYPE         = auto()
    CLIP_RANGE            = auto()
    CLIP_NEG_LEN          = auto()
    CLIP_TOO_LONG         = auto()
    CLIP_TOO_SHORT        = auto()
    ANSWER_NOT_STRING     = auto()
    SPAN_FORMAT           = auto()
    UNKNOWN_QTYPE         = auto()
    TRAILING_GARBAGE      = auto()


def _err(ec: Err, detail: str | None = None) -> tuple[str, str, str]:
    marker = ec.name
    return marker, marker, (detail or marker)


# ============================================================
#  SingleVideoQAEnv
# ============================================================
class SingleVideoQAEnv:

    def __init__(self, sample: Dict,
                 max_frames_len: int = 5,
                 max_audio_len: float = 10.0,
                 max_clip_len: float = 10.0,
                 max_steps: int = 5,
                 processor=None,
                 max_prompt_len=None,
                 max_response_len=None,
                 is_train=None):
        # ......(original large block unchanged)......
        # ------------ below omitted until assignment ends -------------
        self.rng = np.random.RandomState()
        self.sample = sample
        self.mode = sample.get("mode")
        self.video_path = sample.get("video")
        self.fps = sample.get("fps")
        self.question = sample.get("question")
        self.answer = sample.get("answer")
        self.options = sample.get("options", [])
        self.duration = sample.get("duration_seconds")
        self.has_audio = sample.get("has_audio")
        # 1. Save the raw field (could be NUM_count / SIZE_length etc.)
        self.raw_question_type = sample.get("question_type")
        # 2. Extract the major category prefix for all subsequent logic
        self.question_type = self.raw_question_type.split("_", 1)[0].upper()
        #    Keep consistent uppercase: MCQ / TR / FF / NUM / SIZE

        self.processor = processor                         # now a remote handle ###

        self.tok_len_tol = 32
        self.token_warn_margin = 512
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.is_train = is_train

        # 1. Basic resource initialization (as defaults)
        self.max_frames_len = max_frames_len
        self.max_audio_len  = max_audio_len
        self.max_clip_len   = max_clip_len
        self.min_audio_len  = 0.1

        # 2. Read feature flags
        self.use_random_resources = os.getenv("RANDOM_RESOURCE_SFT", "false").lower() in ("1", "true", "yes")
        info_print(f"[env] RANDOM_RESOURCE_SFT={self.use_random_resources}")
        self.use_dynamic_step = os.getenv("USE_DYNAMIC_STEP", "false").lower() in ("1", "true", "t", "yes", "y")
        info_print(f"[env] USE_DYNAMIC_STEP={self.use_dynamic_step}")

        # ======================================================================
        # Phase 1: Determine resource limits (grid randomization with step size 2)
        # ======================================================================
        if self.use_random_resources:
            # --- 1. Max Steps randomization (step size 2) ---
            # Generate sequence like [22, 24, 26, 28, 30, 32]
            s_start, s_end = min(22, max_steps), max_steps
            step_options = list(range(s_start, s_end + 1, 2))
            max_steps = self.rng.choice(step_options)

            # --- 2. Max Frames randomization (step size 2) ---
            # Generate sequence like [30, 32, 34 ... 60]
            f_start, f_end = min(30, max_frames_len), max_frames_len
            frame_options = list(range(f_start, f_end + 1, 2))
            self.max_frames_len = self.rng.choice(frame_options)

            # --- 3. Max Clip length randomization (step size 2) ---
            # Also using 2.0 as step size, e.g. [30.0, 32.0, 34.0 ...]
            c_ceil  = max(30.0, max_clip_len)
            clip_options = [float(x) for x in range(int(30.0), int(c_ceil) + 1, 2)]
            self.max_clip_len = self.rng.choice(clip_options)

            # --- 4. Max Audio length randomization (step size 10) ---
            a_ceil  = max(150.0, max_audio_len)
            audio_options = [float(x) for x in range(int(150.0), int(a_ceil) + 1, 10)]
            self.max_audio_len = self.rng.choice(audio_options)

        # ======================================================================
        # Phase 2: Determine actual step count (Step Calculation)
        # ======================================================================
        if self.use_dynamic_step:
            # Here max_steps may be the original value or the randomized upper limit from Phase 1
            # Calculate actual steps based on video duration and (possibly randomized) max_clip_len
            min_steps = int(os.getenv("MIN_MAX_STEPS", "5"))
            self.max_steps = min(min_steps + int(self.duration / self.max_clip_len), max_steps)
        else:
            # Static assignment
            self.max_steps = max_steps

        self.use_oss = os.getenv("USE_OSS_IN_VIDEOENV", "false").lower() in (
            "1", "true", "t", "yes", "y"
        )

        self.oss_path   = os.getenv("OSS_PATH", "omniagent/agentic_tmp/")
        self.oss_bucket = os.getenv("OSS_BUCKET", "")
        self.ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")
        self.base_tmp_dir = os.getenv("VIDEO_ENV_TMP_DIR", "./video_env_tmp")
        os.makedirs(self.base_tmp_dir, exist_ok=True)
        self.auto_cleanup = os.getenv("VIDEO_ENV_AUTO_CLEANUP", "false").lower() in (
            "1", "true", "yes"
        )

        self.bypass_dur_check = os.getenv("BYPASS_DURATION_CHECK", "false").lower() in ("1", "true", "t", "yes", "y")
        info_print(f"[env] BYPASS_DURATION_CHECK={self.bypass_dur_check}")
        if self.bypass_dur_check:
            self.dur_tol = 99999.0
        else:
            # Default 1.0 second, a very robust production value
            self.dur_tol = 1.0

        # Record last fetched media file paths
        self.last_clip_path = None
        self.last_audio_path = None
        self.last_frame_paths = []

        self.use_format_reward = os.getenv("FORMAT_REWARD", "false").lower() in (
            "1", "true", "t", "yes", "y"
        )
        if self.use_format_reward:
            info_print("FORMAT_REWARD Activated")

        self.use_reason_reward = os.getenv("REASONING_REWARD", "false").lower() in (
            "1", "true", "t", "yes", "y"
        )
        if self.use_reason_reward:
            info_print("REASONING_REWARD Activated")

        self.delete_media_when_error = os.getenv("DELETE_MEDIA_WHEN_ERROR", "false").lower() in (
            "1", "true", "t", "yes", "y"
        )
        info_print(f"[env] DELETE_MEDIA_WHEN_ERROR={self.delete_media_when_error}")

        self.done = False
        self.temp_dir = None
        self.step_count = 0
        self.history = None

        # 1. Get retry mode configuration
        self.retry_mode = os.getenv("RETRY_ON_FORMAT_ERROR", "false").lower() in ("1", "true", "t", "yes", "y")

        # 2. Modify the logic here
        if self.is_train:
            # If retry mode is enabled, don't terminate env on Invalid Action even during training
            # This allows the model to see errors in History and retry
            self.terminate_on_invalid = not self.retry_mode
        else:
            self.terminate_on_invalid = False

        info_print(
            f"[env] is_train={is_train}, terminate_on_invalid={self.terminate_on_invalid}"
        )

        # ======== Length-bonus params & counters ========
        self.len_bonus_on = os.getenv("LEN_BONUS_ENABLE", "false").lower() in (
            "1", "true", "t", "yes", "y"
        )
        self.len_beta      = float(os.getenv("LEN_BETA", "0.25"))
        self.len_bonus_max = float(os.getenv("LEN_BONUS_MAX", "0.5"))


        self.enable_tito =  os.getenv("USE_TITO", "false").lower() in ("1", "true", "t", "yes", "y")  # Save toggle state
        # For compatibility, ensure processor exists if required
        if self.enable_tito and self.processor is None:
            raise ValueError("TITO mode requires a valid processor handle.")

    def upload_and_sign(self, local_path: Path,
                        bucket_name: str | None = None,
                        object_key: str | None = None,
                        prefix: str | None = None) -> tuple[str, str]:
        bucket_name = bucket_name or self.oss_bucket
        prefix = prefix or os.getenv("OSS_PREFIX", "agentic_tmp/")
        p = Path(local_path).absolute()
        
        if not p.is_file():
            # Construct detailed error diagnostic info
            # Note: assumes self already has these attributes.
            # If stored in a sample dict, replace self. with sample.get()
            error_details = (
                f"\n[File Not Found Diagnostic Info]:"
                f"\n  - Missing Path: {p}"
                f"\n  - Video Path: {getattr(self, 'video_path', 'N/A')}"
                f"\n  - FPS: {getattr(self, 'fps', 'N/A')}"
                f"\n  - Question: {getattr(self, 'question', 'N/A')}"
                f"\n  - Answer: {getattr(self, 'answer', 'N/A')}"
                f"\n  - Options: {getattr(self, 'options', 'N/A')}"
                f"\n  - Duration: {getattr(self, 'duration', 'N/A')}s"
                f"\n  - Has Audio: {getattr(self, 'has_audio', 'N/A')}"
            )
            # Throw exception with this info attached
            raise FileNotFoundError(error_details)
        
        bucket = oss_reader.bucket[bucket_name]
        
        # ---------- New: parse local path to match OSS directory ----------
        # Your path structure is: .../video_env_tmp/YYYYMMDD/video_env_PID_UUID/filename.ext
        # p.name is the filename (e.g. STEP_1_frame_10.500.jpg)
        # p.parent.name is the env directory (e.g. video_env_1003_dd24a83c...)
        # p.parent.parent.name is the date directory (e.g. 20260114)
        

        filename = p.name
        env_dir_name = p.parent.name
        date_dir = p.parent.parent.name
        
        # Check if it matches the expected structure (date dir should be 8 digits)
        if date_dir.isdigit() and len(date_dir) == 8:
            # Assemble into OSS path: agentic_tmp/20260114/video_env_xxx/filename.jpg
            if object_key is None:
                object_key = f"{prefix}{date_dir}/{env_dir_name}/{filename}"
        else:
            # If structure doesn't match, fall back to preserving at least the env directory name
            if object_key is None:
                object_key = f"{prefix}{env_dir_name}/{filename}"


        oss_uri = f"oss://{bucket_name}/{object_key}"
        _put_with_retry(bucket, object_key, str(p))
        
        # Signed URL remains unchanged
        signed = bucket.sign_url("GET", object_key, 86400, slash_safe=True)
        return oss_uri, signed

    def build_user_msg(self, text="", file_path:Path|None=None, mime:str|None=None):
        content=[]
        # 1. Add text first (timestamp info)
        if text: 
            content.append({"type":"text","text":text})
        
        # 2. Then add media
        if file_path and mime:
            url=self.upload_and_sign(Path(file_path),self.oss_bucket,prefix=self.oss_path)[1] if self.use_oss else str(file_path)
            t=mime.split('/')[0]
            content.append({"type":t,"image" if t=="image" else t: url})
            
        return {"role":"user","content":content}

    # ---------------- reset -----------------------
    def reset(self):
        # Clean up old directory (only when auto-cleanup is needed and not uploading to OSS)
        if (self.auto_cleanup
                and self.temp_dir
                and os.path.exists(self.temp_dir)
                and not self.use_oss):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

        # ========== Manually create layered temp directory ==========
        today = datetime.now().strftime("%Y%m%d")
        dir_root = os.path.join(self.base_tmp_dir, today)
        os.makedirs(dir_root, exist_ok=True)          # Create if not exists

        # Directory name example: video_env_12345_17c8f0a2baa14e52b6e9a9af43ce1d6e
        self.temp_dir = os.path.join(
            dir_root, f"video_env_{os.getpid()}_{uuid.uuid4().hex}"
        )
        os.makedirs(self.temp_dir, exist_ok=True)
        # ========================================
        self.done=False
        self.step_count=0
        self.total_resp_tokens = 0   # Accumulated total assistant response tokens
        self.current_turn_retries = 0  # Reset counter

        def trunc(x,n=2): return "unknown" if not isinstance(x,(int,float)) else f"{math.floor(x*10**n)/10**n:.{n}f}"
        meta=(f"Video META:\n- duration_seconds: {trunc(self.duration)}\n"
              f"- fps: {trunc(self.fps)}\n- has_audio: {self.has_audio}\n\n")
        if self.question_type == "MCQ":
            opts = ("\nOptions:\n" + "\n".join(self.options) +
                    "\nWhen answering, set action.content to ONE uppercase letter (A, B, C …).")
            qtext = meta + "Question: " + self.question + opts
        elif self.question_type == "TR":
            guide = ("\nWhen answering, set action.content to a JSON array "
                     "of timestamp pairs such as [[10.5, 20.0]].")
            qtext = meta + "Question: " + self.question + guide
        elif self.question_type == "FF":                      # <<< NEW
            guide = ("\nWhen answering, set action.content to **your free-form answer text**.")
            qtext = meta + "Question: " + self.question + guide
        elif self.question_type == "NUM":                       # <<< NUM NEW
            guide = ("\nWhen answering, set action.content to "
                     "**ONE number**, e.g. 42 or 3.14159.")          # <<< NUM NEW
            qtext = meta + "Question: " + self.question + guide # <<< NUM NEW
        elif self.question_type == "SIZE":                       # <<< SIZE NEW
            guide = ("\nWhen answering, set action.content to "
                     "**ONE number**, e.g. e.g., 203 or 10.3.")          # <<< SIZE NEW
            qtext = meta + "Question: " + self.question + guide # <<< SIZE NEW
        else:
            raise ValueError(f"Unknown question_type {self.question_type}")

        self.sample["max_steps"] = self.max_steps
        builder={"OmniAgent":build_video_prompt_omniagent}[self.mode]

        self.system_prompt=builder(self.max_steps,self.max_frames_len,
                                   self.max_audio_len,self.max_clip_len)
        self.history=[{"role":"system","content":[{"type":"text","text":self.system_prompt}]},
                      self.build_user_msg(qtext)]
        info={**self.sample,"won":False,"is_action_valid":True,"error_code":None,"error_message":None}

        # [TITO FIX] Initialize lists and call unified sync
        self.token_segments = [] if self.enable_tito else None
        self.tito_history = [] if self.enable_tito else None

        self._append_step_notice(done=False)

        if self.enable_tito:
            self._sync_tito_state()  # <--- One-call sync, replaces previous hand-written futures

        # [TITO key change] Return dict instead of list, carrying token_segments
        obs_payload = {
            "text": self.history,
            "token_segments": self.token_segments if self.enable_tito else None
        }
        return obs_payload, info


    # ---------------- step --------------------------
    def step(self, action: str, action_ids: list = None):
        # 1. Update text
        self.history.append({"role":"assistant","content":[{"type":"text","text":action}]})
        
        # 2. [TITO Fix] Token-Out closed loop
        if self.enable_tito:
            # Ensure segments are aligned to the size before history insertion (usually aligned, but for safety)
            assert len(self.token_segments) == len(self.history)-1, f"TITO state misalignment: {len(self.token_segments)} segments vs {len(self.history)} history entries"
            assert action_ids is not None, "TITO mode requires non-null action_ids for assistant output"

            # Manually assemble Assistant Token (Sandwich Logic)
            clean_content_ids = [tid for tid in action_ids]
            header_ids = [151644, 77091, 198] # <|im_start|>assistant\n
            footer_ids = [151645, 198]        # <|im_end|>\n
            final_ids = header_ids + clean_content_ids + footer_ids

            # Direct Append, no need for remote encode -- this is TITO's core advantage (reusing tokens from generation)
            seg_data = {"input_ids": final_ids, "token_count": len(final_ids)}
            self.token_segments.append(seg_data)
            
            # Sync update tito_history (Decode back to text for debug)
            # Note: for performance, decode is still remote, but placeholder append can be done immediately
            while len(self.tito_history) < len(self.token_segments) - 1:
                self.tito_history.append(None)

        if self.done:
            return self._fail(*_err(Err.ENV_ALREADY_DONE),terminate=True)

        self.step_count+=1

        # ===== Length-bonus: accumulate tokens & steps =====
        if self.len_bonus_on and self.is_train:
            try:
                tok_this = ray.get(
                    self.processor.count_tokens.remote(
                        action,
                        img_inputs=None,
                        vid_inputs=None,
                        audio_inputs=None,
                        use_audio_in_video=False,
                    )
                )
            except Exception:
                tok_this = 0
            self.total_resp_tokens += tok_this

        obj,ec,msg=self._parse_action(action)
        if ec:
            # extra = "Please output exactly ONE line that matches the schema."
            # full_msg = f"{msg}. {extra}" if msg else extra
            full_msg = f"{msg}."
            return self._fail("INVALID_JSON", ec, full_msg)


        act=obj["action"]; atype=act.get("type","")
        
        # Exceeded max allowed steps, fail immediately
        if self.step_count > self.max_steps:
            return self._fail(*_err(Err.STEP_LIMIT_REACHED), terminate=True)

        # If this is the last step but the action is not answer, fail immediately
        if self.step_count == self.max_steps and atype != "answer":
            return self._fail(*_err(Err.STEP_LIMIT_REACHED), terminate=True)


        # ---------- EARLY ANSWER CHECK ----------
        if atype == "answer" and self.step_count == 1:
            return self._fail(*_err(Err.EARLY_ANSWER,
                                    "Must gather evidence before answering"))

        # ---------- get_frames (new logic: range burst capture) ----------
        if atype == "get_frames":
            s, e = float(act.get("start", 0)), float(act.get("end", 0))
            num = int(act.get("num", 0))
            
            # --- Basic validation ---
            if s < 0 or e > self.duration:
                return self._fail(*_err(Err.FRAME_TS_OOB, f"Range [{s}, {e}] OOB for {self.duration}s"))
            if e < s:
                return self._fail(*_err(Err.CLIP_NEG_LEN, f"end {e} < start {s}"))
            if num > self.max_frames_len:
                return self._fail(*_err(Err.TOO_MANY_FRAMES, f"num {num} > max {self.max_frames_len}"))
            if num < 1:
                return self._fail(*_err(Err.INVALID_FRAME_NUM, "num must be >= 1"))

            # 1. Generate precise timestamp list
            ts = np.linspace(s, e, num).tolist() if num > 1 else [s]
            
            # Expanded View: range information only
            # header = f"Frames {s:.2f}s-{e:.2f}s (num={num})."
            if num > 1:
                header = f"Frames {s:.2f}s-{e:.2f}s (num={num})."
            else:
                header = f"Frame at {s:.2f}s (num={num})."

            parts = [{"type": "text", "text": header}]
            self.last_frame_paths = []  # Clear previous frame paths

            for t in ts:
                try:
                    img = self._get_frame(t)
                    self.last_frame_paths.append(img)  # Record frame path
                except:
                    return self._fail(*_err(Err.FFMPEG_FRAME_FAIL, f"Bad frame {t:.2f}s"), terminate=True, is_action_valid=True)
                if not img or not os.path.isfile(img) or os.path.getsize(img)==0:
                    # Print diagnostic info for troubleshooting
                    print(f"\n[FRAME_FAIL] Failed to extract frame at {t:.2f}s from video: {getattr(self, 'video_path', 'N/A')}")
                    if img and os.path.exists(img):
                        print(f"  - File exists but size is {os.path.getsize(img)} bytes: {img}")
                    else:
                        print(f"  - File does not exist: {img}")
                    return self._fail(*_err(Err.FFMPEG_FRAME_FAIL),terminate=True,is_action_valid=True)
                url = self.upload_and_sign(Path(img), self.oss_bucket, prefix=self.oss_path)[1] if self.use_oss else str(img)
                # Interleaved display
                parts.append({"type": "text", "text": f"Frame {t:.2f}s:"})
                parts.append({"type": "image", "image": url})
            
            self.history.append({"role": "user", "content": parts})
            return self._ret(0.0, False, {**self.sample, "won": False, "is_action_valid": True})

        # ---------- get_audio ----------
        if atype=="get_audio":
            if not self.has_audio:
                return self._fail(*_err(Err.NO_AUDIO))
            s,e=act.get("start"),act.get("end")
            # --- Boundary check: provide clear hints for each ---
            if s < 0:
                return self._fail(*_err(Err.AUDIO_RANGE,
                                        f"start {s:.3f}s < 0"))
            if e > self.duration:
                return self._fail(*_err(Err.AUDIO_RANGE,
                                        f"end {e:.3f}s > duration {self.duration:.3f}s"))
            dur=e-s
            if dur<=0: return self._fail(*_err(Err.CLIP_NEG_LEN))
            if dur<self.min_audio_len:
                return self._fail(*_err(Err.AUDIO_TOO_SHORT,
                                        f"{dur:.3f}s < {self.min_audio_len}s"))
            if dur>self.max_audio_len:
                return self._fail(*_err(Err.AUDIO_TOO_LONG,
                                        f"{dur:.3f}s > {self.max_audio_len}s"))
            try:
                ap=self._get_audio(s,e)
                self.last_audio_path = ap  # Record audio path
            except:
                return self._fail(*_err(Err.FFMPEG_AUDIO_FAIL),terminate=True,is_action_valid=True)
            if not ap or not os.path.isfile(ap) or os.path.getsize(ap) == 0:
                # Same diagnostic info print
                print(f"\n[AUDIO_FAIL] Failed to extract audio segment from video: {getattr(self, 'video_path', 'N/A')}")
                if ap and os.path.exists(ap):
                    print(f"  - File exists but size is {os.path.getsize(ap)} bytes: {ap}")
                else:
                    print(f"  - File does not exist: {ap}")
                return self._fail(*_err(Err.FFMPEG_AUDIO_FAIL), terminate=True, is_action_valid=True)
            # Validate actual audio duration
            #####################################
            dur_expect = e - s
            info_dur   = self.get_media_durations(ap) or {}
            audio_dur   = info_dur.get("audio")

            if (audio_dur is None
                    or audio_dur <= 0
                    or abs(audio_dur - dur_expect) > self.dur_tol*5
                    or audio_dur < self.min_audio_len):
                return self._fail(
                    *_err(Err.FFMPEG_AUDIO_FAIL,
                        f"Bad audio duration {audio_dur} (expect {dur_expect})"),
                    terminate=True,
                    is_action_valid=True,
                )

             #####################################
            self.history.append(self.build_user_msg(f"Audio {s:.2f}s-{e:.2f}s", Path(ap), "audio/wav"))
            return self._ret(0.0,False,{**self.sample,"won":False,"is_action_valid":True})

        # ---------- get_clip ----------
        if atype=="get_clip":
            s,e=act.get("start"),act.get("end")
            # Start/end boundary check, with separate hints
            if s < 0:
                return self._fail(*_err(Err.CLIP_RANGE,
                                        f"start {s:.3f}s < 0"))
            if e > self.duration:
                return self._fail(*_err(Err.CLIP_RANGE,
                                        f"end {e:.3f}s > video duration {self.duration:.3f}s"))
            if e<=s:
                return self._fail(*_err(Err.CLIP_NEG_LEN))
            dur=e-s
            if dur>self.max_clip_len:
                return self._fail(*_err(Err.CLIP_TOO_LONG,
                                        f"{dur:.3f}s > {self.max_clip_len}s"))
            min_dur = 4.0 / float(self.fps)  # seconds corresponding to 4 frames
            if self.fps and self.fps >= 2 and dur * self.fps < 3.999:
                return self._fail(
                    *_err(Err.CLIP_TOO_SHORT,
                        f"{dur:.3f}s < {min_dur:.3f}s (need ≥ 4 frames)"),
                    terminate=False,
                    is_action_valid=True
                )
            try:
                cp=self._get_clip(s,e)
                self.last_clip_path = cp  # Record video clip path
            except:
                return self._fail(*_err(Err.FFMPEG_CLIP_FAIL),terminate=True,is_action_valid=True)
            if not cp or not os.path.isfile(cp) or os.path.getsize(cp) == 0:
                # Prominent diagnostic info
                print(f"\n[CLIP_FAIL] Failed to extract video clip from: {getattr(self, 'video_path', 'N/A')}")
                if cp and os.path.exists(cp):
                    print(f"  - File exists but size is {os.path.getsize(cp)} bytes: {cp}")
                else:
                    print(f"  - File does not exist: {cp}")
                return self._fail(*_err(Err.FFMPEG_CLIP_FAIL), terminate=True, is_action_valid=True)
            # Validate video duration
            #####################################
            # Validate video/audio duration
            dur_expect = e - s
            info_dur   = self.get_media_durations(cp) or {}
            video_dur  = info_dur.get("video")
            audio_dur  = info_dur.get("audio")
            bad = False
            # if video_dur is None or video_dur <= 0: # or (video_dur - dur_expect) > self.dur_tol
            #     bad = True
            # if video_dur and video_dur - self.max_clip_len > self.dur_tol:
            #     bad = True
            # if self.has_audio:
            #     if audio_dur is None or audio_dur <= 0 or audio_dur < self.min_audio_len: # or abs(audio_dur - dur_expect) > self.dur_tol
            #         bad = True

            if video_dur and abs(video_dur - dur_expect) > self.dur_tol*5:
                bad = True
            # 1. Basic video existence check
            elif video_dur is None or video_dur <= 0:
                bad = True
            # 2. Audio logic check
            elif self.has_audio:
                # 2a. Does audio stream exist
                if audio_dur is None or audio_dur <= 0 or audio_dur < self.min_audio_len:
                    bad = True
                # 2b. Audio-video sync/integrity check (with safety guard)
                elif abs(audio_dur - video_dur) > self.dur_tol:
                    bad = True

            if bad:
                return self._fail(
                    *_err(Err.FFMPEG_CLIP_FAIL,
                        f"Bad clip duration v={video_dur}, a={audio_dur}, expect {dur_expect}"),
                    terminate=True,
                    is_action_valid=True,
                )
            #####################################
            self.history.append(self.build_user_msg(f"Clip {s:.2f}s-{e:.2f}s", Path(cp), "video/mp4"))

            return self._ret(0.0,False,{**self.sample,"won":False,"is_action_valid":True})

        # ---------- answer ----------
        if atype == "answer":
            content=act.get("content")
            if self.question_type=="MCQ":
                if not self.answer:
                    won=False; reward=0.0
                else:
                    gt=self.answer[0].strip().upper()
                    won=content.strip().upper()==gt; reward=1.0 if won else 0.0
            elif self.question_type == "TR":
                try: 
                    pred = json.loads(content)
                except Exception as e:
                    return self._fail(*_err(Err.ANSWER_JSON_PARSE, str(e)))

                # 1. Initial GT retrieval
                raw_gt = self.answer

                # 2. If GT is a string, try to parse it into a list object
                if isinstance(raw_gt, str):
                    try:
                        raw_gt = json.loads(raw_gt)
                    except:
                        pass # If parsing fails, type checking will catch it later

                # 3. Strict validation: TR task answer must be a list or tuple (representing a span or list of spans)
                if not isinstance(raw_gt, (list, tuple, np.ndarray)):
                    # Return Fail directly, without any "point-to-span" guessing
                    return self._fail(
                        *_err(Err.GT_JSON_PARSE, 
                             f"TR answer must be a list/tuple or ndarray, but got {type(raw_gt).__name__}: {raw_gt}")
                    )

                # 4. Use _spans_to_ndarray for matrix conversion
                # This function natively supports:
                # - [s, e] -> recognized as a flat list of length 2
                # - [[s, e], [s, e]] -> recognized as a nested list
                try:
                    p = _spans_to_ndarray(pred)
                    g = _spans_to_ndarray(raw_gt)
                except Exception as e:
                    # If the list length is not 2 or the content is not numeric, an error will occur here
                    return self._fail(*_err(Err.SPAN_FORMAT, str(e)))

                # 5. Compute IoU
                iou = overlap_ratio(p, g)
                won     =  iou>=0.5
                reward  =  iou 
            # ---------- FF  (NEW) ----------
            elif self.question_type == "FF":
                # self.answer stores the ground-truth text (single-element list or direct str)
                gt_text = self.answer if isinstance(self.answer, str) else self.answer[0]
                pred_text = content.strip()
                reward = compute_score_free_form(pred_text, gt_text, self.question)
                won = reward >= 0.5
            elif self.question_type in ["NUM", "SIZE"]:                  # <<< NUM NEW
                gt_str = self.answer[0] if isinstance(self.answer, list) else self.answer
                try:                                           # Parse GT
                    gt_val = float(re.findall(r'[+-]?\d+(?:\.\d+)?', str(gt_str))[0])
                except Exception:
                    return self._fail(*_err(Err.GT_JSON_PARSE,
                                            f"cannot parse gt number: {gt_str!r}"))
                # Prediction
                try:
                    pred_val = float(re.findall(r'[+-]?\d+(?:\.\d+)?', content)[0])
                except Exception:
                    return self._fail(*_err(Err.ANSWER_NOT_STRING,
                                            "Cannot parse number from answer."))

                if self.question_type == "NUM":           # 0 / 1
                    if round(gt_val, 2) == round(pred_val, 2):
                        reward = 1.0
                    else:
                        reward = 0.0
                elif self.question_type == "SIZE":
                    reward = mean_relative_accuracy(pred_val, gt_val)
                won = reward >= 0.5
            else:
                return self._fail(*_err(Err.UNKNOWN_QTYPE))

            # Strict reasoning quality check
            eval_result = None
            if self.use_reason_reward and self.is_train and won:
                if self.question_type == "MCQ":
                    gt_str = self.answer[0].strip()
                elif self.question_type == "TR":
                    gt_str = str(self.answer)
                else:  # FF
                    gt_str = self.answer if isinstance(self.answer, str) else self.answer[0]
                
                eval_result = evaluate_reasoning_quality(
                    history=self.history,
                    question=self.question,
                    answer=content,
                    ground_truth=gt_str,
                    question_type=self.question_type,
                    options=self.options
                )
                
                # Any quality issue results in 0 score
                if eval_result and not eval_result.get('pass_quality_check', True):
                    reward = 0.0
                    # won = False
                    # reason = eval_result.get('reason', 'unknown')
                    # logger.warning(f"[reasoning_eval] Quality check failed: {reason}")

                    
            self.done=True
            self.history.append(self.build_user_msg("CORRECT ANSWER" if won else "INCORRECT ANSWER"))

            if self.use_format_reward:
                reward = 0.1 + 0.9*reward

            # ---------- Length bonus (only when enabled & won) ----------
            bonus = 0.0
            avg_tok = 0.0
            if self.len_bonus_on and won and self.step_count > 0 and self.is_train:
                avg_tok   = self.total_resp_tokens / self.step_count
                beta      = self.len_beta          # 0.007
                cutoff    = 100
                bonus_max = self.len_bonus_max     # 2.01375  (or 2)

                # Formula: bonus = bonus_max * exp(-beta * max(avg_tok+1, cutoff))
                bonus = bonus_max * math.exp(-beta * max(avg_tok + 1.0, cutoff))
                print(f"[LENGTH_BONUS] avg_tok={avg_tok:.2f} total_tok={self.total_resp_tokens} bonus={bonus:.4f}")

                reward += bonus

                # Write stats to info for monitoring / tuning
                info_extra = {
                    "_total_resp_tokens": int(self.total_resp_tokens),
                    "_total_steps":       int(self.step_count),
                    "_avg_resp_tokens":   float(avg_tok),
                    "_length_bonus":      float(bonus),
                }


            # Build info, including reasoning evaluation result
            info = {**self.sample, "won": won, "is_action_valid": True}
            # info.update(info_extra)
            if eval_result:
                info["reasoning_eval"] = {
                    "pass_quality_check": eval_result.get("pass_quality_check", True),
                    "reason": eval_result.get("reason", "")
                }

            return self._ret(reward, True, info)


    # ======== Token counting only, no row_dict generation =========
    def _count_tokens_with_expansion(self) -> int:
        if self.enable_tito:
            # 1. Force sync (Sanity Check)
            #    If length mismatches or contains None, trigger sync immediately
            if len(self.token_segments) != len(self.history) or any(s is None for s in self.token_segments):
                raise RuntimeError(f"[TITO FATAL] Token segments out-of-sync BEFORE count! Segments len={len(self.token_segments)}, History len={len(self.history)}")

            # 2. Fast accumulation (Fail-Fast Logic)
            total_count = 0
            for i, seg in enumerate(self.token_segments):
                # A. Fatal error check: should never be None after sync
                if not seg:
                    raise RuntimeError(f"[TITO FATAL] Segment {i} is None AFTER sync! History len={len(self.history)}")
                
                # B. Golden path: directly read pre-stored token_count (fastest)
                if "token_count" in seg:
                    total_count += seg["token_count"]
                
                # C. Silver path: if no token_count, try computing via input_ids (compatibility fallback)
                elif "input_ids" in seg:
                    val = seg["input_ids"]
                    # Compatible with PyTorch Tensor and Python List
                    count = int(val.numel()) if hasattr(val, "numel") else len(val)
                    total_count += count
                
                # D. Dead end: data structure is completely corrupted
                else:
                    raise RuntimeError(f"[TITO FATAL] Segment {i} missing 'token_count' AND 'input_ids'. Keys: {list(seg.keys())}")
            
            return int(total_count)

        # ===============================================
        # [Fallback] Regular remote path when TITO is not enabled
        # ===============================================

        # 1. chat template
        prompt = ray.get(
            self.processor.apply_chat_template.remote(
                self.history, add_generation_prompt=True
            )
        )
        if isinstance(prompt, list): 
            prompt = prompt[0] if len(prompt) == 1 else "".join(prompt)

        # 2. multimodal inputs
        img_inputs, vid_inputs = process_vision_info(self.history)
        audio_inputs = (
            process_audio_info(self.history, use_audio_in_video=True)
            if self.has_audio
            else None
        )
        use_audio = audio_inputs is not None and len(audio_inputs) > 0

        # 3. Remote computation
        return ray.get(
            self.processor.count_tokens.remote(
                prompt,
                img_inputs=img_inputs,
                vid_inputs=vid_inputs,
                audio_inputs=audio_inputs,
                use_audio_in_video=use_audio,
            )
        )

    # -------- Check if full prompt exceeds length limit --------
    def _prompt_too_long(self):
        prompt_len = self._count_tokens_with_expansion()
        exceed = prompt_len > (self.max_prompt_len - self.tok_len_tol)
        return exceed, prompt_len

    # ============================================================
    # helper: unified step-remaining notice append (ensures only one notice exists)
    # ============================================================
    def _append_step_notice(self, done: bool):
        # if done: return
        remain = self.max_steps - self.step_count
        # if remain <= 0: return

        # 1. Clean old notices
        for i, msg in enumerate(self.history):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue

             # Check if it contains a notice
            original_len = len(msg["content"])
            msg["content"] = [
                part for part in msg["content"]
                if not (
                    part.get("type") == "text"
                    and part.get("text", "").lstrip().startswith("[NOTICE]")
                )
            ]

            if len(msg["content"]) != original_len and self.enable_tito and i < len(self.token_segments):
                # [TITO FIX] Mark dirty data
                self.token_segments[i] = None
                if self.tito_history is not None and i < len(self.tito_history):
                    self.tito_history[i] = None

        # 2. Notice text covering all scenarios
        if done:
            if remain <= 0:
                # Scenario A: Step limit exhausted, terminated by system
                notice_text = "\n[NOTICE] STEP LIMIT REACHED. Operation terminated."
            else:
                # Scenario B: [New] Model answered within limit, or failed (e.g. format error), but steps remain
                notice_text = f"\n[NOTICE] Operation terminated. Steps used: {self.step_count}/{self.max_steps}."
        else:
            if remain == 1:
                # Scenario C: Last chance
                notice_text = f"\n[NOTICE] FINAL STEP! You MUST provide your answer now."
            else:
                # Scenario D: Normal countdown
                notice_text = f"\n[NOTICE] Step {self.step_count}/{self.max_steps}. {remain} steps remaining."

        # Append to the last User message
        if self.history[-1]["role"] == "user":
            self.history[-1]["content"].append({"type": "text", "text": notice_text})
            # [TITO FIX] Mark the last entry as dirty
            if self.enable_tito:
                # Ensure segments length is sufficient (though _sync_tito_state will handle it, for safety)
                if len(self.token_segments) < len(self.history):
                    self.token_segments.append(None)
                else:
                    self.token_segments[-1] = None

    # ============================================================
    # [TITO FIX] Core sync mechanism: Dirty-Checking & Batch Processing
    # ============================================================
    def _sync_tito_state(self):
        """
        [High-performance version] Sync tokens only, never decode text.
        """
        if not self.enable_tito:
            return

        # 1. Align lengths (pad with None)
        target_len = len(self.history)
        while len(self.token_segments) < target_len:
            self.token_segments.append(None)
        
        # tito_history also needs length alignment to keep indices consistent, but content is None
        while len(self.tito_history) < target_len:
            self.tito_history.append(None)

        # 2. Identify dirty data (only check if tokens are missing)
        if len(self.token_segments) > target_len:
             # Defensive truncation
            self.token_segments = self.token_segments[:target_len]
            self.tito_history = self.tito_history[:target_len]

        dirty_indices = [i for i, seg in enumerate(self.token_segments) if seg is None]

        if not dirty_indices:
            return

        # 3. Batch concurrent encoding (Batch Encode)
        encode_futures = [
            self.processor.encode_message.remote(self.history[i], self.has_audio) 
            for i in dirty_indices
        ]
        
        # 4. Block and wait for results (only blocks this once)
        new_segments = ray.get(encode_futures)

        # 5. Backfill tokens (never decode text)
        for i, seg in zip(dirty_indices, new_segments):
            self.token_segments[i] = seg

    # ============================================================
    # Unified exit point: all common logic goes here
    # ============================================================
    def _finalize(self, reward: float, done: bool, info: dict,
                  is_action_valid: bool):
        keep_count = 0 if (self.delete_media_when_error and not is_action_valid) else 1
        self._replace_old_media(keep_count)

        if self.enable_tito:
            self._sync_tito_state() # One call handles both Case A and Case B

        exceed, prompt_len = self._prompt_too_long()
        info_print(
            f"[TOKEN_CHECK] exceed={exceed} prompt={prompt_len} "
            f"reserve={self.max_response_len} tol={self.tok_len_tol} "
            f"limit={self.max_prompt_len}"
        )

        # Token budget warning
        # remain_tok = self.max_prompt_len - prompt_len
        # if (not done) and remain_tok <= self.token_warn_margin:
        #     warn = (f"\n[NOTICE] Only {remain_tok} prompt tokens left "
        #             f"(≤ {self.token_warn_margin}).")
        #     if self.history and isinstance(self.history[-1].get("content"), list):
        #         self.history[-1]["content"].append({"type": "text", "text": warn})

        
        # ========== Record detailed token statistics (for SFT filtering) ==========
        info["token_stats"] = {
            "final_history_tokens": prompt_len,         # Total token count of the entire conversation history (most important metric)
            "total_assistant_tokens": self.total_resp_tokens, # Total tokens of all agent responses
            "avg_assistant_tokens": self.total_resp_tokens / max(1, self.step_count), # Average response length per step
            "max_prompt_limit": self.max_prompt_len,    # Hard limit set by the environment
        }

        # Previous config records also preserved
        info["env_config"] = {
            "actual_used_steps": self.step_count,
            "actual_max_steps": self.max_steps,
            "actual_max_frames": self.max_frames_len,
            "actual_max_audio": self.max_audio_len,
            "actual_max_clip": self.max_clip_len,
            "video_duration": self.duration,
            "current_turn_retries": self.current_turn_retries
        }
        # =========================================================

        if exceed:
            # Length exceeded => terminate immediately
            code  = Err.TOKEN_LIMIT_EXCEEDED.name           # marker / error_code
            msg   = f"Prompt length {prompt_len} > {self.max_prompt_len}"
            err_txt = f"{code}: {msg}"                      # <<< Include marker in message

            err_msg = {"type": "text", "text": f"[ERROR] {err_txt}"}

            if self.history and self.history[-1].get("role") == "user":
                # Case A: Modify existing message
                self.history[-1]["content"].append(err_msg)
                
                # [TITO adaptation] Mark the last entry as dirty
                if self.enable_tito:
                    # Ensure list exists and is long enough (prevent out-of-bounds)
                    if self.token_segments and len(self.token_segments) >= len(self.history):
                        self.token_segments[-1] = None
                    # If segments is shorter than history, sync will auto-append None, no action needed here
            else:
                # Case B: Append new message
                self.history.append(self.build_user_msg(f"[ERROR] {err_txt}"))
                
                # [TITO adaptation] Nothing to do here!
                # Because history grew longer but token_segments didn't.
                # Next call to _sync_tito_state will auto-detect the length mismatch,
                # auto-append(None) and compute new tokens.

            reward = 0.0
            done   = True
            info.update({"error_code": code,
                        "error_message": msg,
                        "won": False})

        # Step countdown
        self._append_step_notice(done)

        # Write back state
        if done: self.done = True
        info.setdefault("reward", reward)

        # -------------------------------------------------------
        # Unified pre-exit sync
        # -------------------------------------------------------
        if self.enable_tito:
            self._sync_tito_state() # One call handles both Case A and Case B

        # =================================================================
        # [KEY CHANGE] Lazy Batch Decode
        # Only fill tito_history when done or exceed
        # =================================================================
        # if self.enable_tito and (done or exceed):
        #     # 1. Re-align lengths (in case exceed appended new messages)
        #     while len(self.tito_history) < len(self.token_segments):
        #         self.tito_history.append(None)

        #     # 2. Scan for slots that have tokens but no text
        #     missing_indices = [
        #         i for i, txt in enumerate(self.tito_history) 
        #         if txt is None and self.token_segments[i] is not None
        #     ]
            
        #     if missing_indices:
        #         # 3. Issue batch decode requests
        #         futures = [
        #             self.processor.decode.remote(self.token_segments[i]['input_ids'])
        #             for i in missing_indices
        #         ]
        #         # 4. Block and wait (only one decode blocking per episode)
        #         decoded_texts = ray.get(futures)
                
        #         # 5. Back-fill text
        #         for idx, txt in zip(missing_indices, decoded_texts):
        #             self.tito_history[idx] = txt
            
        #     # 6. (Optional) At this point self.tito_history is complete; can print or store in info
        #     if done: print(f"[DEBUG] Final History: {self.tito_history}")

        obs_payload = {
            "text": self.history,
            "token_segments": self.token_segments if self.enable_tito else None
        }

        # Add media file paths to info
        info["clip_path"] = getattr(self, 'last_clip_path', None)
        info["audio_path"] = getattr(self, 'last_audio_path', None)
        info["frame_paths"] = getattr(self, 'last_frame_paths', [])

        return obs_payload, reward, done, info


    # ============================================================
    # Valid exit -- only one line remains
    # ============================================================
    def _ret(self, reward: float, done: bool, info: dict,
             is_action_valid: bool = True):
        info["reward"] = reward
        return self._finalize(reward, done, info, is_action_valid)


    # ============================================================
    # Invalid exit -- minimal
    # ============================================================
    def _fail(self, marker, code, msg,
              terminate: bool | None = None,
              is_action_valid: bool = False):
        self.current_turn_retries += 1 # Increment on each failure
        # If caller did not explicitly set terminate, decide based on config + step limit
        if terminate is None:
            # Must terminate at the last valid step or when already exceeded
            terminate = terminate or (self.step_count >= self.max_steps) or self.terminate_on_invalid

        err_txt = f"{marker}: {msg}" if msg and msg != marker else str(marker)
        if not terminate:
            err_txt += ("\n[SYSTEM] The previous action was invalid. Analyze the error diagnostic above, resolve the conflict, and issue a corrected command to avoid wasting steps.")

        self.history.append(self.build_user_msg(f"[ERROR] {err_txt}"))

        info = {
            **self.sample,
            "won": False,
            "is_action_valid": is_action_valid,
            "error_code": code,
            "error_message": msg,
        }

        return self._finalize(0.0, terminate, info, is_action_valid)


    # ---------------- parse_action & helpers ---------------------------------
    ###########################################################################
    def _strip_code_fence(self, s: str) -> str | None:
        """
        Strip leading/trailing whitespace and reject outer wrapping.
        Returns None if illegal wrapping is detected (should report INVALID_JSON).
        """
        s = s.strip()
        # Reject ```code block``` or outer paired quotes
        if s.startswith("```") or s.endswith("```"):
            return None
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return None
        return s


    def _parse_action(self, raw: str):
        """
        Strict but minimal JSON parsing:
        - Single-line or multi-line allowed; after stripping whitespace, must be wrapped in { ... };
        - Reject outer code-fence / quotes;
        - No extra characters after parsing;
        - Required top-level keys: observation / think / action;
        - When env var STRICT_CONFIDENCE_CHECK is true, confidence key is also required;
        - Validates observation / think are strings, action is an object, and action.type is valid.
        """
        if not isinstance(raw, str):
            return None, Err.INVALID_JSON.name, "not a string"

        s = self._strip_code_fence(raw)
        if s is None:
            return None, Err.INVALID_JSON.name, \
                "must be raw JSON without code fence or outer quotes"

        # Must start with '{' and end with '}'
        if not (s.startswith("{") and s.rstrip().endswith("}")):
            return None, Err.INVALID_JSON.name, "must start with '{' and end with '}'"

        # Use raw_decode to catch trailing garbage
        try:
            obj, end_pos = json.JSONDecoder().raw_decode(s)
        except json.JSONDecodeError as e:
            return None, Err.INVALID_JSON.name, f"json decode error: {e}"

        if s[end_pos:].strip():                     # Extra characters after parsing
            return None, Err.TRAILING_GARBAGE.name, "extra characters after JSON"

        # ---------- Top-level key validation (dynamic compatibility logic) ----------
        is_strict_conf = os.getenv("STRICT_CONFIDENCE_CHECK", "false").lower() in ("1", "true", "t", "yes", "y")
        
        required = {"observation", "think", "action"}
        if is_strict_conf:
            required.add("confidence")
            
        provided = set(obj.keys())
        
        # 1. Check if required keys are missing (includes confidence in strict mode)
        if not required.issubset(provided):
            miss = required - provided
            return None, Err.INVALID_JSON.name, f"missing keys: {sorted(miss)}"
            
        # 2. Check for unknown keys (only core three + confidence allowed)
        allowed = {"observation", "think", "action", "confidence"}
        extra = provided - allowed
        if extra:
            return None, Err.INVALID_JSON.name, f"unknown keys: {sorted(extra)}"

        # ---------- Basic types ----------
        if not isinstance(obj["observation"], str):
            return None, Err.INVALID_JSON.name, "'observation' must be string"
        if not isinstance(obj["think"], str):
            return None, Err.INVALID_JSON.name, "'think' must be string"
            
        # If confidence appears in the JSON, it must be numeric
        if "confidence" in obj:
            if not isinstance(obj["confidence"], (int, float)):
                return None, Err.INVALID_JSON.name, "'confidence' must be a numeric value"

        act = obj["action"]
        if not isinstance(act, dict):
            return None, Err.INVALID_JSON.name, "'action' not object"

        atype = act.get("type")
        if atype not in {"get_frames", "get_audio", "get_clip", "answer"}:
            return None, Err.UNKNOWN_ACTION_TYPE.name, f"unknown action.type: {atype}"

        # ---------- Further field checks ----------
        if act["type"] == "get_frames": # Validate get_frames parameters
            for fld in ("start", "end"):
                if fld not in act or not isinstance(act[fld], (int, float)):
                    return None, Err.INVALID_JSON.name, f"get_frames requires numeric '{fld}'"
            if "num" not in act or not isinstance(act["num"], int):
                return None, Err.INVALID_JSON.name, "get_frames requires integer 'num'"

        elif act["type"] in {"get_clip", "get_audio"}:
            for fld in ("start", "end"):
                if fld not in act or not isinstance(act[fld], (int, float)):
                    err = Err.CLIP_ARG_TYPE if act["type"] == "get_clip" else Err.AUDIO_ARG_TYPE
                    return None, err.name, f"'{fld}' missing or not number"
        
        elif act["type"] == "answer":
            if "content" not in act or not isinstance(act["content"], str):
                return None, Err.ANSWER_NOT_STRING.name, "'content' missing or not string"

        # All checks passed
        return obj, None, None
    ##################################################################

    # ---------- helper: return the first text segment ----------
    ##################################################################
    def _first_text_part(self, content: list):
        for part in content:
            if part.get("type") == "text":
                return part.get("text", "")
        return ""

    def _extract_media_id(self, user_msg):
        txt = self._first_text_part(user_msg.get("content", []))
        return txt.split()[0] if txt else None

    def _replace_old_media(self, KEEP_RECENT: int):
        if KEEP_RECENT < 0: KEEP_RECENT = 0
        media_kept = 0
        SUFFIX = "[MEDIA OMITTED - Refer to your Observation]"
        
        # [TITO FIX] Step 1: Align lengths first to prevent index out-of-bounds in the loop below
        if self.enable_tito:
            while len(self.token_segments) < len(self.history):
                self.token_segments.append(None)
            while len(self.tito_history) < len(self.history):
                self.tito_history.append(None)

        total_items = len(self.history)
        dirty_flag = False  # Track whether any modifications were made

        # Iterate in reverse order
        for i in range(total_items - 1, -1, -1):
            old = self.history[i]

            # Filter out non-User or non-List content
            if old.get("role") != "user" or not isinstance(old.get("content"), list):
                continue

            # Determine media type
            has_image = any(p.get("type") == "image" for p in old["content"])
            has_other = any(p.get("type") in ("video", "audio") for p in old["content"])
            
            if not (has_image or has_other): continue

            media_kept += 1
            if media_kept <= KEEP_RECENT: 
                continue # Keep the most recent ones

            # ================= Compression logic =================
            raw_header = "Media content"
            if len(old["content"]) > 0 and old["content"][0].get("type") == "text":
                raw_header = old["content"][0]["text"].strip()

            new_text = ""
            if has_image:
                all_ts = []
                for p in old["content"][1:]:
                    if p.get("type") == "text":
                        found = re.findall(r"(\d+(?:\.\d+)?)s", p.get("text"))
                        if found: all_ts.extend(found)
                ts_str_list = ", ".join([f"{float(x):.2f}s" for x in all_ts])
                new_text = f"{raw_header} Timestamps: [{ts_str_list}] {SUFFIX}"
            else:
                new_text = f"{raw_header} {SUFFIX}"

            # [KEY] Modify History (Semantic)
            self.history[i]["content"] = [{"type": "text", "text": new_text}]

            # [TITO FIX] Only mark as None; do not compute immediately to avoid blocking within the loop
            if self.enable_tito:
                self.token_segments[i] = None
                self.tito_history[i] = None  # Clear UI history as well, pending update
                dirty_flag = True

    ##################################################################

    # ======== Helper: unified ffmpeg path ========
    def _ffmpeg(self) -> str:
        return self.ffmpeg_path

    # ======== Helper: run command (retry + timeout + fallback logging) ========
    def _run_ffmpeg(self, cmd: str, retries: int = 2, backoff: float = 1.0, timeout: float = 100.0):
        last_err = None
        for k in range(retries + 1):
            try:
                # Improvement: capture stderr for debugging
                res = subprocess.run(
                    cmd, shell=True,
                    capture_output=True, text=True,
                    check=True, timeout=timeout
                )
                return
            except Exception as e:
                # Get the specific stderr message
                stderr_msg = getattr(e, 'stderr', str(e))
                last_err = stderr_msg
                if k < retries:
                    time.sleep(backoff)
                else:
                    # Print diagnostic info before raising the final exception
                    print("\n" + "!"*30 + " FFMPEG FINAL FAILURE " + "!"*30)
                    print(f"Failed Command: {cmd}")
                    print(f"Video Path:  {getattr(self, 'video_path', 'N/A')}")
                    print(f"FPS:         {getattr(self, 'fps', 'N/A')}")
                    print(f"Question:    {getattr(self, 'question', 'N/A')}")
                    print(f"Answer:      {getattr(self, 'answer', 'N/A')}")
                    print(f"Options:     {getattr(self, 'options', 'N/A')}")
                    print(f"Duration:    {getattr(self, 'duration', 'N/A')}s")
                    print(f"Has Audio:   {getattr(self, 'has_audio', 'N/A')}")
                    print("!"*80 + "\n")
                    
                    # Raise the original exception
                    raise RuntimeError(last_err)
    # ======== Helper: probe duration (used when sample value is missing or unreliable) ========
    def _probe_duration(self) -> float | None:
        if isinstance(self.duration, (int, float)) and self.duration > 0:
            return float(self.duration)
        else:
            return None

    def get_media_durations(self, file_path: str) -> dict | None:
        """
        Analyzes a media file with ffprobe to get video and audio stream durations.

        Args:
            file_path: The absolute or relative path to the media file.

        Returns:
            A dictionary containing 'video' and 'audio' durations in seconds (float),
            or None if an error occurs or a stream is not found.
            Example: {'video': 120.5, 'audio': 120.52}
        """
        # Construct the ffprobe command
        command = [
            "ffprobe",
            "-v", "quiet",              # Suppress all logging except for errors
            "-print_format", "json",    # Output in JSON format
            "-show_streams",            # Get information about streams
            file_path
        ]

        try:
            # Execute the command
            result = subprocess.run(
                command,
                capture_output=True,  # Capture stdout and stderr
                text=True,            # Decode output as text (UTF-8)
                check=True            # Raise CalledProcessError if return code is non-zero
            )
            
            # Parse the JSON output from stdout
            media_info = json.loads(result.stdout)

            # Initialize durations to None
            video_duration = None
            audio_duration = None

            # Iterate through the streams to find video and audio
            if 'streams' in media_info:
                for stream in media_info['streams']:
                    # Find the first video stream and get its duration
                    if stream.get('codec_type') == 'video' and video_duration is None:
                        if 'duration' in stream:
                            video_duration = float(stream['duration'])

                    # Find the first audio stream and get its duration
                    elif stream.get('codec_type') == 'audio' and audio_duration is None:
                        if 'duration' in stream:
                            audio_duration = float(stream['duration'])
            
            return {
                'video': video_duration,
                'audio': audio_duration
            }

        except FileNotFoundError:
            print("Error: ffprobe is not installed or not in your system's PATH.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error processing file with ffprobe: {file_path}")
            print(f"ffprobe stderr:\n{e.stderr}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Failed to parse ffprobe JSON output for {file_path}")
            return None


    # ======== Helper: safe time range clamping (inward shrink) ========
    def _safe_range(self, start: float, end: float, eps: float = 0.05) -> tuple[float, float, float]:
        d = self._probe_duration()
        if d is not None:
            end = min(end, d - eps)
        end = max(end, start)  # Do not allow inversion
        dur = max(0.0, end - start)
        return start, end, dur

    # ===================== Robust frame extraction =====================
    def _get_frame(self, ts: float) -> str:
        out = os.path.join(self.temp_dir, f"STEP_{self.step_count}_frame_{ts:.3f}.jpg")
        ff = self._ffmpeg()
        d = self._probe_duration()

        ts = max(0.0, ts)
        if d is not None:
            ts = min(ts, d - 0.2)

        # Limit threads to prevent Ray cluster crash
        base_args = "-hide_banner -loglevel error -nostdin -y -threads 8"

        # Attempt list
        cmds = [
            # 1. Fastest and most universal approach
            f'{ff} {base_args} -ss {ts:.3f} -i {shlex.quote(self.video_path)} -frames:v 1 -q:v 2 {shlex.quote(out)}',
        ]
        
        # 2. If near the end, add fallback attempts seeking from the end
        if d is not None and ts >= max(0.0, d - 3):
            for offset in [0.5, 1.0, 2.0, 3.0]:
                if offset < d: # Ensure offset does not exceed total duration
                    cmds.append(f'{ff} {base_args} -sseof -{offset} -i {shlex.quote(self.video_path)} -frames:v 1 -q:v 2 {shlex.quote(out)}')

        for cmd in cmds:
            try:
                self._run_ffmpeg(cmd, retries=1)
                if not os.path.isfile(out) or os.path.getsize(out)==0:
                    continue
                return out
            except:
                continue
        
        raise RuntimeError(f"All ffmpeg attempts failed for frame at {ts}")



    # ===================== Robust audio export =====================
    def _get_audio(self, start: float, end: float) -> str:
        """
        Stability-first: decode to WAV (16k/mono/pcm_s16le).
        If you must use AAC (container stream copy), change -c:a below to copy,
        but stability will decrease.
        """
        start, end, dur = self._safe_range(start, end, eps=0.2)
        if dur <= 0:
            raise ValueError(f"Invalid audio segment: start={start}, end={end}")

        out = os.path.join(self.temp_dir, f"STEP_{self.step_count}_audio_{start:.3f}_{end:.3f}.wav")
        ff = self._ffmpeg()

        # Two-pass seek: coarse jump before input + precise seek after input; decode to stable specs
        cmd = (
            f'{ff} -hide_banner -loglevel error -nostdin -y -threads 8 '
            f'-ss {start:.3f} -i {shlex.quote(self.video_path)} '
            f'-ss 0 -t {dur:.3f} '
            f'-map 0:a:0? -vn -ac 1 -ar 16000 -c:a pcm_s16le '
            f'{shlex.quote(out)}'
        )

        # If you insist on AAC (faster but less stable): change the audio codec above to ->  -c:a aac -b:a 128k   or   -c:a copy (when source is AAC and cut at keyframe)
        self._run_ffmpeg(cmd, retries=2)

        return out


    # ===================== Robust video clipping =====================
    # ===================== Robust video clipping =====================
    def _get_clip(self, start: float, end: float) -> str:
        """
        Stability-first: two-pass attempt. Try superfast preset first, fall back to ultrafast on failure.
        """
        start, end, dur = self._safe_range(start, end, eps=0.2)
        if dur <= 0:
            raise ValueError(f"Invalid clip segment: start={start}, end={end}")

        out = os.path.join(self.temp_dir, f"STEP_{self.step_count}_clip_{start:.3f}_{end:.3f}.mp4")
        ff = self._ffmpeg()
        v_path = shlex.quote(self.video_path)
        out_path = shlex.quote(out)

        # 1. First attempt: use superfast preset
        cmd = (
            f'{ff} -hide_banner -loglevel error -nostdin -y -threads 8 '
            f'-ss {start:.3f} -i {v_path} '
            f'-ss 0 -t {dur:.3f} '
            f'-map 0:v:0? -map 0:a:0? '
            f'-c:v libx264 -pix_fmt yuv420p -preset superfast -crf 20 '
            f'-movflags +faststart '
            f'-c:a aac -b:a 128k -ar 48000 '
            f'{out_path}'
        )

        try:
            self._run_ffmpeg(cmd, retries=2)
            return out
        except Exception as e:
            # 2. Second attempt: if above failed, fall back to lower-overhead ultrafast
            print(f"[_get_clip] Superfast failed, retrying with ultrafast fallback: {e}")
            cmd = (
                f'{ff} -hide_banner -loglevel error -nostdin -y -threads 8 '
                f'-ss {start:.3f} -i {v_path} '
                f'-ss 0 -t {dur:.3f} '
                f'-map 0:v:0? -map 0:a:0? '
                f'-c:v libx264 -pix_fmt yuv420p -preset ultrafast -crf 20 '
                f'-movflags +faststart '
                f'-c:a aac -b:a 128k -ar 48000 '
                f'{out_path}'
            )
            try:
                self._run_ffmpeg(cmd, retries=2)
                return out
            except Exception as e2:
                # 3. Complete failure: raise exception
                error_txt = f"Failed to extract clip {start}-{end} from video {self.video_path}"
                print(f"[_get_clip] {error_txt}: {e2}")
                raise RuntimeError(error_txt) from e2

    def close(self):
        # Clean up temporary directory
        if (self.auto_cleanup
                and hasattr(self, "temp_dir")
                and os.path.exists(self.temp_dir)
                and not self.use_oss):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                print(f"Cleaned tmp dir: {self.temp_dir}")
            except Exception as e:
                print(f"Failed to clean tmp dir {self.temp_dir}: {e}")


# ============================================================
#  Ray Worker
# ============================================================
@ray.remote(num_cpus=0, num_gpus=0)
class VideoEnvWorker:
    def __init__(
        self,
        max_frames_len=5,
        max_audio_len=10.0,
        max_clip_len=10.0,
        max_steps=5,
        processor_handle=None,
        max_prompt_len=None,
        max_response_len=None,
        is_train=None,
    ):
        self.max_frames_len = int(max_frames_len)
        self.max_audio_len = int(max_audio_len)
        self.max_clip_len = int(max_clip_len)
        self.max_steps = int(max_steps)
        self.current_env = None

        self.processor = processor_handle  # Only save the handle
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.is_train = is_train


    def reset_with_data(self, video_data: Dict, seed: int = None):
        """
        Create an environment with the provided video path and metadata, then reset it.

        Args:
            video_data: Dictionary containing video information
            video_data = {
                'video': gen_batch.non_tensor_batch.get('video', []),
                'question': gen_batch.non_tensor_batch.get('question', []),
                'answer': gen_batch.non_tensor_batch.get('answer', []),
                'options': gen_batch.non_tensor_batch.get('options', []),
                'fps': gen_batch.non_tensor_batch.get('fps', []),
                'duration_seconds': gen_batch.non_tensor_batch.get('duration_seconds', []),
                'has_audio': gen_batch.non_tensor_batch.get('has_audio', []),
                'mode': gen_batch.non_tensor_batch.get('mode', []),
            }
            seed: Random seed
        """
        if self.current_env is not None:
            # Old env may have open files, numpy/torch tensors, etc.; close before deleting reference
            self.current_env.close()
            self.current_env = None
            import gc; gc.collect()

            
        # Build sample data
        sample = {
            "video": video_data.get('video'),
            "fps": video_data.get('fps'),
            "duration_seconds": video_data.get('duration_seconds'),
            "has_audio": video_data.get('has_audio'),
            "question_type": video_data.get('question_type'),
            "question": video_data.get('question'),
            "answer": video_data.get('answer'),
            "options": video_data.get('options', []),
            "mode": video_data.get('mode'),
        }
        
        # Create new environment instance
        self.current_env = SingleVideoQAEnv(
            sample,
            max_frames_len=self.max_frames_len,
            max_audio_len=self.max_audio_len,
            max_clip_len=self.max_clip_len,
            max_steps=self.max_steps,
            processor     =self.processor,        # << inject
            max_prompt_len=self.max_prompt_len,    # << inject
            max_response_len = self.max_response_len,
            is_train = self.is_train,
        )
        
        return self.current_env.reset()
        
    def step(self, action: str, action_ids: List[int] = None): # Added parameter
        if self.current_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.current_env.step(action, action_ids) # Pass through
    
    def close(self):
        """Close the environment"""
        if self.current_env is not None:
            self.current_env.close()


# ============================================================
#  Multi-env wrapper (VideoQAMultiEnv) -- only __init__ modified
# ============================================================
class VideoQAMultiEnv:
    def __init__(
        self,
        env_num,
        group_n,
        seed,
        max_frames_len,
        max_audio_len,
        max_clip_len,
        max_steps,
        max_prompt_len,
        max_response_len,
        is_train,
        processor_handle,  # required
    ):
        self.env_num = env_num
        self.group_n = group_n
        self.total_workers = env_num * group_n
        self.rng = np.random.RandomState(seed)

        self.max_frames_len = max_frames_len
        self.max_audio_len = max_audio_len
        self.max_clip_len = max_clip_len
        self.max_steps = max_steps
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.is_train = is_train

        self.processor_handle = processor_handle  # Save handle

        self.workers = []
        for _ in range(self.total_workers):
            w = VideoEnvWorker.remote(
                max_frames_len=self.max_frames_len,
                max_audio_len=self.max_audio_len,
                max_clip_len=self.max_clip_len,
                max_steps=self.max_steps,
                processor_handle=self.processor_handle,  # Pass the same handle
                max_prompt_len=self.max_prompt_len,
                max_response_len=self.max_response_len,
                is_train=self.is_train,
            )
            self.workers.append(w)
    
    def reset_with_data(self, video_data_batch):
        """
        Reset using the provided data.

        Args:
            video_data_batch: Dictionary containing video data
            video_data_batch = {
                'video': gen_batch.non_tensor_batch.get('video', []),
                'question_type': gen_batch.non_tensor_batch.get('question_type', []),
                'question': gen_batch.non_tensor_batch.get('question', []),
                'answer': gen_batch.non_tensor_batch.get('answer', []),
                'options': gen_batch.non_tensor_batch.get('options', []),
                'fps': gen_batch.non_tensor_batch.get('fps', []),
                'duration_seconds': gen_batch.non_tensor_batch.get('duration_seconds', []),
                'has_audio': gen_batch.non_tensor_batch.get('has_audio', []),
                'mode': gen_batch.non_tensor_batch.get('mode', []),
            }
        """
        # Generate seeds for reset
        seeds = self.rng.randint(0, 2**16 - 1, size=self.total_workers).tolist()
        
        question = video_data_batch.get('question', [])

        # Send reset_with_data commands to all workers
        futures = []
        for i, (worker, seed) in enumerate(zip(self.workers, seeds)):
            # Cycle through the provided data
            data_idx = i % max(len(question), 1) if len(question) > 0 else 0
            
            # Build video data
            video_data = {
                'video': video_data_batch['video'][data_idx],
                "question_type": video_data_batch['question_type'][data_idx],
                'question': video_data_batch['question'][data_idx],
                'answer': video_data_batch['answer'][data_idx],
                'options': video_data_batch['options'][data_idx],
                'fps': video_data_batch['fps'][data_idx],
                'duration_seconds': video_data_batch['duration_seconds'][data_idx],
                'has_audio': video_data_batch['has_audio'][data_idx],
                'mode': video_data_batch['mode'],
            }
            future = worker.reset_with_data.remote(video_data, seed)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list = []
        info_list = []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def step(self, actions: List[str], action_ids: List[List[int]] = None): # Accept parameters
        """Perform step in parallel"""
        if len(actions) != self.total_workers:
            raise ValueError(f"Expected {self.total_workers} actions, got {len(actions)}")

        # Send step commands to all workers
        futures = []
        # Iterate over both actions and action_ids simultaneously
        # Handle case when action_ids is None
        if action_ids is None:
            action_ids = [None] * len(actions)
        for worker, action, act_id in zip(self.workers, actions, action_ids):
            future = worker.step.remote(action, act_id) # Pass through
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def close(self):
        """Close all Ray actors"""
        # Check if workers attribute exists
        if not hasattr(self, 'workers'):
            return
            
        # Send close commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.close.remote()
            futures.append(future)
            
        # Wait for all workers to close
        try:
            ray.get(futures)
        except Exception as e:
            print(f"Error while closing workers: {e}")
            pass  # Ignore errors during cleanup
        
        # Kill all Ray actors
        for worker in self.workers:
            try:
                ray.kill(worker)
            except Exception as e:
                print(f"Error while killing worker: {e}")
                pass  # Ignore errors during cleanup

    def __del__(self):
        self.close()


# ============================================================
#  factory
# ============================================================
def build_video_envs(
    seed: int,
    env_num: int,
    group_n: int,
    max_frames_len: int,
    max_audio_len: float,
    max_clip_len: float,
    max_steps: int,
    processor_path: str,
    max_prompt_len: int,
    max_response_len: int,
    is_train: bool,
):
    """Build VideoQAMultiEnv"""
    gp_handle = get_global_processor(processor_path)
    return VideoQAMultiEnv(
        env_num=env_num,
        group_n=group_n,
        seed=seed,
        max_frames_len=max_frames_len,
        max_audio_len=max_audio_len,
        max_clip_len=max_clip_len,
        max_steps=max_steps,
        max_prompt_len=max_prompt_len,
        max_response_len=max_response_len,
        is_train=is_train,
        processor_handle=gp_handle,
    )

# ============================================================
#  projection hook (improved action validity check)
# ============================================================
def video_projection(actions, *args, **kwargs):
    """
    dummy one
    """
    return actions, [True] * len(actions)
