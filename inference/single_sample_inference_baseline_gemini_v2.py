#!/usr/bin/env python3
"""
Single-sample baseline inference: call Gemini once with (video + question).

Usage:
    python inference/single_sample_inference_baseline_gemini_v2.py \
        --video_path assets/example_video_mcq.mp4 \
        --question 'Who or what lauds "Immigrant Diaries" as "A SURE FIRE HIT", according to the video?' \
        --question_type MCQ \
        --options "A. Remote Goat.\\nB. The New York Times.\\nC. Variety.\\nD. IndieWire."

Requires DASHSCOPE_API_KEY (or OSS_ACCESS_KEY) in env or .env file.
"""

import base64
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any
import subprocess

import requests
from dotenv import load_dotenv
from pathlib import Path
from agent_system.environments.env_package.oss_reader import OssReader
from datetime import date, datetime

oss_reader = OssReader()


# ----------------------------------------------------------------------
# Environment variables
# ----------------------------------------------------------------------
load_dotenv()
# Get API key from environment variables
API_KEY = os.getenv("OSS_ACCESS_KEY")
if not API_KEY:
    API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    print("⛔️ Please set OSS_ACCESS_KEY or DASHSCOPE_API_KEY in .env or environment variables")
    sys.exit(1)

# API configuration - Endpoint supporting native protocol
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# Default model
DEFAULT_MODEL = "vertex_ai.gemini-3.1-pro-preview"
# DEFAULT_MODEL = "gemini-2.5-pro"


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
            
def upload_and_sign(local_path: Path,
                    bucket_name: str | None = None,
                    object_key: str | None = None,
                    prefix: str | None = None) -> tuple[str, str]:
    bucket_name = bucket_name or os.environ.get("OSS_BUCKET", "")
    prefix = prefix or os.environ.get("OSS_PREFIX", "omniagent/agentic_tmp/")
    p = Path(local_path)
    if not p.is_file():
        raise FileNotFoundError(p)
    bucket = oss_reader.bucket[bucket_name]
    today  = date.today().strftime("%Y%m%d")
    ext    = p.suffix
    object_key = object_key or f"{prefix}{today}/{uuid.uuid4().hex}{ext}"
    oss_uri = f"oss://{bucket_name}/{object_key}"
    _put_with_retry(bucket, object_key, str(p))
    signed = bucket.sign_url("GET", object_key, 86400, slash_safe=True)
    return oss_uri, signed

# ----------------------------------------------------------------------
def call_gemini(messages: List[Dict[str, Any]],
                model: str = DEFAULT_MODEL,
                retries: int = 100,
                backoff: float = 1) -> Dict[str, Any]:
    """
    Single-turn DashScope Gemini call (Native Protocol) with automatic retry.
    Supports capturing Thinking process.
    """
    
    # 1. Convert OpenAI format messages to Gemini Native format contents
    native_contents = []
    for msg in messages:
        role = "user" if msg['role'] == "user" else "model"
        parts = []
        if isinstance(msg['content'], list):
            for item in msg['content']:
                if item.get('type') == 'text':
                    parts.append({"text": item['text']})
                elif item.get('type') == 'video_url':
                    parts.append({
                        "fileData": {
                            "mimeType": "video/mp4",
                            "fileUri": item['video_url']['url']
                        }
                    })
                elif item.get('type') == 'inline_video':
                    parts.append({
                        "inlineData": {
                            "mimeType": item.get('mime_type', 'video/mp4'),
                            "data": item['data']
                        }
                    })
                elif item.get('type') == 'audio_url':
                    parts.append({
                        "fileData": {
                            "mimeType": "audio/mp3",
                            "fileUri": item['audio_url']['url']
                        }
                    })
        else:
            parts.append({"text": msg['content']})
        
        native_contents.append({"role": role, "parts": parts})

    # 2. Build native protocol payload
    payload = {
        "model": model,
        "dashscope_extend_params": {
            "using_native_protocol": True,
            'X-DashScope-DataInspection': '{"input":"disable","output":"disable"}'
        },
        "generationConfig": {
            "maxOutputTokens": 1024*32,
            "thinkingConfig": {
                "includeThoughts": True,
                # "thinkingBudget": -1,
                "thinkingLevel": "high"
            }
        },
        # "system_instruction": {"parts": [{"text":SYSTEM_PROMPT,}]},
        "stream": True,
        "contents": native_contents,
    }

    for k in range(retries):
        try:
            stream_output = ""
            reasoning_stream_output = ""
            
            resp = requests.post(BASE_URL, headers=HEADERS,
                                 json=payload, timeout=300, stream=True)
            resp.raise_for_status()

            for line in resp.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        try:
                            data = json.loads(decoded_line[6:])
                            for candidate in data.get("candidates", []):
                                for part in candidate.get("content", {}).get("parts", []):
                                    if part.get("thought"):
                                        reasoning_stream_output += part.get("text", "")
                                    else:
                                        stream_output += part.get("text", "")
                        except json.JSONDecodeError:
                            continue

            # Simulate original return structure to keep main function logic unchanged
            return {
                "choices": [
                    {
                        "message": {
                            "content": stream_output,
                            "reasoning_content": reasoning_stream_output # Additionally save thinking process
                        }
                    }
                ],
                "usage": {} # Native protocol streaming returns different usage structure; left empty or parse as needed
            }

        except Exception as e:
            if k == retries - 1:
                raise
            print(f"[retry {k+1}/{retries}] error: {e}, wait {backoff}s")
            time.sleep(backoff)


# ========= Video Analysis Tools (unchanged) =========
def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get video information using ffprobe
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=nw=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip().split('=')[1])
    
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "stream=r_frame_rate",
        "-select_streams", "v:0", "-of", "default=nw=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    fps_str = result.stdout.strip().split('=')[1]
    if '/' in fps_str:
        num, den = map(int, fps_str.split('/'))
        fps = num / den if den != 0 else 30.0
    else:
        fps = float(fps_str)
    
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "stream=codec_type",
        "-of", "default=nw=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    has_audio = "audio" in result.stdout
    
    return {
        "duration": duration,
        "fps": fps,
        "has_audio": has_audio
    }

# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Single-sample baseline inference with Gemini")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Path to video file. Falls back to VIDEO_PREFIX + SAMPLE_VIDEO_PATH env vars")
    parser.add_argument("--question", type=str,
                        default="Between which frames is the woman continues her makeup tutorial, meticulously applying teal eyeshadow and blending it out, her voice-over explaining each step as she works visible in the video?")
    parser.add_argument("--question_type", type=str, default="TR", choices=["MCQ", "TR", "FF", "NUM", "SIZE"])
    parser.add_argument("--answer", type=str, default="[105.105, 149.415]")
    parser.add_argument("--options", type=str, default=None, help="MCQ options separated by \\n")
    args = parser.parse_args()

    if args.video_path:
        SAMPLE_URL = args.video_path
    else:
        VIDEO_PREFIX = os.getenv("VIDEO_PREFIX", "")
        video_path = os.getenv("SAMPLE_VIDEO_PATH", "/path/to/video.mp4")
        SAMPLE_URL = os.path.join(VIDEO_PREFIX, video_path)

    question_type = args.question_type
    question = args.question
    answer = args.answer
    options = [opt.strip() for opt in args.options.replace("\\n", "\n").split("\n") if opt.strip()] if args.options else None

    video_info = get_video_info(SAMPLE_URL)

    dur_str = f"{video_info['duration']:.3f}"
    fps_str = f"{video_info['fps']:.3f}"
    has_audio = video_info['has_audio']

    meta_text = (
        "Video META:\n"
        f"- duration_seconds: {dur_str}\n"
        f"- fps: {fps_str}\n"
        f"- has_audio: {has_audio}\n\n"
    )

    if question_type=="MCQ":
        opt_txt = "\nOptions:\n" + "\n".join(options)
        opt_txt += "\nWhen answering, set action.content to ONE uppercase letter (A, B, C …)."
        question = meta_text + "Question: " + question + opt_txt  
    
    if question_type=="TR":
        guide  = (
            "\nWhen answering, set action.content to a JSON array of timestamp "
            "pairs such as [[103.1, 120.7]] (unit: seconds)."
        )
        question = meta_text + "Question: " + question + guide          


    UPLOAD_MODE = os.getenv("UPLOAD_MODE", "oss" if oss_reader.enabled else "inline")  # "inline" | "oss"

    if UPLOAD_MODE == "inline":
        with open(SAMPLE_URL, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        video_part = {
            "type": "inline_video",
            "mime_type": "video/mp4",
            "data": video_b64
        }
    elif UPLOAD_MODE == "oss":
        oss_uri, signed_url = upload_and_sign(SAMPLE_URL)
        print("signed url:", signed_url)
        video_part = {
            "type": "video_url",
            "video_url": {"url": signed_url}
        }

    # ------------------------------ Assemble prompt ------------------------------
    messages = [
        {
            "role": "user",
            "content": [
                video_part,
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]

    # ------------------------------ Call API ------------------------------
    print(f"Prompt: {messages}")
    data = call_gemini(messages)
    # Logic here is fully consistent with the original script
    answer_content = data["choices"][0]["message"]["content"]

    # ------------------------------ Output ------------------------------
    print("\n==== Raw response (Simulated) ====\n")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    print("\n==== Answer field ====\n")
    print(answer_content)


if __name__ == "__main__":
    main()
