import base64
import functools
import json
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from subprocess import CalledProcessError, run

import audioread
import av
import librosa
import numpy as np
import torch

try:
    import oss2
    from oss2.credentials import EnvironmentVariableCredentialsProvider
    HAS_OSS2 = True
except ImportError:
    oss2 = None
    EnvironmentVariableCredentialsProvider = None
    HAS_OSS2 = False

from .oss_reader import OssReader as AudioOSSReader
import sys

logger = logging.getLogger(__name__)


OSS_CONFIG_PATH = next(
    (
        p
        for p in [os.getenv("OSS_CONFIG_PATH"), os.path.expanduser("~/.oss_config.json")]
        if p and os.path.isfile(p)
    ),
    None,
)
if OSS_CONFIG_PATH is None:
    logging.warning("OSS config file not found. OSS features will be disabled. "
                    "Set OSS_CONFIG_PATH or create ~/.oss_config.json to enable.")



SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_LEN_SECONDS = 900.0

def oss2http_url(oss_url: str, expire_second: int = 86400) -> str:
    if not HAS_OSS2:
        raise RuntimeError("oss2 is required for OSS URL conversion. Install with: pip install oss2")
    assert oss_url.startswith("oss://")
    bucket_name, object_key = oss_url[6:].split("/", 1)
    endpoint = os.environ.get("DEFAULT_OSS_ENDPOINT", "oss-cn-shanghai.aliyuncs.com")
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    return bucket.sign_url("GET", object_key, expire_second, slash_safe=True)

def get_bucket(access_key_id, access_key_secret, endpoint, bucket_name):
    if not HAS_OSS2:
        raise RuntimeError("oss2 is required for OSS access. Install with: pip install oss2")
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    return bucket

def load_audio(file: str, sr: int = SAMPLE_RATE, audio_start: float = None, audio_end: float = None):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    if audio_start is not None:
        duration = audio_end-audio_start
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-ss", str(audio_start),
            "-i", file,
            "-t", str(duration),
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]
    else:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def _get_env_max_audio_len() -> float:
    raw = os.environ.get("MAX_AUDIO_LEN", "")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_MAX_AUDIO_LEN_SECONDS
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid MAX_AUDIO_LEN=%r, fallback to default %.1fs",
            raw,
            DEFAULT_MAX_AUDIO_LEN_SECONDS,
        )
        return DEFAULT_MAX_AUDIO_LEN_SECONDS


def _resolve_audio_window(ele: dict, media_type: str, max_audio_len: float) -> tuple[float | None, float | None]:
    start_key = f"{media_type}_start"
    end_key = f"{media_type}_end"
    start = float(ele[start_key]) if start_key in ele else None
    end = float(ele[end_key]) if end_key in ele else None
    if max_audio_len <= 0:
        return start, end

    if start is None:
        start = 0.0
    capped_end = start + max_audio_len
    if end is None or capped_end < end:
        end = capped_end
    return start, end


def _truncate_waveform(audio: np.ndarray, sr: int, max_audio_len: float) -> np.ndarray:
    if max_audio_len <= 0:
        return audio
    max_samples = int(max_audio_len * sr)
    if max_samples <= 0:
        return audio[:0]
    return audio[:max_samples]


class OssReader:
    def __init__(self):
        self.bucket = {}
        if OSS_CONFIG_PATH and os.path.isfile(OSS_CONFIG_PATH):
            with open(OSS_CONFIG_PATH) as fin:
                self.config = json.load(fin)
            self.enabled = True
        else:
            self.config = {}
            self.enabled = False

    def read(self, oss_path):
        assert oss_path.startswith('oss://'), f'{oss_path} is not a valid oss path.'
        bucket_name = oss_path[len('oss://') :].split('/', 1)[0]
        if bucket_name not in self.bucket:
            bucket_meta = self.config[bucket_name]
            auth = oss2.Auth(bucket_meta['access_key_id'], bucket_meta['access_key_secret'])
            self.bucket[bucket_name] = oss2.Bucket(
                auth, bucket_name=bucket_name, endpoint=bucket_meta['endpoint']
            )
        key_path = oss_path[len('oss://') + len(bucket_name) + 1 :]

        key_range = key_path.split('#')
        if len(key_range) > 1:
            key_path, byte_range = key_range[0], key_range[1]
            byte_range = list(map(int, byte_range.split('-')))
            byte_range[1] -= 1
        else:
            byte_range = None

        retry = 10
        for i in range(retry):
            try:
                data = self.bucket[bucket_name].get_object(key_path, byte_range=byte_range).read()
                break
            except Exception as e:
                print('retry=', retry, e, oss_path)
                time.sleep(0.1)
                data = None

        if data is None:
            print('OSS Read File Error {}'.format(oss_path))
        return data
    
    def check_object_exists(self, oss_path: str):
        bucket_name, key_path = self._parse(oss_path)
        return self.bucket[bucket_name].object_exists(key_path)

    def get_public_url(self, oss_path):
        assert oss_path.startswith('oss://'), f'{oss_path} is not a valid oss path.'
        bucket_name = oss_path[len('oss://') :].split('/', 1)[0]
        if bucket_name not in self.bucket:
            bucket_meta = self.config[bucket_name]
            auth = oss2.Auth(bucket_meta['access_key_id'], bucket_meta['access_key_secret'])
            self.bucket[bucket_name] = oss2.Bucket(
                auth, bucket_name=bucket_name, endpoint=bucket_meta['endpoint']
            )
        key_path = oss_path[len('oss://') + len(bucket_name) + 1 :]
        if not self.bucket[bucket_name].object_exists(key_path):
            raise ValueError(f"{key_path} not exists")
        url = self.bucket[bucket_name].sign_url('GET', key_path, 86400, slash_safe=True)
        return url

oss_reader = OssReader()

audio_oss_reader = AudioOSSReader()

def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def get_raw_audio(audio_info: dict):
    start_time = time.time()
    if isinstance(audio_info, dict):
        if 'audio' in audio_info:
            type = 'audio'
        elif 'video' in audio_info:
            type = 'video'
        elif 'audio_to_tts' in audio_info:
            type = 'audio'
        else:
            raise NotImplementedError()

        if 'audio_to_tts' in audio_info:
            audio_path = audio_info['audio_to_tts']
        else:
            audio_path = audio_info[type]
    else:
        audio_path = audio_info
    
    audio_bin_path = audio_path + ".ar16k.bin"
    if audio_oss_reader.check_object_exists(audio_bin_path):  # 1. 先读bin文件
        try:
            if isinstance(audio_info, dict) and f'{type}_start' in audio_info:
                audio = audio_oss_reader.read_audio_bin(audio_bin_path,
                                                        audio_start=float(audio_info[f'{type}_start']),
                                                        audio_end=float(audio_info[f'{type}_end']))
            else:
                audio = audio_oss_reader.read_audio_bin(audio_bin_path)
        except Exception as e:
            try:
                if audio_path.startswith("oss"):
                    public_url = audio_oss_reader.get_public_url(audio_path)
                else:
                    public_url = audio_path
                if isinstance(audio_info, dict) and f'{type}_start' in audio_info:
                    audio = load_audio(public_url, sr=16000,
                                                    audio_start=float(audio_info[f'{type}_start']),
                                                    audio_end=float(audio_info[f'{type}_end']))
                else:
                    audio = load_audio(public_url, sr=16000)
            except Exception as e:
                if 'Please reduce your request rate' in str(e):
                    time.sleep(np.random.uniform(1, 1.25))
                else:
                    time.sleep(np.random.uniform(0.6, 0.75))
                raise ValueError(F"FFMPEG ERROR when read {public_url} {e}")
    elif not audio_oss_reader.check_object_exists(audio_path):  # 2. 检测是否有audio文件
        raise ValueError(F"AUDIO NOT EXIST ERROR when read {audio_path}")
    elif Path(audio_path).suffix.lower() not in {'.wav', '.flac'}:  # 2. librosa不支持格式，走ffmpeg
        try:
            if audio_path.startswith("oss"):
                public_url = audio_oss_reader.get_public_url(audio_path)
            else:
                public_url = audio_path
            if isinstance(audio_info, dict) and f'{type}_start' in audio_info:
                audio = load_audio(public_url, sr=16000, audio_start=float(audio_info[f'{type}_start']),
                                                audio_end=float(audio_info[f'{type}_end']))
            else:
                audio = load_audio(public_url, sr=16000)
            end_time = time.time()
            if end_time - start_time > 2:
                print(
                    f"Read audio {public_url} costs {end_time - start_time} seconds on Rank {torch.distributed.get_rank()}")
        except Exception as e:
            if 'Please reduce your request rate' in str(e):
                time.sleep(np.random.uniform(1, 1.25))
            else:
                time.sleep(np.random.uniform(0.6, 0.75))
            raise ValueError(F"FFMPEG ERROR when read {public_url} {e}")
    else:
        try:
            if isinstance(audio_info, dict) and f'{type}_start' in audio_info:
                audio, sr = audio_oss_reader.librosa_load(audio_path, sample_rate=16000,
                                                            audio_start=float(audio_info[f'{type}_start']),
                                                            audio_end=float(audio_info[f'{type}_end']))
            else:
                audio, sr = audio_oss_reader.librosa_load(audio_path, sample_rate=16000)
        except Exception as e:
            if 'Please reduce your request rate' in str(e):
                time.sleep(np.random.uniform(1, 1.25))
            else:
                time.sleep(np.random.uniform(0.6, 0.75))
            print(f"Librosa don't support {Path(audio_path).suffix.lower()} {e}")
            try:
                if audio_path.startswith("oss"):
                    public_url = audio_oss_reader.get_public_url(audio_path)
                else:
                    public_url = audio_path
                if isinstance(audio_info, dict) and f'{type}_start' in audio_info:
                    audio = load_audio(public_url, sr=16000,
                                                    audio_start=float(audio_info[f'{type}_start']),
                                                    audio_end=float(audio_info[f'{type}_end']))
                else:
                    audio = load_audio(public_url, sr=16000)
            except Exception as e:
                if 'Please reduce your request rate' in str(e):
                    time.sleep(np.random.uniform(1, 1.25))
                else:
                    time.sleep(np.random.uniform(0.6, 0.75))
                raise ValueError(F"FFMPEG ERROR when read {public_url} {e}")
    return audio

def retry(max_retry=10, delay=1.0):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            for attempt in range(1, max_retry + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt >= max_retry:
                        raise
                    logger.warning(
                        f"[retry] {fn.__name__} failed {attempt}/{max_retry}: {e}, "
                        f"sleep {delay}s")
                    time.sleep(delay)          # 始终相同的等待
        return wrapped
    return deco

@retry()
def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    audios = []
    max_audio_len = _get_env_max_audio_len()
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele:
                        path = ele["audio"]
                        audio_start, audio_end = _resolve_audio_window(ele, "audio", max_audio_len)
                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(_truncate_waveform(path, SAMPLE_RATE, max_audio_len))
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = base64.b64decode(base64_data)
                            audio = librosa.load(BytesIO(data), sr=16000)[0]
                            audios.append(_truncate_waveform(audio, SAMPLE_RATE, max_audio_len))
                        elif path.startswith("http://") or path.startswith("https://"):
                            # audios.append(librosa.load(audioread.ffdec.FFmpegAudioFile(path), sr=16000)[0])
                            # print("start load audio")
                            audio = load_audio(path, sr=16000, audio_start=audio_start, audio_end=audio_end)
                            audios.append(audio)
                        elif path.startswith("file://"):
                            audio = load_audio(
                                path[len("file://") :],
                                sr=16000,
                                audio_start=audio_start,
                                audio_end=audio_end,
                            )
                            audios.append(audio)
                        elif path.startswith("oss://"):
                            public_url = oss_reader.get_public_url(path)
                            audio = load_audio(public_url, sr=16000, audio_start=audio_start, audio_end=audio_end)
                            audios.append(audio)
                        else:
                            audio = load_audio(path, sr=16000, audio_start=audio_start, audio_end=audio_end)
                            audios.append(audio)
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                if use_audio_in_video and ele["type"] == "video":
                    if "video" in ele:
                        path = ele["video"]
                        audio_start, audio_end = _resolve_audio_window(ele, "video", max_audio_len)
                        if path.startswith("http://") or path.startswith("https://"):
                            assert _check_if_video_has_audio(
                                path
                            ), "Video must has audio track when use_audio_in_video=True"
                            # audios.append(librosa.load(audioread.ffdec.FFmpegAudioFile(path), sr=16000)[0])
                            print("start load audio in video")
                            audio = load_audio(path, sr=16000, audio_start=audio_start, audio_end=audio_end)
                            audios.append(audio)
                        elif path.startswith("file://"):
                            assert _check_if_video_has_audio(
                                path
                            ), "Video must has audio track when use_audio_in_video=True"
                            audio = load_audio(
                                path[len("file://") :],
                                sr=16000,
                                audio_start=audio_start,
                                audio_end=audio_end,
                            )
                            audios.append(audio)
                        elif path.startswith("oss://"):
                            audio_info = dict(ele)
                            if audio_start is not None:
                                audio_info["video_start"] = audio_start
                            if audio_end is not None:
                                audio_info["video_end"] = audio_end
                            audio = get_raw_audio(audio_info)
                            audios.append(audio)
                        else:
                            audio = load_audio(path, sr=16000, audio_start=audio_start, audio_end=audio_end)
                            audios.append(audio)
                    else:
                        raise ValueError("Unknown video {}".format(ele))
    if len(audios) == 0:
        audios = None
    return audios
