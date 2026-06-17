from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from io import BytesIO

import librosa
import numpy as np
try:
    import oss2
    oss2.defaults.connection_pool_size = 80
    HAS_OSS2 = True
except ImportError:
    oss2 = None
    HAS_OSS2 = False

from smart_open import open as open_
import sys

if HAS_OSS2:
    oss_logger = logging.getLogger("oss2")
    oss_logger.setLevel(logging.CRITICAL)

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


class NoNeedRetryError(Exception):
    pass


def retry_on_error(retry_times=3, raise_on_failure=True, default_return=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for retry in range(retry_times):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    stack = traceback.extract_tb(e.__traceback__)
                    error_file, error_line, _, _ = stack[-1]
                    print(
                        f"[{retry=}/{retry_times}] {kwargs.get('oss_path', args[-1])} {type(e).__name__}({e}) at {error_file}:{error_line}"
                    )
                    if (
                        isinstance(e, NoNeedRetryError)
                        or "The specified key does not exist" in str(e)
                        or "The requested range cannot be satisfied" in str(e)
                    ):
                        break
                    elif "Please reduce your request rate" in str(e):
                        time.sleep(np.random.uniform(1, 1.25))
                    else:
                        time.sleep(np.random.uniform(0.6, 0.75))
            if raise_on_failure:
                raise RuntimeError(
                    f"Failed after {retry_times} retries: {kwargs.get('oss_path', args[-1])}"
                )
            else:
                return default_return

        return wrapper

    return decorator


# def get_bucket(access_key_id, access_key_secret, endpoint, bucket_name):
#     auth = oss2.Auth(access_key_id, access_key_secret)
#     bucket = oss2.Bucket(auth, endpoint, bucket_name)
#     return bucket

def get_bucket(
    access_key_id,
    access_key_secret,
    endpoint,
    bucket_name
):
    if not HAS_OSS2:
        raise RuntimeError("oss2 is required for OSS access. Install with: pip install oss2")
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    return bucket

class SingletonType(type):
    _instance_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with SingletonType._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instance


class OssReader(metaclass=SingletonType):
    def __init__(self):
        if not HAS_OSS2:
            self.config = {}
            self.enabled = False
            self.bucket = {}
            self.s3_bucket = {}
            logging.warning("oss2 not installed. OssReader disabled.")
            return
        if OSS_CONFIG_PATH and os.path.exists(OSS_CONFIG_PATH):
            with open(OSS_CONFIG_PATH) as fin:
                self.config: dict = json.load(fin)
            self.enabled = True
        else:
            self.config = {}
            self.enabled = False
            logging.warning("OssReader initialized in disabled mode (no config).")
        self.bucket = {
            bucket_name: get_bucket(**bucket_meta, bucket_name=bucket_name)
            for bucket_name, bucket_meta in self.config.items()
        }
        self.s3_bucket = {}

    @retry_on_error()
    def read(self, oss_path: str):
        bucket_name, key_path, byte_range = self._parse(oss_path, parse_range=True)
        return (
            self.bucket[bucket_name].get_object(key_path, byte_range=byte_range).read()
        )

    @retry_on_error(raise_on_failure=False, default_return=False)
    def check_object_exists(self, oss_path: str):
        bucket_name, key_path = self._parse(oss_path)
        return self.bucket[bucket_name].object_exists(key_path)

    def get_public_url(self, oss_path: str):
        bucket_name, key_path = self._parse(oss_path)
        return self.bucket[bucket_name].sign_url("GET", key_path, 86400, slash_safe=True)

    def _parse(self, oss_path: str, parse_range: bool = False):
        assert oss_path.startswith("oss://"), f"{oss_path} is not a valid oss path."
        bucket_name = oss_path[len("oss://") :].split("/", 1)[0]
        key_path = oss_path[len("oss://") + len(bucket_name) + 1 :]

        if not parse_range:
            return bucket_name, key_path

        byte_range = None
        if "#" in key_path and all(
            elem in key_path.split("#")[-1] for elem in ["offset=", "size=", "&"]
        ):
            offset_size = key_path.split("#")[-1]
            key_path = "#".join(key_path.split("#")[:-1])
            start = int(offset_size.split("&")[0][len("offset=") :])
            size = int(offset_size.split("&")[-1][len("size=") :])
            byte_range = (start, start + size - 1)
        elif "#" in key_path and "-" in key_path.split("#")[-1]:
            span_range = key_path.split("#")[-1]
            key_path = "#".join(key_path.split("#")[:-1])
            start = int(span_range.split("-")[0])
            end = int(span_range.split("-")[1])
            byte_range = (start, end - 1)
        return bucket_name, key_path, byte_range

    @retry_on_error()
    def read_audio_bin(self, oss_path: str, *, audio_start=0, audio_end=None, sr=16000):
        # mult 2 : signed 16bit = 2 bytes
        offset = int(audio_start * sr * 2)  # 偶数
        if offset % 2 != 0:
            offset += 1
        end = int(audio_end * sr * 2) if audio_end is not None else None
        if end:
            if end % 2 == 0:
                end -= 1
        if end and offset >= end:
            raise NoNeedRetryError(f"offset or end Error {offset=} {end=}")

        bucket_name, key_path = self._parse(oss_path)
        data = (
            self.bucket[bucket_name]
            .get_object(
                key_path,
                byte_range=(offset, end),
                headers={"x-oss-range-behavior": "standard"},
            )
            .read()
        )
        if end and end - offset + 1 < len(data):
            raise NoNeedRetryError(
                f"BinFile byte_range Error {len(data)=}, {end=}, {end-offset=}"
            )
        return np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0

    def read_wav(self, oss_path: str, sample_rate: float, return_type=np.float32):
        # Load the wav from both the oss:// or local path. If the wav is corrupted or not exist, return 'None' value.
        # if return None, the oss path is break
        if "oss://" in oss_path:
            bytes_wav = self.read(oss_path)
            if bytes_wav is None:
                return None, None
            try:
                wav, sr = librosa.load(BytesIO(bytes_wav), sr=sample_rate)
            except:
                raise ValueError(f"{oss_path} is corrupt")
        else:
            try:
                # for local path
                wav, sr = librosa.load(oss_path, sr=sample_rate)
            except Exception as e:
                raise ValueError("local file read error", e, oss_path)
        if return_type is np.float32:
            # The librosa default type is np.float32 (normalized in [-1, 1])
            wav = wav
        elif return_type is np.int16:
            # Convert to the original Int dtype
            wav = (wav * 32768.0).astype(np.int16)
        else:
            raise ValueError("not supported yet")
        return wav, sr

    def get_s3_bucket(self, bucket_name: str):
        import boto3
        import botocore

        bucket_meta = self.config[bucket_name]
        s3 = boto3.client(
            "s3",
            aws_access_key_id=bucket_meta["access_key_id"],
            aws_secret_access_key=bucket_meta["access_key_secret"],
            endpoint_url=f"https://{bucket_meta['endpoint']}",
            config=botocore.config.Config(
                s3={"addressing_style": "virtual", "signature_version": "s3v4"}
            ),
        )
        self.s3_bucket[bucket_name] = s3
        return s3

    def open(self, path: str, mode: str = "r"):
        if path.startswith("oss://"):
            bucket_name, key_path = self._parse(path)
            if bucket_name not in self.s3_bucket:
                self.get_s3_bucket(bucket_name)
            return open_(
                f"s3://{bucket_name}/{key_path}",
                mode,
                transport_params={"client": self.s3_bucket[bucket_name]},
            )
        else:
            return open(path, mode)

    @retry_on_error()
    def librosa_load(self, path: str, *, sample_rate: float, audio_start=None, audio_end=None, return_type=np.float32):
        addition_kwargs = {}
        if audio_start is not None:
            addition_kwargs = dict(offset=audio_start, duration=audio_end - audio_start)
        wav, sr = librosa.load(self.open(path, "rb"), sr=sample_rate, **addition_kwargs)
        if return_type is np.float32:
            wav = wav  # The librosa default type is np.float32 (normalized in [-1, 1])
        elif return_type is np.int16:
            wav = (wav * 32768.0).astype(np.int16)  # Convert to the original Int dtype
        else:
            raise NoNeedRetryError("not supported yet")
        return wav, sr
