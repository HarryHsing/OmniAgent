# from __future__ import annotations

# import base64
# import json
# import logging
# import math
# import os
# import sys
# import time
# import warnings
# from concurrent.futures import ThreadPoolExecutor
# from functools import lru_cache, partial
# from io import BytesIO
# from typing import Optional

# import numpy as np
# import oss2
# import requests
# import torch
# import torchvision
# from oss2.credentials import EnvironmentVariableCredentialsProvider
# from packaging import version
# from PIL import Image
# from torchvision import io, transforms
# from torchvision.transforms import InterpolationMode

# logger = logging.getLogger(__name__)

# # IMAGE_FACTOR = 28
# # MIN_PIXELS = 4 * 28 * 28
# # MAX_PIXELS = 16384 * 28 * 28
# # MAX_RATIO = 200

# # VIDEO_MIN_PIXELS = 128 * 28 * 28
# # VIDEO_MAX_PIXELS = 768 * 28 * 28
# # FRAME_FACTOR = 2
# # FPS = 2.0
# # FPS_MIN_FRAMES = 4
# # FPS_MAX_FRAMES = 64

# IMAGE_FACTOR = 28
# MIN_PIXELS = 4 * 28 * 28
# MAX_PIXELS = 128 * 28 * 28
# MAX_RATIO = 200

# VIDEO_MIN_PIXELS = 128 * 28 * 28
# VIDEO_MAX_PIXELS = 128 * 28 * 28
# FRAME_FACTOR = 2
# FPS = 2.0
# FPS_MIN_FRAMES = 2
# FPS_MAX_FRAMES = 32

# OSS_CONFIG_PATH = next((p for p in [os.getenv('OSS_CONFIG_PATH'), os.path.expanduser('~/.oss_config.json')]
#                         if p and os.path.isfile(p)), None)
# if OSS_CONFIG_PATH is None: sys.exit('OSS config file not found. Set OSS_CONFIG_PATH or create ~/.oss_config.json')
# USE_SHANGHAI_OSS = os.environ.get("USE_SHANGHAI_OSS", False)

# # Set the maximum number of video token inputs.
# # Here, 128K represents the maximum number of input tokens for the VLLM model.
# # Remember to adjust it according to your own configuration.
# VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
# logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


# def oss2http_url(oss_url: str, expire_second: int = 86400) -> str:
#     assert oss_url.startswith("oss://")
#     bucket_name, object_key = oss_url[6:].split("/", 1)
#     endpoint = os.environ.get("DEFAULT_OSS_ENDPOINT", "oss-cn-shanghai.aliyuncs.com")
#     auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
#     bucket = oss2.Bucket(auth, endpoint, bucket_name)
#     return bucket.sign_url("GET", object_key, expire_second, slash_safe=True)


# def round_by_factor(number: int, factor: int) -> int:
#     """Returns the closest integer to 'number' that is divisible by 'factor'."""
#     return round(number / factor) * factor


# def ceil_by_factor(number: int, factor: int) -> int:
#     """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
#     return math.ceil(number / factor) * factor


# def floor_by_factor(number: int, factor: int) -> int:
#     """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
#     return math.floor(number / factor) * factor


# def smart_resize(
#     height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
# ) -> tuple[int, int]:
#     """
#     Rescales the image so that the following conditions are met:

#     1. Both dimensions (height and width) are divisible by 'factor'.

#     2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

#     3. The aspect ratio of the image is maintained as closely as possible.
#     """
#     if max(height, width) / min(height, width) > MAX_RATIO:
#         raise ValueError(
#             f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
#         )
#     h_bar = max(factor, round_by_factor(height, factor))
#     w_bar = max(factor, round_by_factor(width, factor))
#     if h_bar * w_bar > max_pixels:
#         beta = math.sqrt((height * width) / max_pixels)
#         h_bar = floor_by_factor(height / beta, factor)
#         w_bar = floor_by_factor(width / beta, factor)
#     elif h_bar * w_bar < min_pixels:
#         beta = math.sqrt(min_pixels / (height * width))
#         h_bar = ceil_by_factor(height * beta, factor)
#         w_bar = ceil_by_factor(width * beta, factor)
#     return h_bar, w_bar


# def to_rgb(pil_image: Image.Image) -> Image.Image:
#       if pil_image.mode == 'RGBA':
#           white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
#           white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
#           return white_background
#       else:
#           return pil_image.convert("RGB")


# def download_file(url: str, max_retries: int = 10) -> Optional[Image.Image]:
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
#     }
#     for attempt in range(max_retries):
#         try:
#             response = requests.get(url, headers=headers, timeout=5, stream=True)
#             response.raise_for_status()

#             content_type = response.headers.get('content-type', '')
#             if not content_type.startswith('image/'):
#                 pass

#             image_data = BytesIO(response.content)
#             image = Image.open(image_data)
#             image.verify()
#             if image.width > 0 and image.height > 0:
#                 image_data.seek(0)
#                 return Image.open(image_data)
#         except requests.exceptions.RequestException as e:
#             pass
#         except IOError as e:
#             pass
#     return None


# def get_bucket(access_key_id, access_key_secret, endpoint, bucket_name):
#     auth = oss2.Auth(access_key_id, access_key_secret)
#     bucket = oss2.Bucket(auth, endpoint, bucket_name)
#     return bucket


# class OssReader:
#     def __init__(self):
#         self.bucket = {}
#         with open(OSS_CONFIG_PATH) as fin:
#             self.config = json.load(fin)

#     def read(self, oss_path):
#         assert oss_path.startswith('oss://'), f'{oss_path} is not a valid oss path.'
#         bucket_name = oss_path[len('oss://') :].split('/', 1)[0]
#         if bucket_name not in self.bucket:
#             bucket_meta = self.config[bucket_name]
#             auth = oss2.Auth(bucket_meta['access_key_id'], bucket_meta['access_key_secret'])
#             self.bucket[bucket_name] = oss2.Bucket(
#                 auth, bucket_name=bucket_name, endpoint=bucket_meta['endpoint']
#             )
#         key_path = oss_path[len('oss://') + len(bucket_name) + 1 :]

#         key_range = key_path.split('#')
#         if len(key_range) > 1:
#             key_path, byte_range = key_range[0], key_range[1]
#             byte_range = list(map(int, byte_range.split('-')))
#             byte_range[1] -= 1
#         else:
#             byte_range = None

#         retry = 10
#         for i in range(retry):
#             try:
#                 data = self.bucket[bucket_name].get_object(key_path, byte_range=byte_range).read()
#                 break
#             except Exception as e:
#                 print('retry=', retry, e, oss_path)
#                 time.sleep(0.1)
#                 data = None

#         if data is None:
#             print('OSS Read File Error {}'.format(oss_path))
#         return data

#     def get_public_url(self, oss_path):
#         assert oss_path.startswith('oss://'), f'{oss_path} is not a valid oss path.'
#         bucket_name = oss_path[len('oss://') :].split('/', 1)[0]
#         if bucket_name not in self.bucket:
#             bucket_meta = self.config[bucket_name]
#             auth = oss2.Auth(bucket_meta['access_key_id'], bucket_meta['access_key_secret'])
#             self.bucket[bucket_name] = oss2.Bucket(
#                 auth, bucket_name=bucket_name, endpoint=bucket_meta['endpoint']
#             )
#         key_path = oss_path[len('oss://') + len(bucket_name) + 1 :]
#         if not self.bucket[bucket_name].object_exists(key_path):
#             raise ValueError(f"{key_path} not exists in bucket {bucket_name}")
#         url = self.bucket[bucket_name].sign_url('GET', key_path, 86400, slash_safe=True)
#         return url


# oss_reader = OssReader()


# def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
#     if "image" in ele:
#         image = ele["image"]
#     else:
#         image = ele["image_url"]
#     image_obj = None
#     if isinstance(image, Image.Image):
#         image_obj = image
#     elif image.startswith("oss://"):
#         image_obj = Image.open(BytesIO(oss_reader.read(image)))
#         # image_obj = download_file(oss2http_url(image))
#     elif image.startswith("http://") or image.startswith("https://"):
#         image_obj = download_file(image)
#     elif image.startswith("file://"):
#         image_obj = Image.open(image[7:])
#     elif image.startswith("data:image"):
#         if "base64," in image:
#             _, base64_data = image.split("base64,", 1)
#             data = base64.b64decode(base64_data)
#             image_obj = Image.open(BytesIO(data))
#     else:
#         image_obj = Image.open(image)
#     if image_obj is None:
#         raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
#     image = to_rgb(image_obj)
#     ## resize
#     if "resized_height" in ele and "resized_width" in ele:
#         resized_height, resized_width = smart_resize(
#             ele["resized_height"],
#             ele["resized_width"],
#             factor=size_factor,
#         )
#     else:
#         width, height = image.size
#         min_pixels = ele.get("min_pixels", MIN_PIXELS)
#         max_pixels = ele.get("max_pixels", MAX_PIXELS)
#         resized_height, resized_width = smart_resize(
#             height,
#             width,
#             factor=size_factor,
#             min_pixels=min_pixels,
#             max_pixels=max_pixels,
#         )
#     image = image.resize((resized_width, resized_height))

#     return image


# def smart_nframes(
#     ele: dict,
#     total_frames: int,
#     video_fps: int | float,
# ) -> int:
#     """calculate the number of frames for video used for model inputs.

#     Args:
#         ele (dict): a dict contains the configuration of video.
#             support either `fps` or `nframes`:
#                 - nframes: the number of frames to extract for model inputs.
#                 - fps: the fps to extract frames for model inputs.
#                     - min_frames: the minimum number of frames of the video, only used when fps is provided.
#                     - max_frames: the maximum number of frames of the video, only used when fps is provided.
#         total_frames (int): the original total number of frames of the video.
#         video_fps (int | float): the original fps of the video.

#     Raises:
#         ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

#     Returns:
#         int: the number of frames for video used for model inputs.
#     """
#     assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
#     if "nframes" in ele:
#         nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
#     else:
#         fps = ele.get("fps", FPS)
#         min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
#         max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
#         nframes = total_frames / video_fps * fps
#         if nframes > total_frames:
#             logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
#         nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
#         nframes = floor_by_factor(nframes, FRAME_FACTOR)
#     if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
#         raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
#     return nframes


# def calculate_video_frame_range(
#     ele: dict,
#     total_frames: int,
#     video_fps: float,
# ) -> tuple[int, int, int]:
#     """
#     Calculate the start and end frame indices based on the given time range.

#     Args:
#         ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
#         total_frames (int): Total number of frames in the video.
#         video_fps (float): Frames per second of the video.

#     Returns:
#         tuple: A tuple containing (start_frame, end_frame, frame_count).

#     Raises:
#         ValueError: If input parameters are invalid or the time range is inconsistent.
#     """
#     # Validate essential parameters
#     if video_fps <= 0:
#         raise ValueError("video_fps must be a positive number")
#     if total_frames <= 0:
#         raise ValueError("total_frames must be a positive integer")

#     # Get start and end time in seconds
#     video_start = ele.get("video_start", None)
#     video_end = ele.get("video_end", None)
#     if video_start is None and video_end is None:
#         return 0, total_frames - 1, total_frames

#     max_duration = total_frames / video_fps
#     # Process start frame
#     if video_start is not None:
#         video_start_clamped = max(0.0, min(video_start, max_duration))
#         start_frame = math.ceil(video_start_clamped * video_fps)
#     else:
#         start_frame = 0
#     # Process end frame
#     if video_end is not None:
#         video_end_clamped = max(0.0, min(video_end, max_duration))
#         end_frame = math.floor(video_end_clamped * video_fps)
#         end_frame = min(end_frame, total_frames - 1)
#     else:
#         end_frame = total_frames - 1

#     # Validate frame order
#     if start_frame >= end_frame:
#         raise ValueError(
#             f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
#             f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
#             f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
#         )

#     logger.info(f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}")
#     return start_frame, end_frame, end_frame - start_frame + 1


# def _read_video_torchvision(
#     ele: dict,
# ) -> (torch.Tensor, float):
#     """read video using torchvision.io.read_video

#     Args:
#         ele (dict): a dict contains the configuration of video.
#         support keys:
#             - video: the path of video. support "file://", "oss://", "http://", "https://" and local path.
#             - video_start: the start time of video.
#             - video_end: the end time of video.
#     Returns:
#         torch.Tensor: the video tensor with shape (T, C, H, W).
#     """
#     video_path = ele["video"]
#     if version.parse(torchvision.__version__) < version.parse("0.19.0"):
#         if "oss://" in video_path:
#             # video_path = oss2http_url(video_path)
#             video_path = oss_reader.get_public_url(video_path)
#         if "http://" in video_path or "https://" in video_path:
#             warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
#         if "file://" in video_path:
#             video_path = video_path[7:]
#     st = time.time()
#     video, audio, info = io.read_video(
#         video_path,
#         start_pts=ele.get("video_start", 0.0),
#         end_pts=ele.get("video_end", None),
#         pts_unit="sec",
#         output_format="TCHW",
#     )
#     total_frames, video_fps = video.size(0), info["video_fps"]
#     logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
#     nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
#     idx = torch.linspace(0, total_frames - 1, nframes).round().long()
#     sample_fps = nframes / max(total_frames, 1e-6) * video_fps
#     video = video[idx]
#     return video, sample_fps


# def is_decord_available() -> bool:
#     import importlib.util

#     return importlib.util.find_spec("decord") is not None


# def _read_video_decord(
#     ele: dict,
# ) -> (torch.Tensor, float):
#     """read video using decord.VideoReader

#     Args:
#         ele (dict): a dict contains the configuration of video.
#         support keys:
#             - video: the path of video. support "file://", "http://", "https://" and local path.
#             - video_start: the start time of video.
#             - video_end: the end time of video.
#     Returns:
#         torch.Tensor: the video tensor with shape (T, C, H, W).
#     """
#     import decord
#     video_path = ele["video"]
#     st = time.time()
#     if video_path.startswith("oss://"):
#         # video_path = oss2http_url(video_path)
#         video_path = oss_reader.get_public_url(video_path)
#     vr = decord.VideoReader(video_path)
#     total_frames, video_fps = len(vr), vr.get_avg_fps()
#     start_frame, end_frame, total_frames = calculate_video_frame_range(
#         ele,
#         total_frames,
#         video_fps,
#     )
#     nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
#     idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
#     video = vr.get_batch(idx).asnumpy()
#     video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
#     logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
#     sample_fps = nframes / max(total_frames, 1e-6) * video_fps
#     return video, sample_fps


# def is_torchcodec_available() -> bool:
#     import importlib.util

#     return importlib.util.find_spec("torchcodec") is not None


# def _read_video_torchcodec(
#     ele: dict,
# ) -> (torch.Tensor, float):
#     """read video using torchcodec.decoders.VideoDecoder

#     Args:
#         ele (dict): a dict contains the configuration of video.
#         support keys:
#             - video: the path of video. support "file://", "http://", "https://" and local path.
#             - video_start: the start time of video.
#             - video_end: the end time of video.
#     Returns:
#         torch.Tensor: the video tensor with shape (T, C, H, W).
#     """
#     from torchcodec.decoders import VideoDecoder
#     TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
#     logger.info(f"set TORCHCODEC_NUM_THREADS: {TORCHCODEC_NUM_THREADS}")
#     video_path = ele["video"]
#     if video_path.startswith("oss://"):
#         video_path = oss2http_url(video_path)
#     st = time.time()
#     decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
#     video_fps = decoder.metadata.average_fps
#     total_frames = decoder.metadata.num_frames
#     start_frame, end_frame, total_frames = calculate_video_frame_range(
#         ele,
#         total_frames,
#         video_fps,
#     )
#     nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
#     idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
#     sample_fps = nframes / max(total_frames, 1e-6) * video_fps
#     video = decoder.get_frames_at(indices=idx).data
#     logger.info(f"torchcodec:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
#     return video, sample_fps


# def load_single_frame(oss_reader, oss_url, offset_size):
#     offset, size = offset_size
#     try:
#         data = oss_reader.read(f'{oss_url}.fps24.tar#{offset}-{offset+size}')
#         frame = np.array(Image.open(BytesIO(data)).convert("RGB"))
#         return frame
#     except Exception as e:
#         print(f"Error read oss_url: {oss_url} offset: {offset} size: {size}")
#         raise ValueError(f"Error read oss_url: {oss_url} offset: {offset} size: {size}")


# def _read_video_ossframes(
#     ele: dict,
# ) -> (torch.Tensor, float):
#     video_path = ele["video"]
#     assert video_path.startswith("oss://"), f"video_path must start with oss://, but got {video_path}"
#     video_path = video_path_mapping(video_path, use_shanghai_oss=USE_SHANGHAI_OSS)
#     # read frame index from oss
#     info = oss_reader.read(f'{video_path}.fps24.txt').decode('utf-8')
#     info = np.array([d.split() for d in info.splitlines()], dtype=np.int64)

#     total_frames = len(info)
#     video_fps = 24
#     st = time.time()
#     start_frame, end_frame, total_frames = calculate_video_frame_range(
#         ele,
#         total_frames,
#         video_fps,
#     )
#     nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
#     idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
#     sample_fps = nframes / max(total_frames, 1e-6) * video_fps
#     frame_indices = info[idx]
#     # load video frames
#     frames = []
#     max_workers = min(8, nframes)
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         load_frame = partial(load_single_frame, oss_reader, video_path)
#         frames = list(executor.map(load_frame, frame_indices))

#     frames = np.stack(frames)
#     video = torch.tensor(frames).permute(0, 3, 1, 2)  # Convert to TCHW format
#     logger.info(f"ossframes:  {video_path=}, {total_frames=}, {video_fps=}, {nframes=}, {sample_fps=}, time={time.time() - st:.3f}s")
#     return video, sample_fps


# VIDEO_READER_BACKENDS = {
#     "decord": _read_video_decord,
#     "torchvision": _read_video_torchvision,
#     "torchcodec": _read_video_torchcodec,
#     "ossframes": _read_video_ossframes,
# }


# FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", "").strip().lower()


# @lru_cache(maxsize=1)
# def get_video_reader_backend() -> str:
#     if FORCE_QWENVL_VIDEO_READER in ("1", "true", "t", "yes", "y"):
#         video_reader_backend = "torchcodec"
#     elif FORCE_QWENVL_VIDEO_READER in VIDEO_READER_BACKENDS:
#         video_reader_backend = FORCE_QWENVL_VIDEO_READER
#     elif is_torchcodec_available():
#         video_reader_backend = "torchcodec"
#     elif is_decord_available():
#         video_reader_backend = "decord"
#     else:
#         video_reader_backend = "torchvision"
#     print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
#     return video_reader_backend

# # zhenghao add, for getting stuck in fetch video issue
# ##########
# from multiprocessing import Process, Queue
# from typing import Any, Callable


# def _run_with_timeout(fn: Callable, args: tuple, kwargs: dict, timeout_sec: float):
#     """
#     在子进程里执行 fn(*args, **kwargs)。
#     成功 -> 返回结果；异常 -> 返回 ('err', repr(e))；超时 -> 返回 ('timeout', None)
#     """
#     q: Queue = Queue(maxsize=1)

#     def _worker(q: Queue, fn: Callable, args: tuple, kwargs: dict):
#         try:
#             out = fn(*args, **kwargs)
#             q.put(('ok', out))
#         except Exception as e:
#             q.put(('err', repr(e)))

#     p = Process(target=_worker, args=(q, fn, args, kwargs))
#     p.daemon = True
#     p.start()
#     p.join(timeout=timeout_sec)
#     if p.is_alive():
#         p.terminate()
#         p.join(1)
#         return ('timeout', None)

#     if q.empty():
#         return ('err', 'empty_queue')
#     return q.get()
    
# # 可调环境变量（不给就用默认）
# VIDEO_DECODE_TIMEOUT_SEC = float(os.environ.get('VIDEO_DECODE_TIMEOUT_SEC', '60'))
# VIDEO_DECODE_RETRIES = int(os.environ.get('VIDEO_DECODE_RETRIES', '2'))

# def _decode_once_with_timeout(backend_name: str, ele: dict):
#     """对某个 backend 做一次有超时的解码调用。"""
#     fn = VIDEO_READER_BACKENDS[backend_name]
#     status, payload = _run_with_timeout(fn, (ele,), {}, timeout_sec=VIDEO_DECODE_TIMEOUT_SEC)
#     if status == 'ok':
#         return payload  # (video_tensor, sample_fps)
#     elif status == 'timeout':
#         raise TimeoutError(f'{backend_name} decode timeout after {VIDEO_DECODE_TIMEOUT_SEC}s')
#     else:
#         # 'err', payload 是异常字符串
#         raise RuntimeError(f'{backend_name} decode failed: {payload}')
# ##########
# def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
#     if isinstance(ele["video"], str):
#         video_reader_backend = get_video_reader_backend()
#         try:
#             video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
#         except Exception as e:
#             logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
#             print("Error data: ", ele)
#             video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)


#         # primary_backend = get_video_reader_backend()   # 例如 'torchcodec'
#         # retries = VIDEO_DECODE_RETRIES
#         # last_err = None

#         # for attempt in range(1, retries + 1):
#         #     try:
#         #         video, sample_fps = _decode_once_with_timeout(primary_backend, ele)
#         #         break
#         #     except Exception as e:
#         #         last_err = e
#         #         logger.warning(f"[fetch_video] {primary_backend} attempt {attempt}/{retries} failed: {e}")
#         # else:
#         #     # 可选：降级一次到 'decord' 或 'torchvision'（不想降级就直接 raise last_err）
#         #     fallback = "decord" if is_decord_available() else "torchvision"
#         #     logger.warning(f"[fetch_video] falling back to {fallback} after {retries} failed attempts of {primary_backend}. last_err={last_err}")
#         #     video, sample_fps = _decode_once_with_timeout(fallback, ele)

#         nframes, _, height, width = video.shape
#         min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
#         total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
#         max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
#         max_pixels_supposed = ele.get("max_pixels", max_pixels)
#         if max_pixels_supposed > max_pixels:
#             logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
#         max_pixels = min(max_pixels_supposed, max_pixels)
#         if "resized_height" in ele and "resized_width" in ele:
#             resized_height, resized_width = smart_resize(
#                 ele["resized_height"],
#                 ele["resized_width"],
#                 factor=image_factor,
#             )
#         else:
#             resized_height, resized_width = smart_resize(
#                 height,
#                 width,
#                 factor=image_factor,
#                 min_pixels=min_pixels,
#                 max_pixels=max_pixels,
#             )

#         logger.info(f"video frames:  {height=}, {width=}, {min_pixels=}, {max_pixels=}, {resized_height=}, {resized_width=}")
#         video = transforms.functional.resize(
#             video,
#             [resized_height, resized_width],
#             interpolation=InterpolationMode.BICUBIC,
#             antialias=True,
#         ).float()
#         if return_video_sample_fps:
#             return video, sample_fps
#         return video
#     else:
#         assert isinstance(ele["video"], (list, tuple))
#         process_info = ele.copy()
#         process_info.pop("type", None)
#         process_info.pop("video", None)
#         images = [
#             fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
#             for video_element in ele["video"]
#         ]
#         nframes = ceil_by_factor(len(images), FRAME_FACTOR)
#         if len(images) < nframes:
#             images.extend([images[-1]] * (nframes - len(images)))
#         if return_video_sample_fps:
#             return images, process_info.pop("fps", 2.0)
#         return images


# def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
#     vision_infos = []
#     if isinstance(conversations[0], dict):
#         conversations = [conversations]
#     for conversation in conversations:
#         for message in conversation:
#             if isinstance(message["content"], list):
#                 for ele in message["content"]:
#                     if (
#                         "image" in ele
#                         or "image_url" in ele
#                         or "video" in ele
#                         or ele.get("type", "text") in ("image", "image_url", "video")
#                     ):
#                         vision_infos.append(ele)
#     return vision_infos


# def process_vision_info(
#     conversations: list[dict] | list[list[dict]],
#     return_video_kwargs: bool = False,
# ) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

#     vision_infos = extract_vision_info(conversations)
#     ## Read images or videos
#     image_inputs = []
#     video_inputs = []
#     video_sample_fps_list = []
#     for vision_info in vision_infos:
#         if "image" in vision_info or "image_url" in vision_info:
#             image_inputs.append(fetch_image(vision_info))
#         elif "video" in vision_info:
#             video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
#             video_sample_fps_list.append(video_sample_fps)
#             video_inputs.append(video_input)
#         else:
#             raise ValueError("image, image_url or video should in content.")
#     if len(image_inputs) == 0:
#         image_inputs = None
#     if len(video_inputs) == 0:
#         video_inputs = None
#     if return_video_kwargs:
#         return image_inputs, video_inputs, {'fps': video_sample_fps_list}
#     return image_inputs, video_inputs
