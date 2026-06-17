# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Licensed under the Apache License, Version 2.0 (the "License");

import time
import uuid
from typing import Dict, List, Tuple
import os

import numpy as np
import torch
from qwen_omni_utils import process_audio_info
from qwen_vl_utils import process_vision_info
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout.utils import (filter_group_data,
                                                   to_list_of_dict,
                                                   torch_to_numpy)
from verl import DataProto
from verl.models.transformers.qwen2_5_omni import \
    get_rope_index as get_rope_index_omni
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import json

# =============== New utility functions ===============
def _traj_has_exceed_error(info_list: List[Dict]) -> bool:
    """
    Check whether a trajectory was terminated by the environment due to
    reaching max_steps (or step_mismatch).
    Returns True if any step's info['error_code'] belongs to the exceed set.
    """
    exceed_codes = {"STEP_LIMIT_REACHED", "TOKEN_LIMIT_EXCEEDED", "FFMPEG_AUDIO_FAIL", "FFMPEG_FRAME_FAIL", "FFMPEG_CLIP_FAIL"} # 
    for inf in info_list:
        if inf.get("error_code") in exceed_codes:
            return True
    return False
# ==========================================


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    # 2) New method inside TrajectoryCollector
    def _collect_step_stats(self, batch: DataProto, infos, actions):
        ntb = batch.non_tensor_batch

        def _ensure(key, dtype=object):
            if key not in ntb:
                ntb[key] = np.empty(0, dtype=dtype)

        # Field initialization
        _ensure("error_code", object)
        _ensure("action_type", object)
        _ensure("n_frames", float)
        for fld in ("clip_start","clip_end","clip_dur","audio_start","audio_end","audio_dur"):
            _ensure(fld, float)

        # Iterate over each env
        for info, txt in zip(infos, actions):
            err = info.get("error_code", "OK")
            ntb["error_code"] = np.append(ntb["error_code"], err)

            # Default placeholder
            act_type = "ERROR" if err != "OK" else "UNKNOWN"
            n_frames = np.nan
            clip_s = clip_e = clip_d = audio_s = audio_e = audio_d = np.nan

            if err == "OK":
                act_obj = json.loads(txt).get("action", {})
                act_type = act_obj.get("type")

                # --- Compatible with both old and new get_frames logic ---
                if act_type == "get_frames":
                    # Method A (new): start, end, num
                    if "num" in act_obj:
                        n_frames = float(act_obj.get("num", 0))
                        s, e = act_obj.get("start"), act_obj.get("end")
                        if s is not None and e is not None:
                            clip_s, clip_e = float(s), float(e)
                            clip_d = clip_e - clip_s
                    # Method B (legacy): timestamps list
                    elif "timestamps" in act_obj:
                        ts = act_obj.get("timestamps", [])
                        n_frames = len(ts) if isinstance(ts, list) else np.nan

                if act_type in ("get_clip", "get_audio"):
                    def _to_f(x):
                        return float(x)
                    s, e = _to_f(act_obj.get("start")), _to_f(act_obj.get("end"))
                    d = e - s if np.isfinite(s) and np.isfinite(e) else np.nan
                    if act_type == "get_clip":
                        clip_s, clip_e, clip_d = s, e, d
                    else:
                        audio_s, audio_e, audio_d = s, e, d

            # Append columns
            ntb["action_type"] = np.append(ntb["action_type"], act_type)
            ntb["n_frames"]    = np.append(ntb["n_frames"], n_frames)
            ntb["clip_start"]  = np.append(ntb["clip_start"], clip_s)
            ntb["clip_end"]    = np.append(ntb["clip_end"], clip_e)
            ntb["clip_dur"]    = np.append(ntb["clip_dur"], clip_d)
            ntb["audio_start"] = np.append(ntb["audio_start"], audio_s)
            ntb["audio_end"]   = np.append(ntb["audio_end"], audio_e)
            ntb["audio_dur"]   = np.append(ntb["audio_dur"], audio_d)

                    
    # =====================================================================================
    #  Preprocessing, pool_async_multi_turn_loop, and related functions
    # =====================================================================================
    def preprocess_single_sample_omni(self, item: int, gen_batch: DataProto, obs: Dict):
        """
        Qwen2.5-Omni single-sample preprocessing (TITO ultimate fix version)
        1. Fix UnboundLocalError: variables are self-contained within branches, unified to use mm_data.
        2. Fix Index OOB: adopt Piecewise Processing to bypass Processor's global image limit.
        """
        has_audio_list = gen_batch.non_tensor_batch.get("has_audio")
        has_audio = bool(has_audio_list[item]) if has_audio_list is not None else True

        # [TITO Source]
        token_segments = obs.get("token_segments")
        messages = obs["text"][item]

        # 1. [Critical] Pre-initialize all result containers to prevent UnboundLocalError
        ids, mask, pos, raw_prompt_ids = None, None, None, None
        mm_inputs = {}
        # Pre-initialize mm_data to ensure all branches can access it
        mm_data = {"image": [], "video": [], "audio": []}
        
        # Legacy mode temporary variable initialization (for safety)
        img_inputs, vid_inputs, audio_inputs = None, None, None
        
        def to_numpy(x): return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

        # =============================================================
        # Branch A: TITO mode (Strict Piecewise Assembly)
        # =============================================================
        if token_segments is not None and token_segments[item] is not None:
            segments_list = token_segments[item] # List[Dict]
            
            full_input_ids = []
            
            def compact_for_vllm(ids_list):
                if not ids_list: return []

                # Token definitions
                BLOCK_START_TOKENS = {151652, 151647}  # vision_start, audio_start
                BLOCK_END_TOKENS = {151653, 151648}    # vision_end, audio_end
                MM_PAD_TOKENS = {151655, 151656, 151646} # image, video, audio pads

                res = []
                in_block = False
                seen_pad = False

                i = 0
                while i < len(ids_list):
                    token = ids_list[i]
                    
                    if token in BLOCK_START_TOKENS:
                        in_block = True
                        seen_pad = False
                        res.append(token)
                        i += 1
                        continue
                        
                    if token in BLOCK_END_TOKENS:
                        in_block = False
                        res.append(token)
                        i += 1
                        continue

                    # If inside a multimodal block and token is a feature placeholder
                    if in_block and token in MM_PAD_TOKENS:
                        if not seen_pad:
                            res.append(token) # Keep only the first one
                            seen_pad = True
                        # Skip all subsequent consecutive identical pad tokens
                        j = i + 1
                        while j < len(ids_list) and ids_list[j] == token:
                            j += 1
                        i = j
                        continue

                    # Regular text token
                    res.append(token)
                    i += 1
                return res

            
            # Meta (from Env)
            list_image_grid = []
            list_video_grid = []
            list_video_second_per_grid = []
            list_audio_lens = []
            
            list_pixel_values = []
            list_pixel_values_videos = []

            # --- [KEY] Piecewise iteration processing (Piecewise Processing) ---
            # Force message-segment one-to-one correspondence, generate Tensor for each individually
            assert len(segments_list) == len(messages), f"Segments and messages count mismatch: {len(segments_list)} vs {len(messages)}"
            loop_len = len(segments_list)
            
            for i in range(loop_len):
                seg = segments_list[i]
                msg = messages[i]
                
                # A. Collect IDs & Meta (Trust Env)
                seg_ids = seg.get("input_ids", [])
                if isinstance(seg_ids, torch.Tensor): seg_ids = seg_ids.tolist()
                full_input_ids.extend(seg_ids)

                # 2. Directly use the Tensor already processed by the environment
                if seg.get("pixel_values") is not None:
                    list_pixel_values.append(seg["pixel_values"])
                if seg.get("pixel_values_videos") is not None:
                    list_pixel_values_videos.append(seg["pixel_values_videos"])

                if seg.get("image_grid_thw") is not None: list_image_grid.append(seg["image_grid_thw"])
                if seg.get("video_grid_thw") is not None: list_video_grid.append(seg["video_grid_thw"])
                
                spg = seg.get("video_second_per_grid")
                if spg is not None: list_video_second_per_grid.append(spg)
                
                if seg.get("audio_feature_lengths") is not None: list_audio_lens.append(seg["audio_feature_lengths"])

                # --- B. [Fix] Directly reuse the same-source mm_data cached on Env side ---
                seg_mm = seg.get("mm_data")
                if seg_mm:
                    if seg_mm.get("image"):
                        mm_data["image"].extend([to_numpy(x) for x in seg_mm["image"]])
                    if seg_mm.get("video"):
                        mm_data["video"].extend([to_numpy(x) for x in seg_mm["video"]])
                    if seg_mm.get("audio"):
                        mm_data["audio"].extend([to_numpy(x) for x in seg_mm["audio"]])


            # =================================================================
            # [CRITICAL FIX] Manually add Generation Prompt (the starting gun)
            # =================================================================
            # The logic here: after TITO assembles all history, the Assistant Header
            # must be manually appended to tell the model "it's your turn to speak",
            # otherwise the model will output garbled text.
            # Qwen2.5-VL/Omni standard Assistant header: <|im_start|>assistant\n
            # IDs: [151644, 77091, 198] (verified)
            # =================================================================
            
            ASSISTANT_HEADER_IDS = [151644, 77091, 198] 
            full_input_ids.extend(ASSISTANT_HEADER_IDS)
            raw_prompt_ids = compact_for_vllm(full_input_ids)


            # 3. Build Input IDs Tensor
            ids_raw = torch.tensor([full_input_ids], dtype=torch.long)
            mask_raw = torch.ones_like(ids_raw)
            
            ids, mask = verl_F.postprocess_data(
                input_ids=ids_raw,
                attention_mask=mask_raw,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )

            # 4. Aggregate all Tensors
            image_grid_thw = torch.cat(list_image_grid, dim=0) if list_image_grid else None
            video_grid_thw = torch.cat(list_video_grid, dim=0) if list_video_grid else None
            video_second_per_grid = torch.cat(list_video_second_per_grid, dim=0) if list_video_second_per_grid else None
            audio_seqlens = torch.cat(list_audio_lens, dim=0) if list_audio_lens else None

            pixel_values = torch.cat(list_pixel_values, dim=0) if list_pixel_values else None
            pixel_values_videos = torch.cat(list_pixel_values_videos, dim=0) if list_pixel_values_videos else None

            # 5. Populate mm_inputs
            if image_grid_thw is not None: mm_inputs["image_grid_thw"] = image_grid_thw
            if video_grid_thw is not None: mm_inputs["video_grid_thw"] = video_grid_thw
            if audio_seqlens is not None: mm_inputs["audio_feature_lengths"] = audio_seqlens
            if video_second_per_grid is not None: mm_inputs["video_second_per_grid"] = video_second_per_grid
            if pixel_values is not None: mm_inputs["pixel_values"] = pixel_values
            if pixel_values_videos is not None: mm_inputs["pixel_values_videos"] = pixel_values_videos

            # 6. Compute RoPE Position IDs
            if has_audio and audio_seqlens is not None:
                pos = get_rope_index_omni(
                    self.processor, input_ids=ids, 
                    image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                    attention_mask=mask, use_audio_in_video=True,
                    audio_seqlens=audio_seqlens, second_per_grids=video_second_per_grid,
                )[0]
                pos = [pos.permute(1, 0, 2)[0]]
            else:
                pos = [get_rope_index(
                    self.processor, input_ids=ids[0],
                    image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                    second_per_grid_ts=video_second_per_grid, attention_mask=mask[0],
                )]

        # =============================================================
        # Branch B: Legacy Mode
        # =============================================================
        else:
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, 
                **self.config.get("apply_chat_template_kwargs", {})
            )
            if isinstance(prompt, list): prompt = prompt[0]
            # print(f"Current Prompt after applying chat template: {prompt}")  # Debug log
            
            img_inputs, vid_inputs = process_vision_info(messages)
            if has_audio:
                try: audio_inputs = process_audio_info(messages, use_audio_in_video=True)
                except Exception: audio_inputs = None

            proc_kwargs = {"text": [prompt], "return_tensors": "pt"}
            if img_inputs: proc_kwargs["images"] = img_inputs
            if vid_inputs: proc_kwargs["videos"] = vid_inputs
            if audio_inputs: 
                proc_kwargs["audio"] = audio_inputs
                proc_kwargs["use_audio_in_video"] = True
            else:
                proc_kwargs["use_audio_in_video"] = False

            proc_out = self.processor(**proc_kwargs)
            
            # [Fix] Populate mm_data in the Legacy branch
            if img_inputs: mm_data["image"] = [to_numpy(x) for x in img_inputs]
            if vid_inputs: mm_data["video"] = [to_numpy(x) for x in vid_inputs]
            if audio_inputs: mm_data["audio"] = [to_numpy(x) for x in audio_inputs]

            ids = proc_out.pop("input_ids", None)
            mask = proc_out.pop("attention_mask", None)
            ids, mask = verl_F.postprocess_data(
                input_ids=ids, attention_mask=mask,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True, truncation=self.config.data.truncation,
            )
            
            wanted_keys = ["pixel_values", "pixel_values_videos", "input_features", 
                      "image_grid_thw", "video_grid_thw", "audio_feature_lengths", 
                      "video_second_per_grid", "feature_attention_mask"]
            for k in wanted_keys:
                if k in proc_out and proc_out[k] is not None:
                    mm_inputs[k] = proc_out[k]
            
            if audio_inputs and "feature_attention_mask" in proc_out:
                mm_inputs["audio_feature_lengths"] = torch.sum(proc_out["feature_attention_mask"], dim=1)

            enable_audio = proc_kwargs["use_audio_in_video"]
            if enable_audio and audio_inputs:
                 pos = get_rope_index_omni(
                    self.processor, input_ids=ids,
                    image_grid_thw=mm_inputs.get("image_grid_thw"),
                    video_grid_thw=mm_inputs.get("video_grid_thw"),
                    attention_mask=mask, use_audio_in_video=True,
                    audio_seqlens=mm_inputs.get("audio_feature_lengths"),
                    second_per_grids= mm_inputs.get("video_second_per_grid"),
                )[0]
                 pos = [pos.permute(1, 0, 2)[0]]
            else:
                 pos = [get_rope_index(
                    self.processor, input_ids=ids[0],
                    image_grid_thw=mm_inputs.get("image_grid_thw"),
                    video_grid_thw=mm_inputs.get("video_grid_thw"),
                    second_per_grid_ts=mm_inputs.get("video_second_per_grid"),
                    attention_mask=mask[0],
                )]
            
            raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # =============================================================
        # Final Return (Safe)
        # =============================================================
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
             if self.config.data.truncation == "error":
                 raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} > {self.config.data.max_prompt_length}")
             raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length:] 

        row_dict = {
            "input_ids":      ids[0], 
            "attention_mask": mask[0], 
            "position_ids":   pos[0],
            "raw_prompt_ids": raw_prompt_ids, 
            "anchor_obs":     None, 
            "index":          item,
            "data_source":    gen_batch.non_tensor_batch["data_source"][item],
            "multi_modal_inputs": mm_inputs, 
            "multi_modal_data": mm_data,  # [Fix] mm_data is guaranteed to be populated at this point
            "mm_processor_kwargs": {"use_audio_in_video": has_audio}
        }
        if self.config.data.get("return_raw_chat", False):
            row_dict["raw_prompt"] = messages

        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Check if this is a video_env sample
            if self.config.env.env_name == "video_env":
                # Extract per-sample observations using standard processing
                # print("Starting preprocess_single_sample_omni processing (zhenghao)")
                processed = self.preprocess_single_sample_omni(
                    item=item,
                    gen_batch=gen_batch,
                    obs=obs,
                )
                # print("Finished preprocess_single_sample_omni processing (zhenghao)")
            else:
                raise NotImplementedError("Only video_env with Qwen2.5-Omni is implemented in this version.")
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards : np.ndarray,
            episode_lengths : np.ndarray,
            success         : Dict[str, np.ndarray],
            traj_uid        : np.ndarray,
            exceeded        : np.ndarray,           
    ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        
        Returns:
            DataProto: Collected and organized trajectory data
        """

        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)

        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            traj_exceed = bool(exceeded[bs])
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value
                    data['exceed_mask'] = torch.tensor(traj_exceed, dtype=torch.bool)
                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )

        # ----------- Immediately discard temporary caches from this trajectory -----------
        total_batch_list.clear()
        effective_batch.clear()

        del total_batch_list, effective_batch
        import gc; gc.collect()
        # ---------------------------------------------------------------

        return gen_batch_output

    # =====================================================================================
    # [ARCHITECTURAL DESIGN: Pool-based Asynchronous Life-cycle Management]
    #
    # 1. Core Challenge (The Straggler Problem):
    #    The traditional synchronous approach forces all envs to run for a fixed number of
    #    steps (T). If Env A finishes at step 2 while Env B needs 20 steps, Env A is forced
    #    to idle, wasting expensive GPU time waiting for the slowest env (the straggler).
    #
    # 2. Asynchronous Solution (Asynchronous Eviction):
    #    Introduce an `alive_ids` dynamic pool. Each trajectory has an independent lifecycle;
    #    once `done=True`, it is immediately evicted from the pool, and the system no longer
    #    allocates any inference resources to it, eliminating the "weakest link" effect in
    #    agentic tasks.
    #
    # 3. Core Mechanism (Chunking & Micro-batching):
    #    Why split alive_ids into chunks?
    #    - VRAM Safety: Multimodal models (Qwen2.5-Omni) consume enormous VRAM when processing
    #      video/images. Pushing all 512 envs into GPU inference at once would cause OOM.
    #      Chunking is the "safety valve" for VRAM.
    #    - Throughput Matching: `MICRO_RATIO` sets the max alive envs per GPU
    #      in one generation wave: chunk_size_max = world_size * MICRO_RATIO.
    #      Larger rollout batches are split into multiple waves.
    #
    # 4. Train-Inference Consistency (TITO & Causal Isolation):
    #    At the start of each chunk loop, capture the "old observation" snapshot to ensure
    #    that the sequence the model sees during training update is binary-identical to what
    #    it saw during inference. Even if the env appends feedback after the step, it does
    #    not pollute the current training round.
    # =====================================================================================
    def pool_async_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        is_train: bool,
    ):
        t_loop0 = time.perf_counter()

        world_size  = actor_rollout_wg.world_size     # GPU / worker count
        ratio = int(float(os.getenv("MICRO_RATIO", "1")))
        if ratio < 1:
            raise ValueError(f"MICRO_RATIO must be >= 1, got {ratio}")
        chunk_size_max = world_size * ratio
        chunk_size_max = max(1, chunk_size_max)

        # --- New: read retry environment variable ---
        retry_on_format = os.getenv("RETRY_ON_FORMAT_ERROR", "false").lower() in ("1", "true", "t", "yes", "y")

        total_envs  = batch_size = len(gen_batch.batch)
        assert total_envs > 0, "gen_batch is empty!"

        if is_train:
            # Training phase must equal exactly train_bs * n
            expected = self.config.data.train_batch_size * self.config.env.rollout.n
            assert total_envs == expected, (
                f"wrong train env size: expect {expected}, got {total_envs}"
            )
        else:  # validation
            max_envs = self.config.data.val_batch_size * self.config.env.rollout.n
            assert total_envs <= max_envs, (
                f"wrong val env size, expect <= {max_envs}, got {total_envs}"
            )
        # --------------------------------------------------------------

        # ---------- reset ----------
        if getattr(envs, "env_name", "") == "video_env":
            video_keys = ["video", "question_type", "question", "answer",
                          "options", "fps", "duration_seconds", "has_audio"]
            video_data_batch = {k: gen_batch.non_tensor_batch.get(k, [])
                                for k in video_keys}
            obs_all, _ = envs.reset(video_data_batch)
        else:
            obs_all, _ = envs.reset()

        obs_keys = list(obs_all.keys())
        obs_pool = {i: {k: (obs_all[k][i] if obs_all[k] is not None else None)
                        for k in obs_keys}
                    for i in range(total_envs)}

        # ---------- uid ----------
        if is_train and self.config.env.rollout.n > 0:
            repeat_n  = self.config.env.rollout.n
            orig_bs   = total_envs // repeat_n   # Do not use comma
            assert orig_bs == self.config.data.train_batch_size, (
                f"wrong original batch size, N ({self.config.env.rollout.n}), "
                f"orig_bs ({orig_bs}), total envs ({total_envs})."
            )
            uid_batch = np.repeat([str(uuid.uuid4()) for _ in range(orig_bs)],
                                  repeat_n).astype(object)
        else:
            # set all to different uid, each sample is unique
            uid_batch = np.array([str(uuid.uuid4()) for _ in range(total_envs)], dtype=object)

        # ---------- buffers ----------
        alive_ids        = list(range(total_envs))          # Envs not yet done
        env_done         = np.zeros(total_envs, dtype=bool)
        traj_uid         = np.array([str(uuid.uuid4()) for _ in range(total_envs)],
                                    dtype=object)
        env_step_cnt     = np.zeros(total_envs, dtype=np.int32)

        episode_rewards  = np.zeros(total_envs, np.float32)
        episode_lengths  = np.zeros(total_envs, np.float32)

        total_retries = np.zeros(total_envs, np.float32)

        total_batch_list = [[] for _ in range(total_envs)]
        total_infos      = [[] for _ in range(total_envs)]

        global_step = 0
        # ================= Outer loop: until all done =================
        while alive_ids:
            step_t0 = time.perf_counter()
            print(f"\n===== GLOBAL STEP {global_step} | alive={len(alive_ids)} =====")

            # Split current alive_ids into chunks, each <= chunk_size_max
            chunks = [alive_ids[i:i + chunk_size_max]
                      for i in range(0, len(alive_ids), chunk_size_max)]

            # ----------- Process chunks sequentially -----------
            for chunk_idx, idx_chunk in enumerate(chunks):
                chunk_t0 = time.perf_counter()
                # ---- 1. Preprocessing ----
                sub_gen_batch = gen_batch.select_idxs(idx_chunk)

                # [Note A] sub_obs here comes from obs_pool and is the state before the model speaks.
                # Regardless of how step_selected modifies the env history afterwards, sub_obs is
                # already physically isolated, ensuring the generated batch.input_ids is clean.
                sub_obs = {k: [obs_pool[i][k] for i in idx_chunk]
                           for k in obs_keys}
                batch = self.preprocess_batch(sub_gen_batch, sub_obs)

                # ---- 2. pop & pad ----
                pop_keys    = ["input_ids", "attention_mask", "position_ids"]
                pop_nt_keys = ["raw_prompt_ids"]
                for extra in ["multi_modal_data", "raw_prompt",
                              "tools_kwargs", "mm_processor_kwargs"]:
                    if extra in batch.non_tensor_batch:
                        pop_nt_keys.append(extra)

                model_in = batch.pop(batch_keys=pop_keys,
                                     non_tensor_batch_keys=pop_nt_keys)
                model_in.meta_info = gen_batch.meta_info


                divisor  = actor_rollout_wg.world_size   # = dp_size
                bs_cur   = len(model_in)
                pad_needed = (divisor - bs_cur % divisor) % divisor   # How many to pad
                can_pad    = (pad_needed > 0) and (bs_cur + pad_needed <= chunk_size_max)

                if can_pad:                                           # Case 1: can pad, so pad
                    model_in_pad, pad_sz = pad_dataproto_to_divisor(model_in, divisor)
                else:                                                 # Case 2: no padding
                    # Assert: if not padding, bs_cur must already be divisible by divisor
                    assert bs_cur % divisor == 0, (
                        f"Batch size {bs_cur} is not divisible by world_size={divisor}, "
                        f"and padding ({pad_needed}) would exceed chunk_size_max={chunk_size_max}."
                    )
                    model_in_pad, pad_sz = model_in, 0

                # ---- 3. forward ---------------------------------------------------
                try:
                    out_pad = actor_rollout_wg.generate_sequences(model_in_pad)
                except Exception as e:
                    print("\n[DEBUG] world_size:", actor_rollout_wg.world_size)
                    print("\n[DEBUG] chunk_size_max:", chunk_size_max)
                    # print("[DEBUG] out_pad:", model_in_pad)
                    raise RuntimeError(f"generate_sequences failed: {e}") from e
                # ---------------------------------------------------------------
                # After generate_sequences:
                sub_out = unpad_dataproto(out_pad, pad_sz)

                # 1. Extract Response Tensor
                response_tensor = sub_out.batch["responses"]
                response_ids_raw = response_tensor.cpu().tolist()

                CLEAR_TOKEN_LIST = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
                response_ids_clean = [[tid for tid in seq if tid not in CLEAR_TOKEN_LIST] for seq in response_ids_raw]

                responses = self.tokenizer.batch_decode(sub_out.batch["responses"], skip_special_tokens=True)

                # 3. Pass through to Env for execution. Note: feedback is produced here
                # and written into the env's internal history, as well as returned in
                # next_obs and infos.
                time_env_step = time.perf_counter()
                next_obs, rewards, dones, infos = envs.step_selected(
                    idx_chunk, 
                    responses, 
                    action_ids=response_ids_clean
                )
                time_env_step = time.perf_counter() - time_env_step
                print(f"  ├─ env.step_selected time: {time_env_step:.2f}s")

                # [Note B] The infos just produced by step_selected are collected into
                # batch.non_tensor_batch, but only as "metadata" for downstream reward
                # computation or logging. They are NOT appended to batch.input_ids and
                # thus do not participate in the model's token sequence training.
                self._collect_step_stats(batch, infos, responses)

                rewards = np.asarray(rewards).reshape(-1)
                dones   = np.asarray(dones).reshape(-1)

                # ---- 5. merge & log ----
                batch.non_tensor_batch["uid"]         = uid_batch[idx_chunk]
                batch.non_tensor_batch["traj_uid"]    = traj_uid[idx_chunk]
                batch.non_tensor_batch["cur_step"]    = env_step_cnt[idx_chunk]
                batch.non_tensor_batch["rewards"]     = torch_to_numpy(rewards, is_object=True)
                batch.non_tensor_batch["active_masks"]= np.ones(len(idx_chunk), dtype=bool)
                batch.non_tensor_batch["step_infos"]  = np.array(infos, dtype=object)
                batch.non_tensor_batch["is_action_valid"] = np.array(
                    [info.get("is_action_valid", True) for info in infos], dtype=bool)

                # [Note C] The union operation here only merges: [old Prompt] + [model-generated Response].
                # Since the batch object is independent in memory, it does not contain the
                # feedback text just produced by the environment.
                batch = batch.union(sub_out)
                list_dict = to_list_of_dict(batch)

                # ---- 6. per-env update ----
                for loc, env_id in enumerate(idx_chunk):
                    info = infos[loc]
                    is_valid = info.get("is_action_valid", True)
                    is_done = dones[loc]

                    # --- [Core physical deletion and statistics logic] ---
                    if retry_on_format and (not is_valid) and (not is_done):
                        # 1. Record one retry (a physically occurred error)
                        total_retries[env_id] += 1
                        
                        # 2. Physical step count still increments to prevent infinite retries, bounded by max_steps
                        env_step_cnt[env_id] += 1 
                        
                        # 3. Update obs_pool so that the next round can see the error history
                        obs_pool[env_id] = {k: (next_obs[k][loc] if next_obs[k] is not None else None) for k in obs_keys}
                        
                        # 4. Skip all append operations (physical deletion)
                        continue 

                    # --- [Normal retention logic] ---
                    total_infos[env_id].append(infos[loc])
                    total_batch_list[env_id].append(list_dict[loc])

                    episode_rewards[env_id] += rewards[loc]
                    episode_lengths[env_id] += 1 # Training step count increment
                    env_step_cnt[env_id]    += 1 # Physical step count increment

                    obs_pool[env_id] = {k: (next_obs[k][loc] if next_obs[k] is not None else None) for k in obs_keys}
                    if is_done:
                        env_done[env_id] = True

                    # reward mean = len(idx_chunk)*rewards.mean()/dones.sum()
                    rewards_mean = len(idx_chunk) * rewards.mean() / max(1, dones.sum())
                    print(f"  ├─ chunk {chunk_idx:02d} | size={len(idx_chunk):2d} | "
                        f"rewards mean={rewards_mean:.2f} | dones={dones.sum()} | "
                        f"time={time.perf_counter()-chunk_t0:.2f}s")

            # ---- 7. All chunks done: update alive_ids ----
            # [Note D] This is the physical boundary where the "last-step feedback" disappears:
            # Although the env left a CORRECT/INCORRECT farewell message in obs_pool,
            # because env_done[eid] is True, the env is removed from alive_ids.
            # The next iteration no longer runs preprocess_batch for it, so that farewell
            # message never becomes training tokens.
            alive_ids = [eid for eid in alive_ids if not env_done[eid]]

            global_step += 1
            print(f"Step time {time.perf_counter()-step_t0:.2f}s | "
                  f"remaining envs {len(alive_ids)}")

        # ================= gather =================
        success = envs.success_evaluator(
            total_infos      = total_infos,
            total_batch_list = total_batch_list,
            episode_rewards  = episode_rewards,
            episode_lengths  = episode_lengths,
        )

        # 1. Basic retry count
        success['format_retry_count'] = total_retries

        # 2. Compute First Pass Success Rate
        if 'won' in success:
            won_array = np.array(success['won'])
            first_pass_mask = (won_array > 0.5) & (total_retries == 0)
            success['first_pass_success'] = first_pass_mask.astype(np.float32)

        # 3. Compute First Pass Reward Rate
        # Logic: only extract rewards from trajectories that had no retries.
        # For retried trajectories, record 0 so the mean reflects "first-pass high-score capability"
        first_pass_reward = np.where(total_retries == 0, episode_rewards, 0.0)
        success['first_pass_reward'] = first_pass_reward.astype(np.float32)

        # 3. Track which trajectories hit step limit or encountered fatal errors (exceeded)
        # Note: this must be initialized before the return!
        exceeded = torch.zeros(batch_size, dtype=torch.bool) 
        if is_train:
            for i in range(batch_size):
                if _traj_has_exceed_error(total_infos[i]):
                    exceeded[i] = True

        print(f"\n[Async] FINISHED {total_envs} envs in {global_step} steps | "
              f"total wall-clock {time.perf_counter()-t_loop0:.2f}s")

        # 4. Final single Return
        return (total_batch_list, episode_rewards, episode_lengths,
                success, traj_uid, exceeded)

    # =====================================================================================
    # Select one rollout implementation based on config
    # =====================================================================================
    def _run_one_rollout(self, gen_batch, actor_rollout_wg, envs, is_train):
        if getattr(self.config.env.rollout, "pool_async_enabled", False):
            return self.pool_async_multi_turn_loop(gen_batch, actor_rollout_wg, envs, is_train)
        else:
            raise NotImplementedError("Only pool_async_multi_turn_loop is implemented in this version.")

    # =====================================================================================
    # dynamic_multi_turn_loop modification: uses _run_one_rollout
    # =====================================================================================
    def dynamic_multi_turn_loop(self, gen_batch, actor_rollout_wg,
                                envs, is_train):

        total_batch_list, total_episode_rewards, total_episode_lengths = [], [], []
        total_success, total_traj_uid, total_exceeded = [], [], []

        try_cnt, max_try = 0, self.config.algorithm.filter_groups.max_num_gen_batches
        target_num = self.config.data.train_batch_size * self.config.env.rollout.n

        while len(total_batch_list) < target_num and try_cnt < max_try:
            try_cnt += 1
            (batch_list, ep_r, ep_l, suc,
             traj_uid, exceeded) = self._run_one_rollout(
                                        gen_batch, actor_rollout_wg, envs, is_train)

            (batch_list, ep_r, ep_l, suc,
             traj_uid, exceeded) = filter_group_data(
                                        batch_list, ep_r, ep_l, suc,
                                        traj_uid, exceeded,
                                        config=self.config,
                                        last_try=False) # (try_cnt == max_try) / False

            total_batch_list += batch_list
            total_episode_rewards.append(ep_r)
            total_episode_lengths.append(ep_l)
            total_success.append(suc)
            total_traj_uid.append(traj_uid)
            total_exceeded.append(exceeded)

        print(f"[Dynamic] Try {try_cnt}/{max_try} | "
              f"total_batch_size={len(total_batch_list)} | "
              f"target_num={target_num}")
        total_episode_rewards = np.concatenate(total_episode_rewards)
        total_episode_lengths = np.concatenate(total_episode_lengths)
        total_traj_uid        = np.concatenate(total_traj_uid)
        total_exceeded        = np.concatenate(total_exceeded)
        total_success = {k: np.concatenate([s[k] for s in total_success], 0)
                         for k in total_success[0]}

        return (total_batch_list, total_episode_rewards,
                total_episode_lengths, total_success,
                total_traj_uid, total_exceeded)

    # =====================================================================================
    # multi_turn_loop main controller: preserves original return, consistent logic
    # =====================================================================================
    def multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg,
                        envs: EnvironmentManagerBase, is_train: bool = True):

        if is_train:
            print("Before train gen_batch repeat: ", len(gen_batch))
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n,
                                         interleave=True)
            print("After train gen_batch repeat: ", len(gen_batch))
        if self.config.algorithm.filter_groups.enable and is_train:
            lists, R, L, suc, traj_uid, exceeded = self.dynamic_multi_turn_loop(
                gen_batch, actor_rollout_wg, envs, is_train)
        else:
            lists, R, L, suc, traj_uid, exceeded = self._run_one_rollout(
                gen_batch, actor_rollout_wg, envs, is_train)

        gen_batch_output = self.gather_rollout_data(
            lists, R, L, suc, traj_uid, exceeded)

        # Release intermediate caches from this round to prevent memory accumulation in TaskRunner
        del lists, R, L, suc, traj_uid, exceeded
        import gc; gc.collect()

        return gen_batch_output
