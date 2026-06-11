# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import List

import numpy as np
import ray

from agent_system.environments.base import EnvironmentManagerBase


class VideoEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
        self.env_name = config.env.env_name  # mark environment type
        self.max_steps = config.env.max_steps
        self.max_frames_len = config.env.video_star.max_frames_len
        self.max_audio_len = config.env.video_star.max_audio_len
        self.max_clip_len = config.env.video_star.max_clip_len
        self.mode = config.env.video_star.mode
        
    def reset(self, video_data_batch):
        """
        Reset video environment (TITO-adapted version - pass-through complex structures)
        """
        if video_data_batch is None:
            raise ValueError("video_data_batch must be provided")
        video_data_batch["mode"] = self.mode
        
        obs_list, infos = self.envs.reset_with_data(video_data_batch)
        
        text_batch = []
        token_segments_batch = []
        
        for obs in obs_list:
            if isinstance(obs, dict):
                text_batch.append(obs.get('text', []))
                # This will be a List[Dict]
                token_segments_batch.append(obs.get('token_segments'))
            elif isinstance(obs, list):
                text_batch.append(obs)
                token_segments_batch.append(None)
        
        res = {
            'text': text_batch
        }
        
        # Pass back if any element is non-empty
        if any(ts is not None for ts in token_segments_batch):
            res['token_segments'] = token_segments_batch
            
        return res, infos 

    def step(self, actions: List[str], action_ids: List[List[int]] = None):
        """
        Drive all workers to execute one step in parallel, collecting results.
        Previously only step_selected() existed; calling step() from inference scripts
        would fall back to the base class, causing incorrect return format.
        """
        idx_list = list(range(len(actions)))
        return self.step_selected(idx_list, actions, action_ids)

    def step_selected(self, idx_list: List[int], actions: List[str], action_ids: List[List[int]] = None):
        if action_ids is None:
            action_ids = [None] * len(idx_list)

        futures = []
        for env_id, act, act_id in zip(idx_list, actions, action_ids):
            # [Fix] Pass act_id to worker
            futures.append(self.envs.workers[env_id].step.remote(act, act_id))
            
        results = ray.get(futures)

        text_batch = []
        token_segments_batch = []
        reward_list, done_list, info_list = [], [], []
        
        for obs_res, r, d, info in results:
            if isinstance(obs_res, dict):
                text_batch.append(obs_res.get('text'))
                # [Fix] Collect token_segments
                token_segments_batch.append(obs_res.get('token_segments'))
            else:
                text_batch.append(obs_res)
                token_segments_batch.append(None)
                
            reward_list.append(r)
            done_list.append(d)
            info_list.append(info)

        obs_subset = {
            "text": text_batch
        }
        
        # Only include when valid data exists
        if any(ts is not None for ts in token_segments_batch):
            obs_subset["token_segments"] = token_segments_batch

        return obs_subset, np.array(reward_list), np.array(done_list), info_list
    
def make_envs(config):
    """
    Create environments.
    """
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "video_env" in config.env.env_name.lower():
        print("Using Video Environment")
        from agent_system.environments.env_package.video_env import (
            build_video_envs, video_projection)

        train_batch_size = config.data.train_batch_size * group_n
        print(f"[Train Environment] Using train_batch_size {config.data.train_batch_size} * env.rollout.n {group_n} : {train_batch_size}")
        _train_envs = build_video_envs(
            seed        = config.env.seed,
            env_num     = config.data.train_batch_size,
            group_n     = group_n,
            max_frames_len = config.env.video_star.max_frames_len,
            max_audio_len=config.env.video_star.max_audio_len,
            max_clip_len =config.env.video_star.max_clip_len,
            max_steps   = config.env.max_steps,
            processor_path=config.actor_rollout_ref.model.path,
            max_prompt_len=config.data.max_prompt_length,
            max_response_len=config.data.max_response_length,
            is_train=True
        )

        val_batch_size = config.data.val_batch_size * config.env.rollout.n
        if config.env.rollout.pool_async_enabled:
            print(f"[Validate Environment] pool_async_enabled. Using val_batch_size * env.rollout.n: {val_batch_size}")
        else:
            print(f"[Validate Environment] sync_enabled. Using val_batch_size * env.rollout.n: {val_batch_size}")


        _val_envs = build_video_envs(
            seed        = config.env.seed + 1000,
            env_num     = config.data.val_batch_size,
            group_n     = group_n,
            max_frames_len = config.env.video_star.max_frames_len,
            max_audio_len=config.env.video_star.max_audio_len,
            max_clip_len =config.env.video_star.max_clip_len,
            max_steps   = config.env.max_steps,
            processor_path=config.actor_rollout_ref.model.path,
            max_prompt_len=config.data.max_prompt_length,
            max_response_len=config.data.max_response_length,
            is_train=False
        )

        projection_f = partial(video_projection)
        envs = VideoEnvironmentManager(_train_envs, projection_f, config)
        val_envs = VideoEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        raise ValueError(f"Environment not supported: {config.env.env_name}")