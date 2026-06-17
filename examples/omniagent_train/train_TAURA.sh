#!/usr/bin/env bash
set -e
set -x
ENGINE="${ENGINE:-vllm}"
if [[ $# -gt 0 && "$1" != *=* && "$1" != +* && "$1" != -* ]]; then
    ENGINE="$1"
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USAGE_DISABLE=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# ====== Environment Variables ======
# export WANDB_BASE_URL=""  # Set to your WandB server URL if using self-hosted instance
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# ====== Multi-Node Distributed Config ======
export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}
export RAY_num_server_call_thread=1

# ====== No Changes Needed ======
export USE_OSS_IN_VIDEOENV=False
export FORCE_QWENVL_VIDEO_READER=torchcodec
export FFMPEG_PATH=ffmpeg
export OSS_PATH=agentic_tmp/
export OSS_BUCKET="${OSS_BUCKET:-}"
export VIDEO_ENV_AUTO_CLEANUP=False
export VIDEO_ENV_TMP_DIR="./video_env_tmp"
pool_async_enabled=True
ignore_exceed=False
# export FORMAT_REWARD=False
# export REASONING_REWARD=False
use_invalid_action_penalty=False
# Enable bonus and set parameters
# export LEN_BONUS_ENABLE=false
# export LEN_BETA=0.007
# export LEN_BONUS_MAX=2.0

# ====== Training Data Config ======
train_data_size=32                                                      # prompts per training batch before rollout expansion
val_data_size=8                                                         # prompts per validation batch before rollout expansion
N=8                                                                     # rollout copies per prompt; train envs = train_data_size * N, val envs <= val_data_size * N
actor_rollout_ref_N=1                                                   #   to fake actor_rollout_ref.rollout.n for [ppo_mini_batch_size] calculation

mini_batch_size=16                                                      #   [! real_mini_bs=4*N=32 ! manually compensate for fake actor_rollout_ref_N on mini batch size normalization] mini_batch_size/num_of_gpus  16/16=1, self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
ppo_micro_batch_size_per_gpu=1                                          # actor - fsdp
rollout_log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu # rollout generate_sequence  - vllm
ref_log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu     # ref - fsdp
# Rollout generation chunking:
# max alive envs per wave = total GPU count * MICRO_RATIO.
# Larger rollout batches are split into multiple waves; lower MICRO_RATIO for A100 memory headroom.
export MICRO_RATIO=2
export ABS_ON_POLICY=True
# ------------------------------------------------------------
# If ABS_ON_POLICY is enabled, recompute mini_batch_size
# ------------------------------------------------------------
if [[ "${ABS_ON_POLICY,,}" == "true" || "${ABS_ON_POLICY}" == "1" ]]; then
    world_size=$(( GPUS_PER_NODE * NNODES ))                # local variable only
    mini_batch_size=$(( world_size * ppo_micro_batch_size_per_gpu ))
fi
echo "MICRO_RATIO   = ${MICRO_RATIO}"
echo "ABS_ON_POLICY   = ${ABS_ON_POLICY}"

#DAPO
clip_ratio_low=0.2
clip_ratio_high=0.3
enable_filter_groups=False
max_num_gen_batches=1
entropy_coeff=0.000
use_kl_loss=False

TP=1                            #   tensor_model_parallel_size
SP=None                         #   sequence parallel_size
max_prompt_length=65536         #   for data, also for rollout - prompt_length: ${data.max_prompt_length}
max_response_length=4096        #   for data, also for rollout - response_length: ${data.max_response_length}
max_num_batched_tokens=131072   #   for vllm

export DELETE_MEDIA_WHEN_ERROR=True
export STRICT_CONFIDENCE_CHECK=True
export USE_DYNAMIC_STEP=True
# export RANDOM_RESOURCE_SFT=True
# export BYPASS_DURATION_CHECK=True
export RETRY_ON_FORMAT_ERROR=True  # Enable retry and trajectory filtering mode

max_steps=22
export MIN_MAX_STEPS=5
max_frames_len=60
max_audio_len=300
max_clip_len=60
rollout_max_images_len=$max_frames_len   # vllm
rollout_max_videos_len=1                 # vllm
rollout_max_audios_len=1                 # vllm
loss_agg_mode=token-mean
val_top_k=20
val_top_p=0.95
val_do_sample=True

bypass_mode=false
rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null
rollout_rs_threshold=2.0
use_policy_gradient=true

entropy_top_ratio=null          # disable token-level top-k; set to 0.2 etc. to enable
entropy_soft_weight=false       # disable soft-weight; set to true to enable
entropy_weight_temperature=1.0  # only used when soft-weight is enabled
seq_top_ratio=null               # sequence-level hard top-k, ratio 0.2
entropy_weight=true             # enable entropy weight; set to true to enable

PROJECT_NAME=verl_omni
MODEL_NAME=OmniAgent-SFT-7B
MODE=OmniAgent
TRAIN_DATA_NAME=train_RL
VAL_DATA_NAME=VideoMME_Long

# === Data & Model Paths (override via env vars) ===
TRAIN_FILE="${TRAIN_FILE:-/path/to/train_data.jsonl}"
VAL_FILE="${VAL_FILE:-/path/to/val_data.jsonl}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/path/to/models}"

experiment_name=Env2_${MODE}_${TRAIN_DATA_NAME}_${MODEL_NAME}_${train_data_size}_${mini_batch_size}_n${N}_steps_${MIN_MAX_STEPS}-${max_steps}_${max_frames_len}_${max_clip_len}_${max_audio_len}_entropy_${entropy_coeff}_bypass_${bypass_mode}_dyn-step_${USE_DYNAMIC_STEP}_ReTry${RETRY_ON_FORMAT_ERROR}_seq_top_ratio${seq_top_ratio}_ent_weight_${entropy_weight}_LR_1e-6
rollout_data_dir=./log_rollout/log_rollout_train/${PROJECT_NAME}/${experiment_name}
val_data_dir=./log_rollout/log_rollout_val/${PROJECT_NAME}/${experiment_name}
export TENSORBOARD_DIR=./tensorboard/${PROJECT_NAME}/${experiment_name}
mkdir -p logs

# === DRY_RUN support ===
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY_RUN] Would run: python3 -m verl.trainer.main_ppo with experiment=${experiment_name}"
    echo "[DRY_RUN] TRAIN_FILE=${TRAIN_FILE}"
    echo "[DRY_RUN] VAL_FILE=${VAL_FILE}"
    echo "[DRY_RUN] MODEL_PATH=${MODEL_BASE_PATH}/${MODEL_NAME}"
    echo "[DRY_RUN] Exiting without launching training."
    exit 0
fi

# ====== Clean Up Old Ray Processes ======
# || true prevents script exit when no processes to kill
ray stop --force > /dev/null 2>&1 || true
pkill -9 -f verl.trainer.main_ppo || true
pkill -9 -f ray || true
rm -rf /tmp/ray/*
sleep 5 

# ====== Head Node: Start Ray and Launch Training ======
if [ "$NODE_RANK" -eq 0 ]; then
    echo "[Head Node] Starting Ray Head, waiting for Workers to join..."
    ray start --block --head --node-ip-address=${MASTER_ADDR} --port=${MASTER_PORT} --num-gpus=${GPUS_PER_NODE} --num-cpus=88 &

    # ====== Wait for Workers to Join ======
    echo "Waiting 30 seconds for all Worker nodes to connect and register GPUs..."
    sleep 30
    ray status # print status for log verification
    # ========================================

    set +e
    echo "[Head Node] Launching training..."

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.bypass_mode=$bypass_mode \
        algorithm.rollout_correction.use_policy_gradient=$use_policy_gradient \
        algorithm.rollout_correction.rollout_is=$rollout_is \
        algorithm.rollout_correction.rollout_is_threshold=$rollout_is_threshold \
        algorithm.rollout_correction.rollout_rs=$rollout_rs \
        algorithm.rollout_correction.rollout_rs_threshold=$rollout_rs_threshold \
        "data.train_files=${TRAIN_FILE}" \
        "data.val_files=${VAL_FILE}" \
        data.train_batch_size=$train_data_size \
        data.val_batch_size=$val_data_size \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=$max_response_length \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        data.seed=42 \
        +actor_rollout_ref.actor.ignore_exceed=${ignore_exceed} \
        "actor_rollout_ref.model.path=${MODEL_BASE_PATH}/${MODEL_NAME}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.warmup_style=constant \
        actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
        actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
        actor_rollout_ref.actor.kl_loss_coef=0.00 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$rollout_log_prob_micro_batch_size_per_gpu \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enable_prefix_caching=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=$val_do_sample \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
        actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
        actor_rollout_ref.rollout.n=$actor_rollout_ref_N \
        +actor_rollout_ref.rollout.limit_images=$rollout_max_images_len \
        +actor_rollout_ref.rollout.limit_videos=$rollout_max_videos_len \
        +actor_rollout_ref.rollout.limit_audios=$rollout_max_audios_len \
        +actor_rollout_ref.rollout.max_model_len=$max_num_batched_tokens \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ref_log_prob_micro_batch_size_per_gpu \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        +actor_rollout_ref.actor.use_invalid_action_penalty=$use_invalid_action_penalty \
        +actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
        algorithm.use_kl_in_reward=False \
        +algorithm.filter_groups.enable=${enable_filter_groups} \
        +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
        +algorithm.entropy_top_ratio=${entropy_top_ratio} \
        +algorithm.entropy_soft_weight=${entropy_soft_weight} \
        +algorithm.entropy_weight_temperature=${entropy_weight_temperature} \
        +algorithm.entropy_weight=${entropy_weight} \
        +algorithm.entropy_seq_top_ratio=${seq_top_ratio} \
        reward_model.reward_manager=episode \
        +env.env_name=video_env \
        +env.seed=42 \
        +env.rollout.n=$N \
        +env.rollout.pool_async_enabled=${pool_async_enabled} \
        +env.max_steps=$max_steps \
        +env.video_star.max_frames_len=$max_frames_len \
        +env.video_star.max_audio_len=$max_audio_len \
        +env.video_star.max_clip_len=$max_clip_len \
        +env.video_star.mode=$MODE \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb','tensorboard'] \
        trainer.project_name=$PROJECT_NAME \
        trainer.rollout_data_dir=$rollout_data_dir \
        trainer.validation_data_dir=$val_data_dir \
        trainer.experiment_name=$experiment_name \
        trainer.n_gpus_per_node=${GPUS_PER_NODE} \
        trainer.nnodes=${NNODES} \
        trainer.save_freq=10 \
        trainer.test_freq=5000 \
        trainer.total_epochs=1 \
        trainer.val_before_train=False "$@" \
        2>&1 | tee -a "logs/${experiment_name}.log" 

    set -e # restore set -e

    echo "Training finished, shutting down cluster..."
    ray stop --force

# ====== Worker Node: Join Ray Cluster Only ======
else
    echo "[Worker Node] Connecting to Head: ${MASTER_ADDR}:${MASTER_PORT} ..."
    # Worker blocks here until Head executes ray stop
    ray start --block --address=${MASTER_ADDR}:${MASTER_PORT} --num-gpus=${GPUS_PER_NODE}
    echo "[Worker Node] Cluster shut down, Worker exiting."
fi

# Final cleanup as a safety net
ray stop --force > /dev/null 2>&1 || true
