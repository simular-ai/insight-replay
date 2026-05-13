#!/usr/bin/python
# =============================================================================
# Multi-node cluster launcher (InsightReplay):
#   Qwen3-4B-Base + DAPO + DAPO-Math-15k + rollout-time InsightReplay
#   injection, 128 GPUs (16 nodes x 8 H800).
#
# Diff vs baseline_cluster.py:
#   - exp_name
#   - one config line in the training command:
#         actor_rollout_ref.rollout.agent.default_agent_loop=insight_replay_agent
# Mechanism is implemented in
#   training/verl/verl/experimental/agent_loop/insight_replay_agent_loop.py.
#
# This is the exact recipe used for the InsightReplay runs in the paper. It is
# wired up to a Kubernetes-style RL job submission SDK (`pypai` + the
# `aistudio_common` data-store helper) that is **not** part of this release.
# The SDK calls below are illustrative — to reproduce on your own cluster,
# replace the `RLJobBuilder(...)` block at the bottom with whatever your
# scheduler exposes (e.g. `kubectl apply -f ...`, Slurm sbatch, Ray Job
# Submission API, SkyPilot, etc.). The `training_command` string itself is
# portable: it is the same `python3 -m verl.trainer.main_ppo ...` invocation
# you would run by hand.
#
# Hyperparameters track the single-node `run_insightreplay.sh` recipe, scaled
# from 1 to 16 nodes.
# =============================================================================
import base64

# --- Cluster job-submission SDK (proprietary; substitute on your cluster) -----
# These imports name the abstractions used at submission time. The actual API
# they came from is internal infrastructure and not bundled here. Adapt the
# `RLJobBuilder(...)` call at the bottom of the file to your scheduler.
from pypai.job import RLJobBuilder
from pypai.conf import ExecConf, KMConf, CodeRepoConf
from pypai.conf.retry_strategy import RetryStrategy, RetryPolicy
from aistudio_common.openapi.models.data_store import DataStore
# -----------------------------------------------------------------------------

# Path inside the training container where verl is checked out / installed.
PATH_TO_VERL = "/workspace/training/verl"

project_name = "InsightReplay"
exp_name = "InsightReplay-Qwen3-4B-Math-128gpu"
TASK_NAME = f"{project_name}-{exp_name}"

# ==========================================
# Distributed topology (128 GPUs)
# ==========================================
TOTAL_GPU = 128
GPUS_PER_NODE = 8
NNODES = TOTAL_GPU // GPUS_PER_NODE  # = 16 nodes

# Parallelism
SP_SIZE = 2         # Ulysses sequence parallel; DP = 128 / SP_SIZE / FSDP_SIZE
ROLLOUT_TP = 1      # vLLM tensor parallel; # of vLLM instances = 128 / ROLLOUT_TP
FSDP_SIZE = 8

# Batch sizes (scaled up from single-node 64 → 256 for 128 GPUs)
TRAIN_BATCH_SIZE = 256
ROLLOUT_N = 8                 # trajectories per prompt
PPO_MINI_BATCH_SIZE = 128

# ==========================================
# Sequence length config (aligned with run_insightreplay.sh)
# ==========================================
MAX_PROMPT_LENGTH = 2048
MAX_RESPONSE_LENGTH = 30720
ROLLOUT_MAX_NUM_BATCHED_TOKENS = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH  # 32768

# Token-length cap for dynamic batch size (aligned with single-node)
ACTOR_PPO_MAX_TOKEN_LEN = (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 1
INFER_PPO_MAX_TOKEN_LEN = (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2

# ==========================================
# DAPO soft overlong penalty (aligned with single-node)
# ==========================================
OVERLONG_BUFFER_LEN = 4096
OVERLONG_PENALTY_FACTOR = 1.0

# ==========================================
# Paths — substitute with your shared NAS / object store layout
# ==========================================
SHARED_NAS = "/mnt/shared"  # mount point of a shared filesystem on all nodes
DATA_ROOT = f"{SHARED_NAS}/data/math"
TRAIN_FILE = f"{DATA_ROOT}/dapo-math-15k-rl.parquet"
VAL_FILE = f"{DATA_ROOT}/aime-2025-verl.parquet"

MODEL_PATH = f"{SHARED_NAS}/models/Qwen3-4B-Base"

BASE_NAS_DIR = f"{SHARED_NAS}/ckpts/{project_name}/{exp_name}"
LOG_DIR = f"{BASE_NAS_DIR}/logs"
OUTPUT_DIR = f"{BASE_NAS_DIR}/outputs"
CHECKPOINT_DIR = f"{BASE_NAS_DIR}/checkpoints"

# ==========================================
# Container image and node specs (placeholders — fill in for your registry)
# ==========================================
IMAGE_URL = "<your-registry>/<your-image>:<tag>"
master_conf = ExecConf(cpu=132, memory=2048576, disk_m=2048576, gpu_num=GPUS_PER_NODE, num=1, gpu_type="h800", shared_memory=1048576)
worker_conf = ExecConf(cpu=132, memory=2048576, disk_m=2048576, gpu_num=GPUS_PER_NODE, num=NNODES - 1, gpu_type="h800", shared_memory=1048576)

rs = RetryStrategy(retry_policy=RetryPolicy.ON_FAILURE, max_attempt=1)
km_conf = KMConf(image=IMAGE_URL, retry_strategy=rs)

# Shared NAS mount the scheduler should attach to every worker.
shared_store = DataStore(mount_point=SHARED_NAS, store_name="<your-store-name>", sub_path=SHARED_NAS)

# Git checkout of the training code that lands inside each container.
code_repo_config = CodeRepoConf(
    repo_url="<your-git-repo-url>",
    branch="main",
)

env_vars = {
    "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=6400000",
    "RAY_memory_monitor_refresh_ms": "0",
    "RAY_memory_usage_threshold": "0.99",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "PYTHONPATH": f"{PATH_TO_VERL}:$PYTHONPATH",
    "TENSORBOARD_DIR": "/home/admin/logs/tfevent",
    "OMP_NUM_THREADS": "1",
    "RAY_DEDUP_LOGS": "1",
    "NCCL_TIMEOUT": "1800000",
    "VLLM_NCCL_SOK_DISABLE": "1",
    "NCCL_DEBUG": "WARN",
    "TORCH_NCCL_BLOCKING_WAIT": "1",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "VLLM_FLASH_ATTN_VERSION": "2",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTORCH_ALLOC_CONF": "expandable_segments:False",
    "VLLM_CUSTOM_ALLREDUCE": "0",
}

# Log sync — background script that copies in-container dialogue logs into the
# shared NAS every minute so a single rank's logs don't get lost when its pod
# is restarted.
sync_script = f"""
while true; do
for d in $(find /tmp /root /home /workspace -type d -name 'dialogue_logs*' 2>/dev/null | grep -v '/mnt/'); do
b=$(basename "$d")
mkdir -p "{OUTPUT_DIR}/$b"
cp -ru "$d"/* "{OUTPUT_DIR}/$b/" 2>/dev/null || true
done
sleep 60
done
"""
b64_sync = base64.b64encode(sync_script.encode("utf-8")).decode("utf-8")

init_cmd = (
    f"set -e; cd {PATH_TO_VERL}; "
    f"mkdir -p {OUTPUT_DIR} {CHECKPOINT_DIR} {LOG_DIR}; "
    f"rm -rf {PATH_TO_VERL}/outputs; ln -s {OUTPUT_DIR} {PATH_TO_VERL}/outputs; "
    f"echo '{b64_sync}' | base64 -d > /tmp/sync_logs.sh; "
    f"nohup bash /tmp/sync_logs.sh > /dev/null 2>&1 &"
)

# Training command — matches single-node hyperparams, FSDP2, dynamic batch size.
# Diff vs baseline: adds actor_rollout_ref.rollout.agent.default_agent_loop=insight_replay_agent
training_command = f"""
set -o pipefail;
ulimit -n 65535;
mkdir -p /home/admin/logs/tfevent &&
nohup bash -c "while true; do cp -ru /home/admin/logs/tfevent/* {LOG_DIR}/ 2>/dev/null || true; sleep 60; done" > /dev/null 2>&1 &
python3 -m verl.trainer.main_ppo \
data.train_files="{TRAIN_FILE}" \
data.val_files="{VAL_FILE}" \
data.prompt_key=prompt \
data.truncation='left' \
data.max_prompt_length={MAX_PROMPT_LENGTH} \
data.max_response_length={MAX_RESPONSE_LENGTH} \
data.train_batch_size={TRAIN_BATCH_SIZE} \
data.shuffle=True \
actor_rollout_ref.rollout.n={ROLLOUT_N} \
algorithm.adv_estimator=grpo \
algorithm.use_kl_in_reward=False \
algorithm.kl_ctrl.kl_coef=0.0 \
actor_rollout_ref.actor.use_kl_loss=False \
actor_rollout_ref.actor.kl_loss_coef=0.0 \
actor_rollout_ref.actor.clip_ratio_low=0.2 \
actor_rollout_ref.actor.clip_ratio_high=0.28 \
actor_rollout_ref.model.path="{MODEL_PATH}" \
actor_rollout_ref.model.trust_remote_code=True \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.actor.optim.weight_decay=0.1 \
actor_rollout_ref.actor.ppo_mini_batch_size={PPO_MINI_BATCH_SIZE} \
actor_rollout_ref.actor.entropy_coeff=0 \
actor_rollout_ref.actor.grad_clip=1.0 \
actor_rollout_ref.actor.loss_agg_mode="token-mean" \
actor_rollout_ref.actor.use_torch_compile=False \
actor_rollout_ref.actor.strategy=fsdp2 \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ulysses_sequence_parallel_size={SP_SIZE} \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu={ACTOR_PPO_MAX_TOKEN_LEN} \
actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
actor_rollout_ref.actor.fsdp_config.fsdp_size={FSDP_SIZE} \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
actor_rollout_ref.actor.fsdp_config.offload_policy=False \
actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
actor_rollout_ref.actor.checkpoint.save_contents="['model', 'optimizer', 'extra']" \
actor_rollout_ref.ref.strategy=fsdp2 \
actor_rollout_ref.ref.use_torch_compile=False \
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={INFER_PPO_MAX_TOKEN_LEN} \
actor_rollout_ref.ref.ulysses_sequence_parallel_size={SP_SIZE} \
actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
actor_rollout_ref.ref.fsdp_config.param_offload=False \
actor_rollout_ref.ref.fsdp_config.offload_policy=False \
actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.ignore_eos=False \
actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
actor_rollout_ref.rollout.tensor_model_parallel_size={ROLLOUT_TP} \
actor_rollout_ref.rollout.enable_chunked_prefill=True \
actor_rollout_ref.rollout.max_num_batched_tokens={ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.enable_prefix_caching=False \
actor_rollout_ref.rollout.free_cache_engine=True \
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={INFER_PPO_MAX_TOKEN_LEN} \
actor_rollout_ref.rollout.temperature=1.0 \
actor_rollout_ref.rollout.top_p=1.0 \
actor_rollout_ref.rollout.top_k=-1 \
actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
actor_rollout_ref.rollout.val_kwargs.do_sample=True \
actor_rollout_ref.rollout.val_kwargs.n=32 \
actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144 \
actor_rollout_ref.rollout.agent.default_agent_loop=insight_replay_agent \
reward.reward_manager.name=dapo \
+reward.reward_kwargs.overlong_buffer_cfg.enable=True \
+reward.reward_kwargs.overlong_buffer_cfg.len={OVERLONG_BUFFER_LEN} \
+reward.reward_kwargs.overlong_buffer_cfg.penalty_factor={OVERLONG_PENALTY_FACTOR} \
+reward.reward_kwargs.overlong_buffer_cfg.log=False \
+reward.reward_kwargs.max_resp_len={MAX_RESPONSE_LENGTH} \
trainer.critic_warmup=0 \
trainer.logger="['console','tensorboard']" \
trainer.project_name="{project_name}" \
trainer.experiment_name="{exp_name}" \
trainer.n_gpus_per_node="{GPUS_PER_NODE}" \
trainer.nnodes="{NNODES}" \
trainer.balance_batch=False \
trainer.val_before_train=True \
trainer.test_freq=20 \
trainer.save_freq=50 \
trainer.total_epochs=10 \
trainer.default_local_dir="{CHECKPOINT_DIR}" \
trainer.resume_mode=auto \
trainer.log_val_generations=10 \
$@ 2>&1 | tee {OUTPUT_DIR}/run_math_128gpu.log
"""

cd_cmd = f"cd {PATH_TO_VERL}"
training_command = " ".join(line for line in training_command.strip().split() if line.strip())
command = f"pip list && {cd_cmd} && {training_command}"

# --- Submit the job ---------------------------------------------------------
# Replace this block with the equivalent for your cluster's scheduler.
job = RLJobBuilder(
    name=TASK_NAME,
    working_dir=f"{PATH_TO_VERL}",
    source_root="",
    main_file="",
    command=command,
    init_command=init_cmd,
    master=master_conf,
    worker=worker_conf,
    km_conf=km_conf,
    k8s_app_name="<your-k8s-app>",
    k8s_priority="high",
    host_network=True,
    rdma=True,
    envs=env_vars,
    code_repo_configs=[code_repo_config],
    data_stores=[shared_store],
    object_store_memory=str(500 * 1024 * 1024 * 1024),
    tag=f"type=grpo,basemodel=Qwen3-4B,dev_pattern=verl,exp_name={exp_name},project_name={project_name}",
).run(enable_wait=False)

print("Task submitted. Monitor it through your cluster's job dashboard.")
