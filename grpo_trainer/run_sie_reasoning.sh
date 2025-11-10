# #!/usr/bin/env bash

# set -euo pipefail
# set -x

# ########################################
# # 0. 路径配置
# ########################################

# MODEL_PATH=${MODEL_PATH:-/data/models/Qwen/Qwen2-0.5B-Instruct}

# # SIE 数据路径 - 使用转换后的 verl 格式
# SIE_TRAIN=/data/sunwenhe/SIE_reasoning/verl_data/verl_cwq_sie100.parquet
# SIE_VAL=/data/sunwenhe/SIE_reasoning/verl_data/verl_cwq_sie100.parquet

# EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_0p5b_cwq_sie100_grpo}

# # 添加自定义 reward 路径到 PYTHONPATH
# export PYTHONPATH=/data/sunwenhe/SIE_reasoning/prepare_SIE_data:${PYTHONPATH:-}

# ########################################
# # 1. 检查数据文件是否存在
# ########################################
# if [[ ! -f "$SIE_TRAIN" ]]; then
#   echo "[ERROR] Training file not found: $SIE_TRAIN"
#   echo "Please run convert_to_verl.py first!"
#   exit 1
# fi

# if [[ ! -f "$SIE_VAL" ]]; then
#   echo "[ERROR] Validation file not found: $SIE_VAL"
#   echo "Please run convert_to_verl.py first!"
#   exit 1
# fi

# echo "[INFO] Using training data: $SIE_TRAIN"
# echo "[INFO] Using validation data: $SIE_VAL"

# ########################################
# # 2. 运行 verl GRPO 训练
# ########################################
# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     \
#     data.train_files=${SIE_TRAIN} \
#     data.val_files=${SIE_VAL} \
#     data.train_batch_size=4 \
#     data.max_prompt_length=1024 \
#     data.max_response_length=512 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     \
#     actor_rollout_ref.model.path=${MODEL_PATH} \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.optim.lr=2e-6 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=4 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0.0 \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.actor.use_torch_compile=False \
#     \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.n=1 \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
#     actor_rollout_ref.rollout.enable_chunked_prefill=False \
#     \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     actor_rollout_ref.ref.use_torch_compile=False \
#     \
#     algorithm.kl_ctrl.kl_coef=0.001 \
#     algorithm.use_kl_in_reward=False \
#     \
#     reward_model.enable=True \
#     +reward_model.input_tokenizer=${MODEL_PATH} \
#     +reward_model.external_reward_function='sie_reward_verl:sie_reward_function' \
#     reward_model.micro_batch_size_per_gpu=1 \
#     \
#     trainer.critic_warmup=0 \
#     trainer.logger='["console","tensorboard"]' \
#     trainer.project_name='verl_cwq_sie' \
#     trainer.experiment_name=${EXPERIMENT_NAME} \
#     trainer.n_gpus_per_node=1 \
#     trainer.nnodes=1 \
#     trainer.save_freq=50 \
#     trainer.test_freq=50 \
#     trainer.total_epochs=1 \
#     trainer.total_training_steps=200 \
#     "$@"



# #!/usr/bin/env bash

# set -euo pipefail
# set -x

# export WANDB_API_KEY="6159ab5867c053d3ac643f895c608add2dd04f8c"
# ########################################
# # 0. 路径配置
# ########################################

# # 换成你本地的 7B 路径
# MODEL_PATH=${MODEL_PATH:-/data/models/Qwen/Qwen2.5-7B-Instruct}
# # MODEL_PATH=${MODEL_PATH:-/data/models/Qwen/Qwen2-0.5B-Instruct}


# # 用你已经转成 verl 的 parquet（17k 那份）
# SIE_SRC=/data/sunwenhe/SIE_reasoning/verl_data/verl_cwq_sie100.parquet
# SIE_TRAIN=/data/sunwenhe/SIE_reasoning/verl_data/verl_cwq_sie100-train.parquet
# SIE_VAL=/data/sunwenhe/SIE_reasoning/verl_data/verl_cwq_sie100-val.parquet

# EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2p5_7b_cwq_sie_grpo}
# # EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_0p5b_cwq_sie_grpo}

# # 自定义 reward 路径
# export PYTHONPATH=/data/sunwenhe/SIE_reasoning/prepare_SIE_data:${PYTHONPATH:-}

# ########################################
# # 1. 9:1 切 train / val（只切一次）
# ########################################
# if [[ ! -f "$SIE_TRAIN" || ! -f "$SIE_VAL" ]]; then
#   echo "[INFO] split parquet 9:1 -> train/val"
#   python - << 'PY'
# import pyarrow.parquet as pq

# src = "/data/sunwenhe/SIE_reasoning/verl_data/verl_cwq_sie100.parquet"
# train_out = "/data/sunwenhe/SIE_reasoning/verl_data/verl_cwq_sie100-train.parquet"
# val_out = "/data/sunwenhe/SIE_reasoning/verl_data/verl_cwq_sie100-val.parquet"

# table = pq.read_table(src)
# n = table.num_rows
# val_n = max(1, n // 10)
# train_n = n - val_n

# pq.write_table(table.slice(0, train_n), train_out)
# pq.write_table(table.slice(train_n, val_n), val_out)

# print(f"[split] total={n}, train={train_n}, val={val_n}")
# PY
# fi

# echo "[INFO] Using training data: $SIE_TRAIN"
# echo "[INFO] Using validation data: $SIE_VAL"

# ########################################
# # 2. GRPO 训练（8×H20, 7B）
# ########################################
# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     \
#     data.train_files=${SIE_TRAIN} \
#     data.val_files=${SIE_VAL} \
#     data.train_batch_size=128 \
#     data.max_prompt_length=2048 \
#     data.max_response_length=512 \
#     data.filter_overlong_prompts=True \
#     data.truncation=error \
#     \
#     actor_rollout_ref.model.path=${MODEL_PATH} \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.optim.lr=8e-7 \
#     actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
#     actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
#     \
#     actor_rollout_ref.actor.ppo_mini_batch_size=128 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.actor.ppo_epochs=1 \
#     \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0.0 \
#     \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.actor.use_torch_compile=False \
#     \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.n=1 \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.prompt_length=2048 \
#     actor_rollout_ref.rollout.response_length=512 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
#     actor_rollout_ref.rollout.enable_chunked_prefill=False \
#     \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     actor_rollout_ref.ref.use_torch_compile=False \
#     \
#     algorithm.kl_ctrl.kl_coef=0.001 \
#     algorithm.use_kl_in_reward=False \
#     \
#     reward_model.enable=False \
#     custom_reward_function.path=/data/sunwenhe/SIE_reasoning/prepare_SIE_data/sie_reward_verl.py \
#     custom_reward_function.name=my_sie_reward   \
#     \
#     trainer.critic_warmup=0 \
#     trainer.logger='["console","wandb"]' \
#     trainer.project_name='verl_cwq_sie' \
#     trainer.experiment_name=${EXPERIMENT_NAME} \
#     trainer.n_gpus_per_node=4 \
#     trainer.nnodes=1 \
#     trainer.total_epochs=1 \
#     trainer.save_freq=100 \
#     trainer.test_freq=100 \
#     "$@"



#!/usr/bin/env bash

set -euo pipefail
set -x

# 你的 wandb key
export WANDB_API_KEY="6159ab5867c053d3ac643f895c608add2dd04f8c"

########################################
# 0. 固定路径配置（不会变的）
########################################

# 模型
MODEL_PATH=${MODEL_PATH:-/data/models/Qwen/Qwen2.5-7B-Instruct}


python - << 'PY'
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path

val_path = Path("/data/sunwenhe/SIE_reasoning/verl_data/cwq_0_validation.parquet")
out_path = Path("/data/sunwenhe/SIE_reasoning/verl_data/cwq_0_validation.max4096.parquet")

table = pq.read_table(val_path)

# 这里按字符长度筛一遍，保守一点设成 12000 字符
# （一般 4k token 左右，大多数都会落在这个范围）
prompt_col = table["prompt"]
mask = []
for s in prompt_col.to_pylist():
    text = s if isinstance(s, str) else str(s)
    mask.append(len(text) <= 12000)

filtered = table.filter(pa.array(mask))
pq.write_table(filtered, out_path)
print(f"filtered val: {len(filtered)} rows -> {out_path}")
PY


# 验证集：你说除了 training set 其他都不变，所以 val 不动
SIE_VAL=/data/sunwenhe/SIE_reasoning/verl_data/cwq_0_validation.max4096.parquet

# 自定义 reward 路径
export PYTHONPATH=/data/sunwenhe/SIE_reasoning/prepare_SIE_data:${PYTHONPATH:-}

# 要跑的几个 ratio
RATIOS=("0" "25" "50" "75" "100")

########################################
# 循环跑 5 次
########################################
for R in "${RATIOS[@]}"; do
  echo "=============================="
  echo "[INFO] start training for ratio=${R}"
  echo "=============================="

  # 这一轮用的 training parquet
  SIE_TRAIN=/data/sunwenhe/SIE_reasoning/verl_data/sie_${R}.parquet

  # 给这轮起一个不一样的实验名，方便区分
  EXPERIMENT_NAME=qwen2p5_7b_cwq_sie_grpo_r${R}

  echo "[INFO] Using training data: $SIE_TRAIN"
  echo "[INFO] Using validation data: $SIE_VAL"
  echo "[INFO] Experiment name: $EXPERIMENT_NAME"

  python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      \
      data.train_files=${SIE_TRAIN} \
      data.val_files=${SIE_VAL} \
      data.train_batch_size=128 \
      data.max_prompt_length=8192 \
      data.max_response_length=2048 \
      data.filter_overlong_prompts=True \
      data.truncation=error \
      \
      actor_rollout_ref.model.path=${MODEL_PATH} \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.optim.lr=8e-7 \
      actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
      actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
      \
      actor_rollout_ref.actor.ppo_mini_batch_size=128 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
      actor_rollout_ref.actor.ppo_epochs=2 \
      \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.kl_loss_coef=0.001 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.actor.entropy_coeff=0.0 \
      \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.actor.use_torch_compile=False \
      \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.n=1 \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
      actor_rollout_ref.rollout.max_model_len=4608 \
      actor_rollout_ref.rollout.prompt_length=4096 \
      actor_rollout_ref.rollout.response_length=512 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
      actor_rollout_ref.rollout.enable_chunked_prefill=False \
      \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      actor_rollout_ref.ref.use_torch_compile=False \
      \
      algorithm.kl_ctrl.kl_coef=0.001 \
      algorithm.use_kl_in_reward=False \
      \
      reward_model.enable=False \
      custom_reward_function.path=/data/sunwenhe/SIE_reasoning/prepare_SIE_data/sie_reward_verl.py \
      custom_reward_function.name=my_sie_reward \
      \
      trainer.critic_warmup=0 \
      trainer.logger='["console","wandb"]' \
      trainer.project_name='verl_cwq_sie' \
      trainer.experiment_name=${EXPERIMENT_NAME} \
      trainer.n_gpus_per_node=4 \
      trainer.nnodes=1 \
      trainer.total_epochs=2 \
      trainer.save_freq=100 \
      trainer.test_freq=100
done