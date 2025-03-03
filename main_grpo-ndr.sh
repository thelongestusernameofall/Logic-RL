#!/bin/bash
#
#   Author  :   simon huang
#   Date    :   2025年02月25日14:20:30
#   
#   For GRPO Training on NDR analysis job. 
#

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_DIR=wandb-ndr-r1-7b-v1.0
export WANDB_PROJECT=GRPO_LgRL-simon-ndr

# wandb server start --port 9090
export WANDB_BASE_URL=http://10.176.205.21:9000
export WANDB_API_KEY=local-5239e89783ebebea9bac5509e2bd1a8e734f55f7
# wandb login --relogin --host=http://10.176.205.21:9000
# export WANDB_MODE=offline


set -x
MODEL_PATH=/data1/models/Deepseek/DeepSeek-R1-Distill-Qwen-7B
export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/ndr/parquet/train.parquet \
    data.val_files=data/ndr/parquet/val.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=6200 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='GRPO_LgRL-simon-ndr' \
    trainer.experiment_name='NDR-R1-7B' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=ndrr17b \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=5 $@ 2>&1 | tee grpo-ndr.log
