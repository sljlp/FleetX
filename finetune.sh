
set -uex

set_gpu_config(){
    gpu=gpu

}

set_dcu_config(){
    gpu=gpu
}

set_xpu_config(){
    gpu=xpu
    export PYTHONPATH=/Paddle/Paddle/build/python/:/Paddle/fleetx/PaddleSlim:$PYTHONPATH
    export BKCL_PCIE_RING=1
}

set_npu_config(){
    gpu=npu
}

set_xpu_config
npu-smi info && set_npu_config || true
nvidia-smi && set_gpu_config || true
rocm-smi && set_dcu_config || true
device=$gpu

finetune_345M_sharding_dp(){
  sharding_stage=1
  [[ $sharding_degree -gt 1 ]] && sharding_stage=2
  launch_opt="--log_dir log_345M --device ${DEVICES}" \
  exported_opt="-o Engine.mix_precision.use_pure_fp16=${use_fp16}
  -o Distributed.sharding.sharding_degree=${sharding_degree}
  -o Distributed.sharding.sharding_stage=${sharding_stage}
  -o Distributed.dp_degree=${dp_degree}
  -o Data.Train.sampler.name=GPTBatchSampler
  -o Data.Eval.sampler.name=GPTBatchSampler
  -o Global.micro_batch_size=${micro_batch_size}
  -o Global.device=${device}
  -o Model.use_recompute=${use_recompute}" \
  bash projects/gpt/dist_finetune_gpt_345M_single_card.sh $JOB
}

# default micro_batch_size 32
# fp32
fp32_dp(){
    JOB=$JOB \
    micro_batch_size=32 \
    use_recompute=False \
    DEVICES=0,1,2,3 \
    log=log_fp32_dp \
    use_fp16=False \
    sharding_degree=1 \
    dp_degree=4 \
    finetune_345M_sharding_dp
}

fp32_sharding(){
    JOB=$JOB \
    micro_batch_size=32 \
    use_recompute=False \
    DEVICES=0,1,2,3 \
    log=log_fp32_sharding \
    use_fp16=False \
    sharding_degree=4 \
    dp_degree=1 \
    finetune_345M_sharding_dp
}


fp32_sharding_dp(){
    JOB=$JOB \
    micro_batch_size=32 \
    use_recompute=False \
    DEVICES=0,1,2,3 \
    log=log_sharding_dp \
    use_fp16=False \
    sharding_degree=2 \
    dp_degree=2 \
    finetune_345M_sharding_dp
}

fp32_sharding_dp_recompute(){
    JOB=$JOB \
    micro_batch_size=32 \
    use_recompute=True \
    DEVICES=4,5,6,7 \
    log=log_sharding_dp_rec \
    use_fp16=False \
    sharding_degree=2 \
    dp_degree=2 \
    finetune_345M_sharding_dp
}

fp16_sharding_dp_recompute(){
    JOB=$JOB \
    micro_batch_size=32 \
    use_recompute=True \
    DEVICES=0,1,2,3 \
    log=log_fp16_sharding_dp_rec \
    use_fp16=True \
    sharding_degree=2 \
    dp_degree=2 \
    finetune_345M_sharding_dp
}

export JOB=SST2
# 1.5 step/s, total step: 2104 step/epoch *3 epch 4卡 总耗时 1.2h
# fp32_dp

# 1.44 step/s, total step: 2104 step/epoch *3 epch 4卡
# fp32_sharding

# # 1.32 step/s, total step: 2104 step/epoch *3 epch 4卡
fp32_sharding_dp

# 1.31 step/s, total step: 2104 step/epoch *3 epch 4卡
# fp32_sharding_dp_recompute

# #4.35 step/s, total step: 2104 step/epoch *3 epch 4卡 
# fp16_sharding_dp_recompute

