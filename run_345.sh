
set -x
log=output_345
ckpt=ckpt_345/epoch_0_step_0/
rm -rf $log

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

set_xpu_config

nvidia-smi && set_gpu_config
rocm-smi && set_dcu_config

python tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
    -o Engine.max_steps=20 \
    -o Engine.mix_precision.use_pure_fp16=False \
    -o Distributed.sharding.sharding_degree=1 \
    -o Distributed.sharding.sharding_stage=1 \
    -o Global.device=$gpu \
    -o Model.use_recompute=False \
    -o Model.hidden_dropout_prob=0 \
    -o Model.attention_probs_dropout_prob=0 \
    -o Model.num_layers=8 \
    -o Engine.save_load.ckpt_dir="$ckpt" \
    -o Engine.save_load.load_meta_state=False
