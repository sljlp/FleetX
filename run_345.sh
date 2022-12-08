
set -x
log=output_345
ckpt=ckpt_345
rm -rf $log

set_gpu_config(){
    gpu=gpu

}

set_dcu_config(){
    gpu=gpu
}

nvidia-smi && set_gpu_config
rocm-smi && set_dcu_config

python tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
    -o Engine.max_steps=2000 \
    -o Engine.mix_precision.use_pure_fp16=True \
    -o Distributed.sharding.sharding_degree=1 \
    -o Distributed.sharding.sharding_stage=1 \
    -o Global.device=$gpu \
    -o Model.use_recompute=False \
    -o Model.hidden_dropout_prob=0 \
    -o Model.attention_probs_dropout_prob=0 \
    -o Model.num_layers=24

    # -o Engine.save_load.output_dir="$log" \
    # -o Model.num_attention_heads=12 \
    # -o Engine.save_load.ckpt_dir="$ckpt"
    