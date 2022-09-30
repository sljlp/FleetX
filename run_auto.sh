# export PYTHONPATH=/gpt3/Paddle/build/python
export PYTHONPATH=/gpt3/PaddleFleetX/paddle_test

export FLAGS_USE_STANDALONE_EXECUTOR=False # 设置执行器环境变量
log_dir=output_auto_dp2_loadmp4_step2
rm -rf $log_dir
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1" \
    ./tools/auto.py \
    -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_1.3B_dp2.yaml \
    -o Engine.max_steps=2 \
    -o Engine.save_load.output_dir="$log_dir" \
    -o Engine.save_load.ckpt_dir="/gpt3/PaddleFleetX/output_dp2mp4"

# log_dir=output_auto_dp2_loadmp4_step10
# rm -rf $log_dir
# python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1" \
#     ./tools/auto.py \
#     -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_1.3B_dp2.yaml \
#     -o Engine.max_steps=10 \
#     -o Engine.save_load.output_dir="$log_dir" \
#     -o Engine.save_load.ckpt_dir="/gpt3/PaddleFleetX/output_dp2mp4"


