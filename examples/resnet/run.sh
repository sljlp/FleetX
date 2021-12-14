export PYTHONPATH=/code_lp/paddle/Paddle/build/python/:
export FLAGS_cudnn_deterministic=True
export CUDA_VISIBLE_DEVICES=4
LOG=${LOG:-log}
python3 -m paddle.distributed.fleet.launch \
--log_dir $LOG \
--run_mode="collective" \
train_fleet_dygraph.py
