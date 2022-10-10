# export PYTHONPATH=/gpt3/Paddle/build/python
export PYTHONPATH=/gpt3/PaddleFleetX/paddle_test
log=output_dp2mp2pp2
rm -rf $log
python -m paddle.distributed.launch --log_dir $log --devices "0,1,2,3,4,5,6,7" \
    tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp2mp2pp2.yaml \
    -o Engine.max_steps=2 \
    -o Engine.save_load.output_dir="$log"

mv `find $log/epoch_0_step_0 -name "serial_model_dist*" -type f` $log
