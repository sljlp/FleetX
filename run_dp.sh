# export PYTHONPATH=/gpt3/Paddle/build/python
export PYTHONPATH=/gpt3/PaddleFleetX/paddle_test
log=output_dp2mp4
rm -rf $log
python -m paddle.distributed.launch --log_dir $log --devices "0,1,2,3,4,5,6,7" \
    tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp2mp4.yaml \
    -o Engine.max_steps=2 \
    -o Engine.save_load.output_dir="$log"

ln -s serial_model.pdparams $log/epoch_0_step_0/mp_00_sharding_00_pp_00/auto_dist0.pdparams || true
ln -s serial_model.pdattr $log/epoch_0_step_0/mp_00_sharding_00_pp_00/auto_dist0.pdattr || true

for id in 0 1 2 3 ; do
ln -s epoch_0_step_0/mp_0${id}_sharding_00_pp_00/serial_model.pdattr output_dp2mp4/auto_dist${id}.pdattr
ln -s epoch_0_step_0/mp_0${id}_sharding_00_pp_00/serial_model.pdparams output_dp2mp4/auto_dist${id}.pdparams
done