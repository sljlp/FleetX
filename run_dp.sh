unset PYTHONPATH
# log=output_dp2sharding4
# rm -rf $log
# python -m paddle.distributed.launch --log_dir $log --devices "0,1,2,3,4,5,6,7" \
#     tools/train.py \
#     -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp2sharding4.yaml \
#     -o Engine.max_steps=2 \
#     -o Engine.save_load.output_dir="$log"
#     # -o Engine.save_load.ckpt_dir="output_dp2sharding4/epoch_0_step_0/"
# path=`find $log -name "dist_saved.pdopt" |grep 'sharding_00'`
# mv $path `dirname $path`/model_state.pdopt

# find $log 

log=output_dp2sharding4_save
rm -rf $log
python -m paddle.distributed.launch --log_dir $log --devices "0,1,2,3,4,5,6,7" \
    tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp2sharding4.yaml \
    -o Engine.max_steps=2 \
    -o Engine.save_load.output_dir="$log" \
    -o Engine.save_load.ckpt_dir="output_dp2sharding4/epoch_0_step_0/"

