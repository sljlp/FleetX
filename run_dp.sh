set -xue

unset PYTHONPATH
log=output_dp1sharding4
rm -rf $log
python -m paddle.distributed.launch --log_dir $log --devices "0,1,2,3" \
    tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp1sharding4.yaml \
    -o Engine.max_steps=2 \
    -o Engine.save_load.output_dir="$log"
#     # -o Engine.save_load.ckpt_dir="output_dp2sharding4/epoch_0_step_0/"

# path=`find $log -name "dist_saved.pdopt" |grep 'sharding_00'`
# exit
for s in `seq 0 3`; do
[[ -f $log/epoch_0_step_0/mp_00_sharding_0${s}_pp_00/tmp/dist_saved.pdopt || -h $log/epoch_0_step_0/mp_00_sharding_0${s}_pp_00/tmp/dist_saved.pdopt ]] && mv $log/epoch_0_step_0/mp_00_sharding_0${s}_pp_00/tmp/dist_saved.pdopt $log/epoch_0_step_0/mp_00_sharding_0${s}_pp_00/dist_saved.pdopt 
done

for s in `seq 1 3`; do
rm -f $log/epoch_0_step_0/mp_00_sharding_0${s}_pp_00/dist_saved.pdopt
ln -s ../mp_00_sharding_00_pp_00/dist_saved.pdopt $log/epoch_0_step_0/mp_00_sharding_0${s}_pp_00/dist_saved.pdopt
done

# for s in `seq 0 3`; do
# mkdir -p $log/epoch_0_step_0/mp_00_sharding_0${s}_pp_00/tmp
# [[ -h $log/epoch_0_step_0/mp_00_sharding_0${s}_pp_00/dist_saved.pdopt || -f $log/epoch_0_step_0/mp_00_sharding_0${s}_pp_00/dist_saved.pdopt ]] && mv $log/epoch_0_step_0/mp_00_sharding_0${s}_pp_00/dist_saved.pdopt $log/epoch_0_step_0/mp_00_sharding_0${s}_pp_00/tmp
# done

log=output_dp1sharding4_save
rm -rf $log
python -m paddle.distributed.launch --log_dir $log --devices "0,1,2,3" \
    tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp1sharding4.yaml \
    -o Engine.max_steps=2 \
    -o Engine.save_load.output_dir="$log" \
    -o Engine.save_load.ckpt_dir="output_dp1sharding4/epoch_0_step_0/"

