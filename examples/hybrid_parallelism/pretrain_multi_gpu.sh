set -x

EXP PYTHONPATH=./atarashi/:$PYTHONPATH

EXP PADDLE_WITH_GLOO=0
EXP GLOG_v=1
EXP NCCL_DEBUG=INFO
EXP FLAGS_call_stack_level=2
EXP FLAGS_allocator_strategy=naive_best_fit

rm -rf *.prototxt
rm -rf core.*

task_name='gpt3-230B-1pp1dp2mp'
output_dir=output/${task_name}
rm -rf ${output_dir}

EXP CUDA_VISIBLE_DEVICES=6,7

python3.7 -m paddle.distributed.fleet.launch \
	--log_dir ${output_dir}/log \
run_pretraining.py \
	--global_bsz 64 \
	--micro_bsz 64 \
	--max_seq_len 512 \
	--ernie_config_file config/ernie_base_config.json \
	--learning_rate 1e-4 \
	--log_steps 1 \
	--num_train_steps 1000 \
	--save_steps 500 \
	--output_dir ${output_dir} \
	--use_recompute true \
	--use_sharding true \
	--use_sop false \
	--num_mp=2 \
	--num_sharding=1 \
	--num_pp=1 \
	--num_dp=1 \

