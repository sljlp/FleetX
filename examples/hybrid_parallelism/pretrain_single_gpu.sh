set -x

export PYTHONPATH=./atarashi/:$PYTHONPATH

export PADDLE_WITH_GLOO=0
export GLOG_v=1
export NCCL_DEBUG=INFO
export FLAGS_call_stack_level=2
export FLAGS_allocator_strategy=naive_best_fit

rm -rf *.prototxt
rm -rf core.*

task_name='gpt3-230B-32pp4dp2mp'
output_dir=output/${task_name}
rm -rf ${output_dir}

export CUDA_VISIBLE_DEVICES=0

python -m paddle.distributed.fleet.launch \
	--log_dir ${output_dir}/log \
run_pretraining.py \
	--global_bsz 64 \
	--micro_bsz 64 \
	--max_seq_len 512 \
	--ernie_config_file config/ernie_base_config.json \
	--learning_rate 1e-4 \
	--log_steps 1 \
	--num_train_steps 2000 \
	--save_steps 10 \
	--output_dir ${output_dir} \
	--use_recompute true \
	--use_sharding false \
	--use_sop false \
	--num_mp=1 \
	--num_sharding=1 \
	--num_pp=1 \
	--num_dp=1 \

