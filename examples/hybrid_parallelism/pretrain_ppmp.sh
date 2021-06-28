set -x

export PYTHONPATH=./atarashi/:$PYTHONPATH

export PADDLE_WITH_GLOO=0
export GLOG_v=1
export NCCL_DEBUG=INFO
export FLAGS_call_stack_level=2
export FLAGS_allocator_strategy=naive_best_fit

export PYTHONPATH=/code_lp/paddle/Paddle/build/develop/python:$PYTHONPATH
# export PYTHONPATH=/code_lp/paddle/paddle_2.1/Paddle/build/python:$PYTHONPATH

PP=2
DP=1
MP=4

MS=1
BS=8

task_name='gpt3-230B-'${PP}pp${DP}dp${MP}mp
output_dir=output/${task_name}
# rm -rf ${output_dir}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3.7 -m paddle.distributed.fleet.launch \
	--log_dir ${output_dir}/log \
run_pretraining.py \
	--global_bsz $BS \
	--micro_bsz $MS \
	--max_seq_len 512 \
	--ernie_config_file config/ernie_base_config.json \
	--learning_rate 1e-4 \
	--log_steps 1 \
	--num_train_steps 700 \
	--save_steps 200 \
	--output_dir ${output_dir} \
	--use_recompute true \
	--use_sharding true \
	--use_sop false \
	--num_mp=$MP \
	--num_sharding=1 \
	--num_pp=$PP \
	--num_dp=$DP \
#         --init_checkpoint output/gpt3-230B-1pp1dp2mp/step_2


