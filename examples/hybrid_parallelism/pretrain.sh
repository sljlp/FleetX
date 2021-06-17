set -x

export PYTHONPATH=./atarashi/:$PYTHONPATH

export PADDLE_WITH_GLOO=0
export GLOG_v=1
export NCCL_DEBUG=INFO
export FLAGS_call_stack_level=2
export FLAGS_allocator_strategy=naive_best_fit

export PYTHONPATH=/code_lp/paddle/Paddle/build/python:$PYTHONPATH
# export PYTHONPATH=/code_lp/paddle/paddle_2.1/Paddle/build/python:$PYTHONPATH

PP=1
DP=1
MP=2

MS=64
BS=64

# task_name='gpt3-230B-'${PP}pp${DP}dp${MP}mp
task_name='continue-gpu'


output_dir=output/${task_name}
# rm -rf ${output_dir}

export CUDA_VISIBLE_DEVICES=6,7 

python3.7 -m paddle.distributed.fleet.launch \
	--log_dir ${output_dir}/log \
run_pretraining.py \
	--global_bsz 64 \
	--micro_bsz 64 \
	--max_seq_len 512 \
	--ernie_config_file config/ernie_base_config.json \
	--learning_rate 1e-4 \
	--log_steps 1 \
	--num_train_steps 600 \
	--save_steps 500 \
	--output_dir ${output_dir} \
	--use_recompute true \
	--use_sharding true \
	--use_sop false \
	--num_mp=2 \
	--num_sharding=1 \
	--num_pp=$PP \
	--num_dp=$DP \
         --init_checkpoint output/gpt3-230B-1pp1dp2mp/step_1000


