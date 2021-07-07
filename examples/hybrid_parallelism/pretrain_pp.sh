set -x

export PYTHONPATH=./atarashi/:$PYTHONPATH

# export PYTHONPATH=/code_lp/paddle/Paddle/build/develop/python/:$PYTHONPATH
# export PYTHONPATH=/code_lp/paddle/Paddle/build/develop/python:$PYTHONPATH
export PYTHONPATH=/code_lp/paddle/Paddle/build/fix_pp_precise_1f1b/python:$PYTHONPATH

export GLOG_v=1
# export GLOG_vmodule="matmul_v2_op=3"
export NCCL_DEBUG=INFO
export FLAGS_call_stack_level=2
export FLAGS_allocator_strategy=naive_best_fit
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
rm -rf *.prototxt
rm -rf core.*

task_name='newest-pp-1f1b-fixed-bs8'
output_dir=output/${task_name}
rm -rf ${output_dir}
export CUDA_VISIBLE_DEVICES=0,1 #,2,3,4,5,6,7
python3.7 -m paddle.distributed.fleet.launch \
        --gpus="0,1" \
	--log_dir ${output_dir}/log \
run_pretraining.py \
	--global_bsz 8 \
	--micro_bsz 1 \
	--max_seq_len 512 \
	--ernie_config_file config/ernie_base_config.json \
	--learning_rate 1e-4 \
	--log_steps 1 \
	--num_train_steps 250 \
	--save_steps 100 \
	--output_dir ${output_dir} \
	--use_recompute true \
	--use_sharding true \
	--num_mp=1 \
	--num_sharding=1 \
	--num_pp=2 \
	--num_dp=1 \
	--debug false
        # --init_checkpoint output/pp-test-1f1b/step_1 \
    # --debug false \
