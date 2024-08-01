MODEL_NAME=$1
GPUS_PER_MODEL=4
MODEL_REPLICAS=1
lm_eval --model vllm \         
    --model_args pretrained=$MODEL_NAME,tensor_parallel_size=$GPUS_PER_MODEL,dtype=auto,gpu_memory_utilization=0.5,data_parallel_size=$MODEL_REPLICAS,max_model_len=4096 \
    --tasks mmlu \
    --batch_size auto \
    --output_path ./benchmark_results.json
# display the results
cat ./benchmark_results.json