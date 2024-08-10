#!/bin/bash
set -e

mkdir -p ./benchmark_results

MODEL_NAME="$1"
GPUS_PER_MODEL=1
MODEL_REPLICAS=1
lm_eval --model vllm --model_args pretrained=$MODEL_NAME,tensor_parallel_size=$GPUS_PER_MODEL,dtype=auto,gpu_memory_utilization=0.5,data_parallel_size=$MODEL_REPLICAS,max_model_len=4096 \
    --tasks mmlu \
    --batch_size auto \
    --output_path ./benchmark_results/
# find the json file
OUTPUT_FILE=$(find ./benchmark_results/ -type f -name '*.json')

# Check if the JSON file was found
if [ -z "$OUTPUT_FILE" ]; then
  echo "No JSON file found in the output path."
  exit 1
fi

# display
# cat "$OUTPUT_FILE"