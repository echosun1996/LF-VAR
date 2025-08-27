#!/bin/bash

# Get GPU memory usage and total memory
gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
gpu_memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)

# Initialize variables
gpu_sum=0
gpu_id=()

# Iterate through each GPU's memory usage
index=0
while IFS= read -r usage; do
  # Get total memory
  total_memory=$(echo "$gpu_memory_total" | sed -n "$((index + 1))p")

  # Calculate if memory usage exceeds 80%
  if [ "$usage" -lt "$((total_memory * 80 / 100))" ]; then
    gpu_sum=$((gpu_sum + 1))
    gpu_id+=("$index")
  fi
  index=$((index + 1))
done <<< "$gpu_memory_usage"

# Convert GPU number array to comma-separated string
gpu_id=$(IFS=,; echo "${gpu_id[*]}")
echo "GPU IDs: $gpu_id"

current_host=$(hostname)

if [ "$current_host" == "mmai-2" ]; then
    echo "current hostname $current_host, start to replace the GPU ID...."

    IFS=, read -r -a gpu_id_array <<< "$gpu_id"

    for i in "${!gpu_id_array[@]}"; do
        if [ "${gpu_id_array[$i]}" == "0" ]; then
            gpu_id_array[$i]="1"
        elif [ "${gpu_id_array[$i]}" == "1" ]; then
            gpu_id_array[$i]="0"
        fi
    done
    gpu_id=$(IFS=,; echo "${gpu_id_array[*]}")
fi


# Output results
echo "Total GPUs with memory usage < 20%: $gpu_sum"
echo "GPU IDs: $gpu_id"

