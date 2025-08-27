#!/bin/bash
##############################
# VAR
# https://github.com/FoundationVision/VAR
COMPARE_VAR=true
pre_train_VAR=false
inference_VAR=true
##############################
home_dir="$HOME"
root_path=$(pwd)
OUTPUT_DIR=$root_path/data/compare_results/VAR
dataset_dir=$root_path/data/local
RUN_ROOT_PATH=$root_path/compare_models/reps/VAR
checkpoint_path=$root_path/compare_models/checkpoints
replace_root_path=$root_path/compare_models/replace

echo "### Total GPUs with usage < 20%: $gpu_sum"
echo "### GPU IDs: $gpu_id"

mkdir -p $OUTPUT_DIR


# Define categories and their corresponding prompts
keys=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
values=(
    "An image of a skin area with actinic keratoses or intraepithelial carcinoma."
    "An image of a skin area with basal cell carcinoma."
    "An image of a skin area with benign keratosis-like lesions."
    "An image of a skin area with dermatofibroma."
    "An image of a skin area with melanoma."
    "An image of a skin area with melanocytic nevi."
    "An image of a skin area with a vascular lesion."
)


cp $replace_root_path/VAR-data.py $RUN_ROOT_PATH/utils/data.py
#cp $replace_root_path/VAR-inference.py $RUN_ROOT_PATH/inference.py
cp $replace_root_path/VAR-infer.py $RUN_ROOT_PATH/infer.py
cp $replace_root_path/VAR-trainer.py $RUN_ROOT_PATH/trainer.py


if [ "$COMPARE_VAR" = true ]; then
    echo "Comparing VAR..."
    mkdir -p $root_path/compare_models/reps/
    cd $root_path/compare_models/reps/

    if [ -d "VAR" ];
    then
        echo "[√] VAR repo already Downloaded."
    else
        echo "Downloading VAR repo..."
        git clone https://github.com/FoundationVision/VAR.git VAR
    fi



    mkdir -p $root_path/compare_models/checkpoints/
    cd $root_path/compare_models/checkpoints/
    if [ -f "var_d16.pth" ];
        then
            echo "[√] VAR checkpoints exists."
        else
            wget -c -O "var_d16.pth" "https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth"
            echo "✅ var_d16.pth donwload complete."
    fi

    if [ "$pre_train_VAR" = true ]; then

      cd $RUN_ROOT_PATH
      MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))

#      CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT train.py \
#      --depth=16 --bs=10 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --saln=1 --pn=512 --twde=0.08 --local_out_dir_path $OUTPUT_DIR/train --data_path $dataset_dir"/HAM10000/input"

      CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT train.py \
      --depth=16 --bs=35 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --local_out_dir_path $OUTPUT_DIR --data_path $dataset_dir"/HAM10000/input"

    fi

    if [ "$inference_VAR" = true ]; then

      cd $OUTPUT_DIR/
      rm -rf 0 1 2 3 4 5 6
      rm -rf akiec bcc bkl df mel nv mel vasc

      cd $RUN_ROOT_PATH
      MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
#      CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT infer.py \
#      --depth=16 --bs=10 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --saln=1 --pn=512 --twde=0.08 --local_out_dir_path $OUTPUT_DIR/inference --data_path $dataset_dir"/HAM10000/input"

      CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT infer.py \
      --depth=16 --bs=35 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --local_out_dir_path $OUTPUT_DIR --data_path $dataset_dir"/HAM10000/input"

    fi

#    if [ -f "Derm-T2IM.yaml" ];
#      then
#          echo "[√] Derm-T2IM.yaml exists."
#      else
#          wget -c -O "Derm-T2IM.yaml" "https://huggingface.co/MAli-Farooq/Derm-T2IM/resolve/main/Derm-T2IM.yaml?download=true"
#          echo "✅ Derm-T2IM.yaml donwload complete."
#    fi
#
#    if [ -f "Derm-T2IM.json" ];
#      then
#          echo "[√] Derm-T2IM.json exists."
#      else
#          # Set YAML and JSON file paths
#          yaml_path="$root_path/compare_models/checkpoints/Derm_T2IM/Derm-T2IM.yaml"
#          json_path="$root_path/compare_models/checkpoints/Derm_T2IM/model_index.json"
#
#          # Call Python script to convert YAML to JSON
##          python $root_path/py_scripts/preprocessing/convert_yaml_to_json.py --yaml_path "$yaml_path" --json_path "$json_path"
##          echo "✅ Derm-T2IM.json donwload complete."
#    fi

#
#    # Iterate through array
#    for i in "${!keys[@]}"; do
#        key=${keys[$i]}
#        prompt=${values[$i]}
#        output_dir=$OUTPUT_DIR/inference/$key
#        rm -rf $output_dir
#        mkdir -p $output_dir
#        echo "Category: $key"
#        echo "Prompt: $prompt"
#        python $RUN_ROOT_PATH/inferance.py --prompt "$prompt" --output $output_dir --n 1500 --batch_size 30 --pretrain $root_path/compare_models/checkpoints/Derm-T2IM.safetensors
#    done

fi