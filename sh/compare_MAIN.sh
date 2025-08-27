#!/bin/bash
set -e
##############################
# Main
COMPARE_MAIN=true
pre_train_MAIN=false
inference_MAIN=false
extract_radiomics=false
pre_train_radiomics_MAIN=false
inference_radiomics_MAIN=false
inference_radiomics_fixed_MAIN=false
cross_inference_MAIN_MICCAI=false
cross_inference_MAIN_tabsyn=false
##############################
home_dir="$HOME"
root_path=$(pwd)
OUTPUT_DIR=$root_path/data/compare_results/main
dataset_dir=$root_path/data/local
RUN_ROOT_PATH=$root_path/main
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


if [ "$COMPARE_MAIN" = true ]; then
    echo "Comparing MAIN..."

    mkdir -p $root_path/compare_models/checkpoints/
    cd $root_path/compare_models/checkpoints/
    if [ -f "var_d16.pth" ];
        then
            echo "[√] MAIN checkpoints exists."
        else
            wget -c -O "var_d16.pth" "https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth"
            echo "✅ var_d16.pth donwload complete."
    fi

    if [ "$pre_train_MAIN" = true ]; then
      cd $RUN_ROOT_PATH
      MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))

#      CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT train.py \
#      --depth=16 --bs=10 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --saln=1 --pn=512 --twde=0.08 --local_out_dir_path $OUTPUT_DIR/train --data_path $dataset_dir"/HAM10000/input"

      CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT train.py \
      --depth=16 --bs=35 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --local_out_dir_path $OUTPUT_DIR --data_path $dataset_dir"/HAM10000/input"

    fi

    if [ "$inference_MAIN" = true ]; then

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

    if [ "$extract_radiomics" = true ]; then
      cd $RUN_ROOT_PATH"/radiomics/"
      echo "Run Radiomics Main"
      meta_path=$dataset_dir/HAM10000/input/metadata.csv
      python radiomics_main.py --root-path $dataset_dir"/HAM10000/original" --meta-path $meta_path

    fi
    if [ "$pre_train_radiomics_MAIN" = true ]; then
      cd $RUN_ROOT_PATH
      MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
      CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT train.py \
      --depth=16 --bs=35 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --local_out_dir_path $OUTPUT_DIR --data_path $dataset_dir"/HAM10000/input"

    fi
    if [ "$inference_radiomics_MAIN" = true ]; then
      cd $RUN_ROOT_PATH
      MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
      CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT infer.py \
      --depth=16 --bs=35 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --local_out_dir_path $OUTPUT_DIR --data_path $dataset_dir"/HAM10000/input"
    fi

    if [ "$inference_radiomics_fixed_MAIN" = true ]; then

      RUN_ROOT_PATH_FIXED=$root_path"/data/compare_results/main_fixed_radiomics/"
      original_main_path=$root_path"/data/compare_results/main/"

      if [ ! -d "$RUN_ROOT_PATH_FIXED" ]; then
        mkdir -p $RUN_ROOT_PATH_FIXED
        cp $original_main_path"/ar-ckpt-best.pth" $RUN_ROOT_PATH_FIXED"/"
        cp $original_main_path"/ar-ckpt-last.pth" $RUN_ROOT_PATH_FIXED"/"
      fi
      fixed_csv=$dataset_dir"/HAM10000/input/radiomics_fixed.csv"

      cd $RUN_ROOT_PATH
      MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
      CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT infer.py \
      --depth=16 --bs=35 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --local_out_dir_path $RUN_ROOT_PATH_FIXED --data_path $dataset_dir"/HAM10000/input" --fixed_csv_path $fixed_csv
    fi
    # Here is the cross inference for MICCAI conferance paper
    if [ "$cross_inference_MAIN_MICCAI" = true ]; then

      RUN_ROOT_PATH_FIXED=$root_path"/data/compare_results/main_cross_infer/"
      original_main_path=$root_path"/data/compare_results/main/"

      if [ ! -d "$RUN_ROOT_PATH_FIXED" ]; then
        mkdir -p $RUN_ROOT_PATH_FIXED
        cp $original_main_path"/ar-ckpt-best.pth" $RUN_ROOT_PATH_FIXED"/"
        cp $original_main_path"/ar-ckpt-last.pth" $RUN_ROOT_PATH_FIXED"/"
      fi
      fixed_csv=$dataset_dir"/HAM10000/input/radiomics_fixed.csv"

      cd $RUN_ROOT_PATH
      MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
      CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT infer.py \
      --depth=16 --bs=35 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --local_out_dir_path $RUN_ROOT_PATH_FIXED --data_path $dataset_dir"/HAM10000/input" --fixed_csv_path $fixed_csv --cross_infer true
    fi

    # Here is the cross inference for journal paper
    if [ "$cross_inference_MAIN_tabsyn" = true ]; then

      RUN_ROOT_PATH_TABSYN=$root_path"/data/compare_results/main_cross_infer_tabsyn/"
      original_main_path=$root_path"/data/compare_results/main/"

      if [ ! -d "$RUN_ROOT_PATH_TABSYN" ]; then
        mkdir -p $RUN_ROOT_PATH_TABSYN
        cp $original_main_path"/ar-ckpt-best.pth" $RUN_ROOT_PATH_TABSYN"/"
        cp $original_main_path"/ar-ckpt-last.pth" $RUN_ROOT_PATH_TABSYN"/"
      fi
      tabsyn_csv=$root_path"/data/compare_results/MixedTypeTabular/HAM10000_tabsyn.csv"

      cd $RUN_ROOT_PATH
      MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
      CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT infer.py \
      --depth=16 --bs=35 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --local_out_dir_path $RUN_ROOT_PATH_TABSYN --data_path $dataset_dir"/HAM10000/input" --tabsyn_csv_path $tabsyn_csv --cross_infer true
    fi

fi