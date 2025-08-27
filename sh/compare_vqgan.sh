#!/bin/bash
##############################
# MAGE
# https://github.com/LTH14/mage
#gpu_sum=1
#gpu_id=0
COMPARE_VQGAN=true
fine_tune_VQGAN=true
#class_unconditional_generation=false
##############################
home_dir="$HOME"
root_path=$(pwd)
OUTPUT_DIR=$root_path/data/compare_results/VQGAN
checkpoint_path=$root_path/compare_models/checkpoints
mage_root_path=$root_path/compare_models/reps/VQGAN
replace_root_path=$root_path/compare_models/replace
dataset_dir=$root_path/data/local


echo "### Total GPUs with usage < 20%: $gpu_sum"
echo "### GPU IDs: $gpu_id"

backup_and_replace() {
    # Define the paths to the original file and backup file
    helpers_path="$1"
    backup_path="${helpers_path%.py}_backup.py"

    # Check if the backup file exists; if not, create the backup
    if [ ! -f "$backup_path" ]; then
        cp "$helpers_path" "$backup_path"
        echo "Backup created at $backup_path"
    else
        echo "Backup already exists at $backup_path"
    fi

    # Use sed to replace the line with the passed replacement content
    sed -i "s/$2/$3/" "$helpers_path"
    echo "Replacement done in $helpers_path"
}


mkdir -p $OUTPUT_DIR
if [ "$COMPARE_VQGAN" = true ]; then
    echo "Comparing VQGAN..."
    mkdir -p $root_path/compare_models/reps/
    cd $root_path/compare_models/reps/

    if [ -d "VQGAN" ];
    then
        echo "[√] MAGE repo already Downloaded."
    else
        echo "Downloading VQGAN repo..."
        git clone https://github.com/CompVis/taming-transformers.git VQGAN
    fi

    # replace files
#    cp $replace_root_path/MEGA-gen_img_uncond.py $mage_root_path/gen_img_uncond.py
#    cp $replace_root_path/MEGA-pos_embed.py $mage_root_path/util/pos_embed.py

    cd $checkpoint_path
    if [ -f "vqgan_jax_strongaug.ckpt" ];
        then
            echo "[√] MAGE checkpoints exists."
        else
            wget -c -O "vqgan_jax_strongaug.ckpt" "https://drive.usercontent.google.com/download?id=13S_unB87n6KKuuMdyMnyExW0G1kplTbP&export=download&authuser=0&confirm=t&uuid=11a8bafc-c712-4bf0-973f-eaec5a955ba8&at=AENtkXacGEGoaNO8oUA63zwkD6cG%3A1730434831798"
            echo "✅ vqgan_jax_strongaug.ckpt donwload complete."
    fi

    if [ -f "mage-vitb-1600.pth" ];
    then
        echo "[√] MAGE checkpoints exists."
    else
        wget -c -O "mage-vitb-1600.pth" "https://drive.usercontent.google.com/download?id=1Q6tbt3vF0bSrv5sPrjpFu8ksG3vTsVX2&export=download&authuser=0&confirm=t&uuid=39fcad8c-7c21-4e84-96d1-13aedac960f1&at=AENtkXam8kwqfOjarc4-Zh8zq2w2%3A1731562489824"
        echo "✅ mage-vitb-1600.pth donwload complete."
fi

    MASTER_SERVER_ADDRESS="127.0.0.1"

    # Fine-tune MAGE
    if [ "$fine_tune_MAGE" = true ]; then
        replace_file_path="$home_dir/miniconda/envs/skin_generative/lib/python3.8/site-packages/timm/models/layers/helpers.py"
        backup_and_replace $replace_file_path "from torch._six import container_abc" "import collections.abc as container_abc"
        replace_file_path="$home_dir/miniconda/envs/skin_generative/lib/python3.8/site-packages/timm/models/layers/helpers.py"
        PRETRAIN_CHKPT=$checkpoint_path/mage-vitb-1600.pth


        echo $replace_root_path/MEGA-misc.py
        echo $mage_root_path/util/misc.py
        cp "$replace_root_path/MEGA-misc.py" "$mage_root_path/util/misc.py"
        cp "$replace_root_path/MEGA-datasets.py" "$mage_root_path/util/datasets.py"
        cp "$replace_root_path/MEGA-vqgan.py" "$mage_root_path/taming/models/vqgan.py"
        cp "$replace_root_path/MEGA-main_finetune.py" "$mage_root_path/main_finetune.py"

        IMAGENET_DIR=$dataset_dir/GenerativeDataset
        OUTPUT_DIR_FT=$OUTPUT_DIR"/fine_tune"
        mkdir -p $OUTPUT_DIR_FT
        PORT=$RANDOM
        echo "🏃 [MAGE] Fine-tuning on $PORT..."
        cd $mage_root_path
        torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 \
          --master_addr="${MASTER_SERVER_ADDRESS}" --master_port=$PORT \
          main_finetune.py \
          --batch_size 100 \
          --model vit_base_patch16 \
          --global_pool \
          --finetune ${PRETRAIN_CHKPT} \
          --epochs 100 \
          --blr 2.5e-4 --layer_decay 0.65 --interpolation bicubic \
          --weight_decay 0.05 --drop_path 0.1 --reprob 0 --mixup 0.8 --cutmix 1.0 \
          --output_dir ${OUTPUT_DIR_FT} \
          --log_dir ${OUTPUT_DIR_FT} \
          --data_path ${IMAGENET_DIR} \
          --dist_eval --dist_url tcp://${MASTER_SERVER_ADDRESS}:$PORT

    fi
    if [ "$class_unconditional_generation" = true ]; then
      echo "🏃 [MAGE] Class Unconditional Generation"
      PRETRAIN_CHKPT=$checkpoint_path/vqgan_jax_strongaug.ckpt
      OUTPUT_DIR_CUG=$OUTPUT_DIR"/class_unconditional_generation"
      mkdir -p $OUTPUT_DIR_CUG
      cd $mage_root_path
      python gen_img_uncond.py --temp 6.0 --num_iter 20 \
        --ckpt ${PRETRAIN_CHKPT} --batch_size 32 --num_images 50000 \
        --model mage_vit_base_patch16 --output_dir ${OUTPUT_DIR_CUG}
    fi
fi