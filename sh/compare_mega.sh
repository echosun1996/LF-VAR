#!/bin/bash
##############################
# MAGE
# https://github.com/LTH14/mage
COMPARE_MAGE=true
pre_train_MAGE=false
finetune_MAGE=flase
class_unconditional_generation_pre_train=true
class_unconditional_generation_finetune=true
linear_probing=false
##############################
home_dir="$HOME"
root_path=$(pwd)
OUTPUT_DIR=$root_path/data/compare_results/MAGE
checkpoint_path=$root_path/compare_models/checkpoints
mage_root_path=$root_path/compare_models/reps/MAGE
replace_root_path=$root_path/compare_models/replace
dataset_dir=$root_path/data/local
export CUDA_VISIBLE_DEVICES=$gpu_id

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
if [ "$COMPARE_MAGE" = true ]; then
    echo "Comparing MAGE..."
    mkdir -p $root_path/compare_models/reps/
    cd $root_path/compare_models/reps/

    if [ -d "MAGE" ];
    then
        echo "[√] MAGE repo already Downloaded."
    else
        echo "Downloading MAGE repo..."
        git clone https://github.com/LTH14/mage.git MAGE
    fi

    # replace files
    cp $replace_root_path/MEGA-gen_img_uncond.py $mage_root_path/gen_img_uncond.py
    cp $replace_root_path/MEGA-pos_embed.py $mage_root_path/util/pos_embed.py

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

    replace_file_path="$home_dir/miniconda/envs/skin_generative/lib/python3.8/site-packages/timm/models/layers/helpers.py"
    backup_and_replace $replace_file_path "from torch._six import container_abc" "import collections.abc as container_abc"
    cp "$replace_root_path/MEGA-misc.py" "$mage_root_path/util/misc.py"
    cp "$replace_root_path/MEGA-datasets.py" "$mage_root_path/util/datasets.py"
    cp "$replace_root_path/MEGA-vqgan.py" "$mage_root_path/taming/models/vqgan.py"

    if [ "$pre_train_MAGE" = true ]; then
      cp "$replace_root_path/MEGA-main_pretrain.py" "$mage_root_path/main_pretrain.py"

      folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")

      PORT=$RANDOM
      for folder in "${folders[@]}"; do
        OUTPUT_DIR_PT=$OUTPUT_DIR"/pre_train/"$folder
        IMAGENET_DIR=$dataset_dir/HAM10000/input/train_val/HAM10000_img_class

        PRETRAIN_CHKPT=$checkpoint_path/vqgan_jax_strongaug.ckpt

        PRETRAIN_CHKPT_PTH_RESUME=$root_path/data/compare_results/MAGE/pre_train/$folder/checkpoint-last.pth


        echo "🏃 [MAGE] Pre_train $folder on $PORT..."
        cd $mage_root_path

        CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 \
        --master_addr="${MASTER_SERVER_ADDRESS}" --master_port=$PORT \
        main_pretrain.py \
        --batch_size 64 \
        --model mage_vit_base_patch16 \
        --mask_ratio_min 0.5 --mask_ratio_max 1.0 \
        --mask_ratio_mu 0.55 --mask_ratio_std 0.25 \
        --epochs 500 \
        --warmup_epochs 40 \
        --blr 1.5e-4 --weight_decay 0.05 \
        --output_dir ${OUTPUT_DIR_PT} \
        --log_dir ${OUTPUT_DIR_PT} \
        --data_path ${IMAGENET_DIR} \
        --dist_url tcp://${MASTER_SERVER_ADDRESS}:$PORT\
        --vqgan_jax_strongaug ${PRETRAIN_CHKPT}\
        --class_filter $folder
        #        --start_epoch 100 \
#        --resume ${PRETRAIN_CHKPT_PTH_RESUME}\
      done
    fi


    # Fine-tune MAGE
    if [ "$finetune_MAGE" = true ]; then
#        PRETRAIN_CHKPT=$checkpoint_path/mage-vitb-1600.pth
        PRETRAIN_CHKPT=$checkpoint_path/vqgan_jax_strongaug.ckpt


        echo $replace_root_path/MEGA-misc.py
        echo $mage_root_path/util/misc.py

        cp "$replace_root_path/MEGA-main_finetune.py" "$mage_root_path/main_finetune.py"

        folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")

        for folder in "${folders[@]}"; do
          OUTPUT_DIR_FT=$OUTPUT_DIR"/finetune/"$folder
          IMAGENET_DIR=$dataset_dir/HAM10000/input/train_val/HAM10000_img_class

          mkdir -p $OUTPUT_DIR_FT
          PORT=$RANDOM
          echo "🏃 [MAGE] Fine-tuning on $PORT..."
          cd $mage_root_path
          torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 \
            --master_addr="${MASTER_SERVER_ADDRESS}" --master_port=$PORT \
            main_finetune.py \
            --batch_size 90 \
            --model vit_base_patch16 \
            --global_pool \
            --vqgan_jax_strongaug ${PRETRAIN_CHKPT} \
            --epochs 100 \
            --blr 2.5e-4 --layer_decay 0.65 --interpolation bicubic \
            --weight_decay 0.05 --drop_path 0.1 --reprob 0 --mixup 0.8 --cutmix 1.0 \
            --output_dir ${OUTPUT_DIR_FT} \
            --log_dir ${OUTPUT_DIR_FT} \
            --data_path ${IMAGENET_DIR} \
            --dist_eval --dist_url tcp://${MASTER_SERVER_ADDRESS}:$PORT
        done
    fi
    if [ "$class_unconditional_generation_pre_train" = true ]; then
      echo "🏃 [MAGE] Class Unconditional Generation (Pre_train)"
      PRETRAIN_CHKPT_CKPT=$checkpoint_path/vqgan_jax_strongaug.ckpt

      folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")

      for folder in "${folders[@]}"; do
        PRETRAIN_CHKPT_PTH=$root_path"/data/compare_results/MAGE/pre_train/"$folder"/checkpoint-last.pth"
        OUTPUT_DIR_CUG=$OUTPUT_DIR"/class_unconditional_generation_pre_train/"$folder
        rm -rf $OUTPUT_DIR_CUG
        mkdir -p $OUTPUT_DIR_CUG
        cd $mage_root_path
        python gen_img_uncond.py --temp 6.0 --num_iter 20 \
          --ckpt ${PRETRAIN_CHKPT_PTH} --batch_size 32 --num_images 1500 \
          --vqgan_jax_strongaug ${PRETRAIN_CHKPT_CKPT} \
          --output_resize 512 \
          --model mage_vit_base_patch16 --output_dir ${OUTPUT_DIR_CUG}
      done

    fi
    if [ "$class_unconditional_generation_finetune" = true ]; then
      echo "🏃 [MAGE] Class Unconditional Generation (Finetune)"
      PRETRAIN_CHKPT_CKPT=$checkpoint_path/vqgan_jax_strongaug.ckpt

      folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")

      for folder in "${folders[@]}"; do
        PRETRAIN_CHKPT_PTH=$root_path"/data/compare_results/MAGE/finetune/"$folder"/checkpoint-last.pth"
        OUTPUT_DIR_CUG=$OUTPUT_DIR"/class_unconditional_generation_finetune/"$folder
        rm -rf $OUTPUT_DIR_CUG
        mkdir -p $OUTPUT_DIR_CUG
        cd $mage_root_path
        python gen_img_uncond.py --temp 6.0 --num_iter 20 \
          --ckpt ${PRETRAIN_CHKPT_PTH} --batch_size 32 --num_images 1500 \
          --vqgan_jax_strongaug ${PRETRAIN_CHKPT_CKPT} \
          --output_resize 512 \
          --model mage_vit_base_patch16 --output_dir ${OUTPUT_DIR_CUG}
      done

    fi

    if [ "$linear_probing" = true ]; then
      echo "🏃 [MAGE] Linear Probing"
      PORT=$RANDOM
      IMAGENET_DIR=$dataset_dir/GenerativeDataset
#      PRETRAIN_CHKPT=$root_path/data/compare_results/MAGE/finetune/checkpoint-last.pth
      PRETRAIN_CHKPT=$checkpoint_path/mage-vitb-1600.pth
      OUTPUT_DIR_LP=$OUTPUT_DIR"/linear_probing"
      PRETRAIN_CHKPT_CKPT=$checkpoint_path/vqgan_jax_strongaug.ckpt

      cp "$replace_root_path/MEGA-main_linprobe.py" "$mage_root_path/main_linprobe.py"
      cp "$replace_root_path/MEGA-crop.py" "$mage_root_path/util/crop.py"

      mkdir -p $OUTPUT_DIR_LP
      cd $mage_root_path
      torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 \
        --master_addr="${MASTER_SERVER_ADDRESS}" --master_port=$PORT \
          main_linprobe.py \
        --batch_size 128 \
        --model vit_base_patch16 \
        --finetune ${PRETRAIN_CHKPT} \
        --vqgan_jax_strongaug ${PRETRAIN_CHKPT_CKPT} \
        --epochs 90 \
        --blr 0.1 \
        --weight_decay 0.0 \
        --output_dir ${OUTPUT_DIR_LP} \
        --data_path ${IMAGENET_DIR} \
        --log_dir ${OUTPUT_DIR_LP} \
        --dist_eval --dist_url tcp://${MASTER_SERVER_ADDRESS}:$PORT
    fi
fi