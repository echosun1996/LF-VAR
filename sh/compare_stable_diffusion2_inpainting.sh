#!/bin/bash
##############################
# COMPARE_SD2I
# https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
#gpu_sum=1
#gpu_id=1,2
COMPARE_SD2I=true
zeroShot_SD2I=flase
fine_tune_SD2I=true

##############################
home_dir="$HOME"
root_path=$(pwd)
OUTPUT_DIR=$root_path/data/compare_results/SD2I
checkpoint_path=$root_path/compare_models/checkpoints
run_root_path=$root_path/compare_models/run/SD2I
replace_root_path=$root_path/compare_models/replace
dataset_dir=$root_path/data/local
export CUDA_VISIBLE_DEVICES=$gpu_id

echo "### Total GPUs with usage < 20%: $gpu_sum"
echo "### GPU IDs: $gpu_id"


# https://github.com/facebookresearch/xformers
# Testing the xFormers
python -m xformers.info


mkdir -p $OUTPUT_DIR
if [ "$COMPARE_SD2I" = true ]; then
    echo "Comparing stable-diffusion-2-inpainting..."

    if [ "$zeroShot_SD2I" = true ]; then
      echo "Start zeroShot..."
      # HAM1000 test
      data_root_path=$dataset_dir/HAM10000/input/test
      img_path=$data_root_path/HAM10000_img
      seg_path=$data_root_path/HAM10000_seg
      meta_path=$dataset_dir/HAM10000/input/HAM10000_metadata.csv
      output_path=$OUTPUT_DIR/zeroShot/HAM10000_test
      python $run_root_path/SD2I-zeroshot.py --image $img_path --seg $seg_path --meta $meta_path --output $output_path
    fi

    if [ "$finetune_SD2I" = true ]; then
      echo "Start fine-tune..."
      cd $run_root_path
      accelerate launch SD2I-train_dreambooth_inpaint.py \
        --pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting"  \
        --instance_data_dir="images/dog" \
        --output_dir="stable-diffusion-inpainting-dog" \
        --instance_prompt="a photo of a sks dog" \
        --resolution=256 \
        --mixed_precision="fp16" \
        --train_batch_size=1 \
        --learning_rate=5e-6 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --max_train_steps=500 \
        --gradient_accumulation_steps=2 \
        --gradient_checkpointing \
        --train_text_encoder \
        --seed="0"
    fi
fi


