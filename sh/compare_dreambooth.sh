#!/bin/bash
##############################
# COMPARE_SD2I
# https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
#gpu_sum=1
#gpu_id=1,2
COMPARE_dreambooth=true
finetune_dreambooth=false
finetune_dreambooth_with_text_encoder=false
inferesnce_dreambooth=true
inferesnce_dreambooth_with_text_encoder=false
##############################
home_dir="$HOME"
root_path=$ROOT_PATH

OUTPUT_DIR=$root_path/data/compare_results/Dreambooth
checkpoint_path=$root_path/compare_models/checkpoints
run_root_path=$root_path/compare_models/run/Dreambooth
replace_root_path=$root_path/compare_models/replace
dataset_dir=$root_path/data/local
export CUDA_VISIBLE_DEVICES=$gpu_id

echo "### Total GPUs with usage < 20%: $gpu_sum"
echo "### GPU IDs: $gpu_id"


# https://github.com/facebookresearch/xformers
# Testing the xFormers
python -m xformers.info


mkdir -p $OUTPUT_DIR
if [ "$COMPARE_dreambooth" = true ]; then
    echo "Comparing dreambooth..."

    if [ "$finetune_dreambooth" = true ]; then
      echo "Start Fine-tune Multi Subject Dreambooth for Inpainting Models..."
      # https://github.com/huggingface/diffusers/tree/main/examples/research_projects/multi_subject_dreambooth_inpainting

#      cd $run_root_path
      output_path=$OUTPUT_DIR/train
      MODEL_NAME="runwayml/stable-diffusion-inpainting"
      metadata_file=$dataset_dir"/HAM10000/input/metadata.csv"

      accelerate launch $run_root_path/train_multi_subject_dreambooth_inpainting.py \
        --pretrained_model_name_or_path=$MODEL_NAME  \
        --metadata-file=$metadata_file \
        --output_dir=$output_path \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=2 \
        --learning_rate=3e-6 \
        --max_train_steps=5000 \
        --report_to_wandb
    fi

    if [ "$finetune_dreambooth_with_text_encoder" = true ]; then
      echo "Start Fine-tune Multi Subject Dreambooth for Inpainting Models(with text encoder)..."
      # https://github.com/huggingface/diffusers/tree/main/examples/research_projects/multi_subject_dreambooth_inpainting

#      cd $run_root_path
      output_path=$OUTPUT_DIR/train_with_text_encoder
      MODEL_NAME="runwayml/stable-diffusion-inpainting"
      metadata_file=$dataset_dir"/HAM10000/input/metadata.csv"
      mkdir -p $output_path

      accelerate launch $run_root_path/train_multi_subject_dreambooth_inpainting.py \
        --pretrained_model_name_or_path=$MODEL_NAME  \
        --metadata-file=$metadata_file \
        --output_dir=$output_path \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=2 \
        --learning_rate=3e-6 \
        --max_train_steps=5000 \
        --report_to_wandb \
        --train_text_encoder
    fi

    if [ "$inferesnce_dreambooth_with_text_encoder" = true ]; then
      echo "Start Inference Multi Subject Dreambooth for Inpainting Models(with text encoder)..."
      # https://github.com/huggingface/diffusers/tree/main/examples/research_projects/multi_subject_dreambooth_inpainting

      output_path=$OUTPUT_DIR/infer_with_text_encoder
      mkdir -p $output_path


      cd $run_root_path
      python model_downloader.py --model_dir $OUTPUT_DIR/infer_with_text_encoder

      # Copy files
      rm -rf $OUTPUT_DIR/infer_with_text_encoder/models

#      cp -r $OUTPUT_DIR/infer_with_text_encoder/models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590/* $OUTPUT_DIR/infer_with_text_encoder/models
      cp -r --remove-destination $OUTPUT_DIR/train/checkpoint-5000/* $OUTPUT_DIR/infer_with_text_encoder/models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590

      metadata_file=$dataset_dir"/HAM10000/input/metadata.csv"
      python inference.py --model_dir $OUTPUT_DIR/infer_with_text_encoder --metadata_file $metadata_file --output $OUTPUT_DIR/infer_with_text_encoder --data_root $ROOT_PATH
    fi

    if [ "$inferesnce_dreambooth" = true ]; then
      echo "Start Inference Multi Subject Dreambooth for Inpainting Models..."
      # https://github.com/huggingface/diffusers/tree/main/examples/research_projects/multi_subject_dreambooth_inpainting

      output_path=$OUTPUT_DIR/infer
      rm -rf $output_path
      mkdir -p $output_path


      cd $run_root_path
      python model_downloader.py --model_dir $OUTPUT_DIR/infer

      # Copy files
      rm -rf $OUTPUT_DIR/infer/models

#      cp -r $OUTPUT_DIR/infer_with_text_encoder/models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590/* $OUTPUT_DIR/infer_with_text_encoder/models
#      cp -r --remove-destination $OUTPUT_DIR/train/checkpoint-5000/* $OUTPUT_DIR/infer/models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590

      metadata_file=$dataset_dir"/HAM10000/input/metadata.csv"
      python inference.py --model_dir $OUTPUT_DIR/infer --metadata_file $metadata_file --output $OUTPUT_DIR/infer --data_root $ROOT_PATH
    fi

fi


