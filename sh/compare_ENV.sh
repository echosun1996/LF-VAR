#!/bin/bash
# Suspending the script if any command fails
set -e

# Random seed
RANDOM=$(date +%s | cut -c 7-10 | sed 's/^0*//')

# Activate the 'skin_generative' environment
CONDA_PATH="$HOME/miniconda/etc/profile.d/conda.sh"
if [[ -f "$CONDA_PATH" ]]; then
    # A40-24Q
    export HF_HOME=/mnt/huggingface_cache
    source $CONDA_PATH
fi
CONDA_PATH="$HOME/opt/miniconda3/etc/profile.d/conda.sh"
if [[ -f "$CONDA_PATH" ]]; then
    source $CONDA_PATH
fi

# Active environment
conda activate skin_generative
echo "✅ The 'skin_generative' environment is now active."

#sudo apt install imagemagick
#mogrify -path /mnt/SkinGenerativeModel/code/data/local/HAM10000/input/train/HuggingFace/akiec/HAM10000_img_class_png/ -format png /mnt/SkinGenerativeModel/code/data/local/HAM10000/input/train/HuggingFace/akiec/HAM10000_img_class/*.jpg

pip install -r sh/requirements.txt --quiet
echo "✅ Requirements all installed."


# relogin wandb
wandb login --relogin 52e1d262f0a0f8911bdf0b02938c845b023f1bd5

# Check data folder
if [ ! -d "data" ]; then
    echo "Error: 'data' folder not found in the current directory. Init first!"
    exit 1
fi

checkpoint_path=compare_models/checkpoints
results_path=data/compare_results
mkdir -p $checkpoint_path
mkdir -p $results_path


