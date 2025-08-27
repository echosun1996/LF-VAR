#!/bin/bash
##############################
# Derm-T2IM
# https://github.com/MAli-Farooq/Derm-T2IM
# https://huggingface.co/MAli-Farooq/Derm-T2IM
COMPARE_Derm_T2IM=true
##############################
home_dir="$HOME"
root_path=$(pwd)
OUTPUT_DIR=$root_path/data/compare_results/Derm-T2IM
dataset_dir=$root_path/data/local
RUN_ROOT_PATH=$root_path/compare_models/run/Derm-T2IM

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

if [ "$COMPARE_Derm_T2IM" = true ]; then
    echo "Comparing Derm-T2IM..."

    mkdir -p $root_path/compare_models/checkpoints/
    cd $root_path/compare_models/checkpoints/
    if [ -f "Derm-T2IM.safetensors" ];
      then
          echo "[√] Derm-T2IM.safetensors exists."
      else
          wget -c -O "Derm-T2IM.safetensors" "https://huggingface.co/MAli-Farooq/Derm-T2IM/resolve/main/Derm-T2IM.safetensors?download=true"
          echo "✅ Derm-T2IM.safetensors donwload complete."
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


    # Iterate through array
    for i in "${!keys[@]}"; do
        key=${keys[$i]}
        prompt=${values[$i]}
        output_dir=$OUTPUT_DIR/inference/$key
        rm -rf $output_dir
        mkdir -p $output_dir
        echo "Category: $key"
        echo "Prompt: $prompt"
        python $RUN_ROOT_PATH/inferance.py --prompt "$prompt" --output $output_dir --n 1500 --batch_size 30 --pretrain $root_path/compare_models/checkpoints/Derm-T2IM.safetensors
    done

fi