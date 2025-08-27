#!/bin/bash
##############################
# MixedType Tabular Data Synthesis
# https://github.com/amazon-science/tabsyn
INIT_MixedTypeTabular=false
Training_MixedTypeTabular=false
Synthesis_MixedTypeTabular=true


##############################
home_dir="$HOME"
root_path=$(pwd)
OUTPUT_DIR=$root_path/data/compare_results/MixedTypeTabular
dataset_dir=$root_path/data/local
RUN_ROOT_PATH=$root_path/compare_models/run/MixedTypeTabular

echo "### Total GPUs with usage < 20%: $gpu_sum"
echo "### GPU IDs: $gpu_id"

mkdir -p $OUTPUT_DIR


conda activate tabsyn
# Define categories and their corresponding prompts
# number  = 1-7
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


if [ "$INIT_MixedTypeTabular" = true ]; then
    echo "Comparing MixedTypeTabular..."
    mkdir -p $root_path/compare_models/reps/
    cd $root_path/compare_models/reps/

    if [ -d "MixedTypeTabular" ];
    then
        echo "[√] MixedTypeTabular repo already Downloaded."
    else
        echo "Downloading MixedTypeTabular repo..."
        git clone https://github.com/amazon-science/tabsyn.git MixedTypeTabular
    fi
    
    echo "Comparing MixedTypeTabular..."
    cd $root_path/compare_models/run/MixedTypeTabular
    python convert_category_to_number.py --path $root_path/compare_models/reps/MixedTypeTabular/data/HAM10000/radiomics_final_test.csv
    python convert_category_to_number.py --path $root_path/compare_models/reps/MixedTypeTabular/data/HAM10000/radiomics_final_train.csv
    python convert_category_to_number.py --path $root_path/compare_models/reps/MixedTypeTabular/data/HAM10000/radiomics_final_trainval.csv
    python convert_category_to_number.py --path $root_path/compare_models/reps/MixedTypeTabular/data/HAM10000/radiomics_final_val.csv
    cd $root_path/compare_models/reps/MixedTypeTabular
    python process_dataset.py --dataname HAM10000
fi
if [ "$Training_MixedTypeTabular" = true ]; then
    echo "Training MixedTypeTabular..."
    cd $root_path/compare_models/reps/MixedTypeTabular
    echo "Training MixedTypeTabular [vae]..."
    python main.py --dataname HAM10000 --method vae --mode train
    echo "Training MixedTypeTabular [tabsyn]..."
    python main.py --dataname HAM10000 --method tabsyn --mode train
fi
if [ "$Synthesis_MixedTypeTabular" = true ]; then
    echo "Synthesis MixedTypeTabular..."
    cd $root_path/compare_models/reps/MixedTypeTabular
    python main.py --dataname HAM10000 --method tabsyn --mode sample --save_path $OUTPUT_DIR"/HAM10000_tabsyn.csv"
fi