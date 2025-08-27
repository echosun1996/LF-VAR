#!/bin/bash
##############################
# RadiomicsFill-Mammo
# https://github.com/nainye/RadiomicsFill
INIT_RadiomicsFill=false
COMPARE_RadiomicsFill=false
HEAD2_RadiomicsFill=true
##############################
home_dir="$HOME"
root_path=$(pwd)
OUTPUT_DIR=$root_path/data/compare_results/RadiomicsFill
dataset_dir=$root_path/data/local
RUN_ROOT_PATH=$root_path/compare_models/run/RadiomicsFill

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


if [ "$INIT_RadiomicsFill" = true ]; then
    echo "Comparing RadiomicsFill..."
    mkdir -p $root_path/compare_models/reps/
    cd $root_path/compare_models/reps/

    if [ -d "RadiomicsFill" ];
    then
        echo "[√] RadiomicsFill repo already Downloaded."
    else
        echo "Downloading MARadiomicsFill repo..."
        git clone https://github.com/nainye/RadiomicsFill.git RadiomicsFill
    fi
  
fi

if [ "$COMPARE_RadiomicsFill" = true ]; then
    echo "Comparing RadiomicsFill-Mammo..."

    # Need to save radiomics features from "data/local/HAM10000/input/radiomics_finial.csv" to train/test/val csv file
    cd $root_path/compare_models/run/RadiomicsFill
    python radiomics_separate.py --data_path $dataset_dir"/HAM10000/input/radiomics_final.csv" --input_path $dataset_dir"/HAM10000/input" --output_path $OUTPUT_DIR

    cd $root_path/compare_models/reps/RadiomicsFill
    bash ./scripts/train_MET_VinDr-Mammo_embed32_enc6_dec3.sh

fi



if [ "$HEAD2_RadiomicsFill" = true ]; then
    echo "HEAD2 RadiomicsFill-Mammo..."

    cd $root_path/compare_models/reps/RadiomicsFill
    bash ./scripts/HEAD2.sh

fi