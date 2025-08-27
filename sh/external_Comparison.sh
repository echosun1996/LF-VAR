#!/bin/bash
##############################
# External Comparison
# MEGA, MEGA Adapter, Derm, VAR compare directly.
# Diffusion and ours need to re-generate.

# compare with ISIC7,8,9
# Dermofit, PH2
COMPARE_ISIC2017=true
COMPARE_Dermofit=false
COMPARE_PH2=true
##############################
root_path=$(pwd)
OUTPUT_DIR=$root_path/data/compare_results/External_Comparison
dataset_dir=$root_path/data/local

dreambooth_root_path=$root_path/compare_models/run/Dreambooth
dreambooth_model_path=$root_path/data/compare_results/Dreambooth

main_root_path=$root_path/main

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

if [ "$COMPARE_ISIC2017" = true ]; then
    echo "Comparing ISIC2017..."

    # generate metadata.csv
    # from ISIC-2017_Test_v2_Part3_GroundTruth.csv  generate "metadata.csv"
    in_csv_path=$dataset_dir"/ISIC2017/original/ISIC-2017_Test_v2_Part3_GroundTruth.csv"
    out_csv_path=$dataset_dir"/ISIC2017/input/metadata.csv"
    
    # Check if input file exists
    if [ ! -f "$in_csv_path" ]; then
        echo "Error: Input file not found: $in_csv_path"
        exit 1
    fi
    
    # If output file doesn't exist, execute generation
    if [ ! -f "$out_csv_path" ]; then
        echo "Generating metadata file: $out_csv_path"
        
        # Ensure output directory exists
        out_dir=$(dirname "$out_csv_path")
        mkdir -p "$out_dir"
        
        # Execute Python script
        RUN_ROOT_PATH=$root_path/py_scripts/preprocessing
        cd $RUN_ROOT_PATH
        python generate_metadata_ISIC2017.py \
            --root_path $root_path \
            --in_csv_path $in_csv_path \
            --out_csv_path $out_csv_path \
            --keys "${keys[@]}" \
            --values "${values[@]}"
    else
        echo "Metadata file already exists: $out_csv_path"
    fi

    # generate radiomics features.
    echo "Run Radiomics on ISIC2017"
          radiomics_output="$dataset_dir/ISIC2017/original/ISIC_2017_Test/radiomics/5.Finial.csv"
    
    # Check if radiomics output file exists
    if [ ! -f "$radiomics_output" ]; then
        echo "Generating Radiomics features: $radiomics_output"
        
        # Ensure output directory exists
        radiomics_dir=$(dirname "$radiomics_output")
        mkdir -p "$radiomics_dir"
        
        # Execute Radiomics feature extraction
        cd $root_path/main"/radiomics/"
        meta_path=$dataset_dir/ISIC2017/input/metadata.csv
        python radiomics_main.py \
            --root-path $dataset_dir"/ISIC2017/original/ISIC_2017_Test" \
            --meta-path $meta_path \
            --img-folder-name "ISIC-2017_Test_v2_Data" \
            --seg-folder-name "ISIC-2017_Test_v2_Part1_GroundTruth"
    else
        echo "Radiomics features file already exists: $radiomics_output"
    fi

    cp $radiomics_output $dataset_dir"/ISIC2017/input/radiomics.csv"

    #  Compare Diffusion \
    output_path=$OUTPUT_DIR/ISIC2017/Dreambooth
    # Check if output_path exists, execute if not
    if [ ! -d "$output_path" ]; then
        mkdir -p "$output_path"
        cd "$dreambooth_root_path"
        metadata_file="$dataset_dir/ISIC2017/input/metadata.csv"
        python inference.py --model_dir "$dreambooth_model_path/infer" --metadata_file "$metadata_file" --output "$output_path" --data_root "$root_path"
    fi

    # Generate HuggingFace folder
    output_path=$dataset_dir/ISIC2017/input/test/HuggingFace
    if [ ! -d "$output_path" ]; then
        mkdir -p "$output_path"
        # Execute Python script
        RUN_ROOT_PATH=$root_path/py_scripts/preprocessing
        cd $RUN_ROOT_PATH
        python generate_HuggingFace_folder.py \
        --meta-path $dataset_dir/ISIC2017/input/metadata.csv \
        --root-path $root_path\
        --output-path $output_path \
        --dataset "ISIC2017"
    fi

    #  Compare with ours
    output_path=$OUTPUT_DIR/ISIC2017/main
    if [ ! -d "$output_path" ]; then
      original_main_path=$root_path"/data/compare_results/main/"
      mkdir -p $output_path

      cp $original_main_path"/ar-ckpt-best.pth" $output_path"/"
      cp $original_main_path"/ar-ckpt-last.pth" $output_path"/"

    fi
    cd $main_root_path
    MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
    CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT infer.py \
    --depth=16 --bs=35 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --local_out_dir_path $output_path --data_path $dataset_dir"/ISIC2017/input" --dataset "ISIC2017"


fi


########################################################################################

if [ "$COMPARE_Dermofit" = true ]; then
    echo "Comparing Dermofit..."

    # generate metadata.csv
    in_path=$dataset_dir"/Dermofit/original/lesionlist.txt"
    out_path=$dataset_dir"/Dermofit/input/metadata.csv"

    # Check if input file exists
    if [ ! -f "$in_path" ]; then
        echo "Error: Input file not found: $in_path"
        exit 1
    fi

    # If output file does not exist, execute generation
    if [ ! -f "$out_path" ]; then
        echo "Generating metadata file: $out_path"

        # Ensure output directory exists
        out_dir=$(dirname "$out_path")
        mkdir -p "$out_dir"

        # Execute Python script
        RUN_ROOT_PATH=$root_path/py_scripts/preprocessing
        cd $RUN_ROOT_PATH
        python generate_metadata_Dermofit.py \
            --root_path $root_path \
            --in_path $in_path \
            --out_path $out_path \
            --keys "${keys[@]}" \
            --values "${values[@]}"
    else
        echo "Metadata file already exists: $out_csv_path"
    fi

    # generate radiomics features.
    echo "Run Radiomics on Dermofit"
    radiomics_output="$dataset_dir/Dermofit/original/radiomics/5.Finial.csv"
    
    # Check if radiomics output file exists
    if [ ! -f "$radiomics_output" ]; then
        echo "Generating Radiomics features: $radiomics_output"

        # Execute Radiomics feature extraction
        cd $root_path/main"/radiomics/"
        meta_path=$dataset_dir/Dermofit/input/metadata.csv
        python radiomics_main.py \
            --root-path $dataset_dir"/Dermofit/original" \
            --meta-path $meta_path \
            --img-folder-name "Dermofit_Data" \
            --seg-folder-name "Dermofit_GroundTruth"
    else
        echo "Radiomics features file already exists: $radiomics_output"
    fi

    cp $radiomics_output $dataset_dir"/Dermofit/input/radiomics.csv"

    #  Compare Diffusion \
    output_path=$OUTPUT_DIR/Dermofit/Dreambooth
    # Check if output_path exists, execute if not
    if [ ! -d "$output_path" ]; then
        mkdir -p "$output_path"
        cd "$dreambooth_root_path"
        metadata_file="$dataset_dir/Dermofit/input/metadata.csv"
        python inference.py --model_dir "$dreambooth_model_path/infer" --metadata_file "$metadata_file" --output "$output_path" --data_root "$root_path"
    fi

    # Generate HuggingFace folder
    output_path=$dataset_dir/Dermofit/input/test/HuggingFace
    if [ ! -d "$output_path" ]; then
        mkdir -p "$output_path"
        # Execute Python script
        RUN_ROOT_PATH=$root_path/py_scripts/preprocessing
        cd $RUN_ROOT_PATH
        python generate_HuggingFace_folder.py \
        --meta-path $dataset_dir/Dermofit/input/metadata.csv \
        --root-path $root_path\
        --output-path $output_path \
        --dataset "Dermofit"
    fi


    #  Compare with ours
    output_path=$OUTPUT_DIR/Dermofit/main
    if [ ! -d "$output_path" ]; then
      original_main_path=$root_path"/data/compare_results/main/"
      mkdir -p $output_path
      cp $original_main_path"/ar-ckpt-best.pth" $output_path"/"
      cp $original_main_path"/ar-ckpt-last.pth" $output_path"/"
    fi

    cd $main_root_path
    MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
    CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT infer.py \
    --depth=16 --bs=35 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --local_out_dir_path $output_path --data_path $dataset_dir"/Dermofit/input" --dataset "Dermofit"


fi

########################################################################################

if [ "$COMPARE_PH2" = true ]; then
    echo "Comparing PH2..."

    # generate metadata.csv
    in_path=$dataset_dir"/PH2/original/PH2_dataset.txt"
    out_path=$dataset_dir"/PH2/input/metadata.csv"

    # Check if input file exists
    if [ ! -f "$in_path" ]; then
        echo "Error: Input file not found: $in_path"
        exit 1
    fi

    # If output file does not exist, execute generation
    if [ ! -f "$out_path" ]; then
        echo "Generating metadata file: $out_path"

        # Ensure output directory exists
        out_dir=$(dirname "$out_path")
        mkdir -p "$out_dir"

        # Execute Python script
        RUN_ROOT_PATH=$root_path/py_scripts/preprocessing
        cd $RUN_ROOT_PATH
        python generate_metadata_PH2.py \
            --root_path $root_path \
            --in_path $in_path \
            --out_path $out_path \
            --keys "${keys[@]}" \
            --values "${values[@]}"
    else
        echo "Metadata file already exists: $out_csv_path"
    fi

    # generate radiomics features.
    echo "Run Radiomics on PH2"
    radiomics_output="$dataset_dir/PH2/original/radiomics/5.Finial.csv"

    # Check if radiomics output file exists
    if [ ! -f "$radiomics_output" ]; then
        echo "Generating Radiomics features: $radiomics_output"

        # Execute Radiomics feature extraction
        cd $root_path/main"/radiomics/"
        meta_path=$dataset_dir/PH2/input/metadata.csv
        python radiomics_main.py \
            --root-path $dataset_dir"/PH2/original" \
            --meta-path $meta_path \
            --img-folder-name "PH2_Data" \
            --seg-folder-name "PH2_GroundTruth"
    else
        echo "Radiomics features file already exists: $radiomics_output"
    fi

    cp $radiomics_output $dataset_dir"/PH2/input/radiomics.csv"

    # Compare Diffusion
    output_path=$OUTPUT_DIR/PH2/Dreambooth
    if [ ! -d "$output_path" ]; then
        mkdir -p "$output_path"
        cd "$dreambooth_root_path"
        metadata_file="$dataset_dir/PH2/input/metadata.csv"
        python inference.py --model_dir "$dreambooth_model_path/infer" --metadata_file "$metadata_file" --output "$output_path" --data_root "$root_path"
    fi

    # Generate HuggingFace folder
    output_path=$dataset_dir/PH2/input/test/HuggingFace
    if [ ! -d "$output_path" ]; then
      # Execute Python script
      RUN_ROOT_PATH=$root_path/py_scripts/preprocessing
      cd $RUN_ROOT_PATH
      python generate_HuggingFace_folder.py \
      --meta-path $dataset_dir/PH2/input/metadata.csv \
      --root-path $root_path\
      --output-path $output_path \
      --dataset "PH2"
    fi

    #  Compare with ours
    output_path=$OUTPUT_DIR/PH2/main
    if [ ! -d "$output_path" ]; then
      original_main_path=$root_path"/data/compare_results/main/"
      mkdir -p $output_path

      cp $original_main_path"/ar-ckpt-best.pth" $output_path"/"
      cp $original_main_path"/ar-ckpt-last.pth" $output_path"/"

    fi
    cd $main_root_path
    MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
    CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT infer.py \
    --depth=16 --bs=35 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --local_out_dir_path $output_path --data_path $dataset_dir"/PH2/input" --dataset "PH2"


fi