#!/bin/bash

# If parameters are passed, assign them to choice, otherwise prompt user input
if [[ -n $1 ]]; then
    choice=$1
else
    # Display welcome interface and options
    echo "Please enter the corresponding number to choose a program to execute:"
    echo "1) Init Server." # init
    echo "2) Download Dataset."
    echo "3) Compare MAGE"
    echo "4) Compare VQGAN" # https://github.com/CompVis/taming-transformers
    echo "5) Compare stable-diffusion-2-inpainting" # https://github.com/CompVis/taming-transformers
    echo "6) Compare dreambooth"
    echo "7) Compare Derm-T2IM"
    echo "8) Compare VAR" # https://github.com/FoundationVision/VAR
    echo "9) GPU Check"
    echo "f) Figures and Fixed Radiomics Features Generation"
    echo "r) Compare RadiomicsFill-Mammo" 
    echo "t) Compare Mixed-Type Tabular Data Synthesis" 
    echo "x) External Comparison"
    echo "m) Compare MAIN"
    echo "e) Evaluate all methods"
    echo "ELSE) Exit"

    # Read user input
    read -p "Choose an option: " choice
fi
ROOT_PATH=$(pwd)
root_path=$(pwd)
# Execute the corresponding program based on the user's choice
case $choice in
    1)
        echo "Start init the server..."
        source ./sh/init_server.sh
        ;;
    2)
        echo "Start download the dataset..."
        source ./sh/compare_ENV.sh
        source ./sh/dataset_downloader.sh
        ;;
    3)
        echo "Compare MAGE..." # Finish
        source ./sh/GPU_check.sh
        source ./sh/compare_ENV.sh
        source ./sh/compare_mega.sh
        ;;
    4)
        echo "Compare VQGAN..."
        source ./sh/GPU_check.sh
        source ./sh/compare_ENV.sh
        source ./sh/compare_vqgan.sh
        ;;
    5)
        echo "Compare stable-diffusion-2-inpainting..."
        source ./sh/GPU_check.sh
        source ./sh/compare_ENV.sh
        source ./sh/compare_stable_diffusion2_inpainting.sh
        ;;
    6)
        echo "Compare dreambooth..."
        source ./sh/GPU_check.sh
        source ./sh/compare_ENV.sh
        source ./sh/compare_dreambooth.sh
        ;;
    7)
        echo "Compare Derm-T2IM..." # Finish
        source ./sh/GPU_check.sh
        source ./sh/compare_ENV.sh
        source ./sh/compare_Derm_T2IM.sh
        ;;
    8)
        echo "Compare VAR..."
        source ./sh/GPU_check.sh
        source ./sh/compare_ENV.sh
        source ./sh/compare_VAR.sh
        ;;
    9)
        echo "GUP test!"
        source ./sh/compare_ENV.sh
        python py_scripts/sys_check/gpu_check.py
        exit 0
        ;;
    r)
        echo "Compare RadiomicsFill-Mammo..." # Finish
        source ./sh/GPU_check.sh
        source ./sh/compare_ENV.sh
        source ./sh/compare_RadiomicsFill_Mammo.sh
        ;;
    t)
        echo "Compare Mixed-Type Tabular Data Synthesis..." # Finish
        source ./sh/GPU_check.sh
        source ./sh/compare_ENV.sh
        source ./sh/compare_Mixed-TypeTabular.sh
        ;;
    x)
        echo "External Comparison..." # Finish
        source ./sh/GPU_check.sh
        source ./sh/compare_ENV.sh
        source ./sh/external_Comparison.sh
        ;;
    f)
        echo "Figures and Fix Radiomics Features Generation..."
        source ./sh/compare_ENV.sh
        source ./sh/figures_and_fixed_radiomics_generation.sh
        ;;
    m)
        echo "Compare MAIN..."
        source ./sh/GPU_check.sh
        source ./sh/compare_ENV.sh
        source ./sh/compare_MAIN.sh
        ;;

    e)
        echo "Evaluate all methods..."
        source ./sh/GPU_check.sh
        source ./sh/compare_ENV.sh
        source ./sh/metrics.sh
        ;;
    *)
        echo "Invalid option, exit."
        ;;
esac