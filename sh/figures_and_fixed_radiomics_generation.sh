#!/bin/bash
set -e
##############################
# Generate the figures and calculate the

Draw_and_calculate=false # Deprecated
clustering_pca_figures=true
clustering_compare_figures=true
clustering_compare_all_figures=true
##############################
home_dir="$HOME"
root_path=$(pwd)
fig_out_path=$root_path/figures


HAM10000_radiomics_path=$root_path/data/local/HAM10000/input
HAM10000_radiomics_input_path=$HAM10000_radiomics_path/radiomics_finial.csv
HAM10000_radiomics_fixed_path=$HAM10000_radiomics_path/radiomics_fixed.csv


mkdir -p $fig_out_path




# Deprecated
if [ "$Draw_and_calculate" = true ]; then
  python $root_path/py_scripts/vis/clustering_pca_plot.py --input-file-path $HAM10000_radiomics_input_path --output-figure-path $fig_out_path/cluster.png --output-csv-path $HAM10000_radiomics_fixed_path
fi


if [ "$clustering_pca_figures" = true ]; then
  input_path=$HAM10000_radiomics_path/test/HuggingFace
  python $root_path/py_scripts/vis/clustering_pca_figures.py --input-image-root-path $input_path --output-figure-path $fig_out_path/HAM10000_original_cluster.png
fi

if [ "$clustering_compare_figures" = true ]; then
  input_path=$HAM10000_radiomics_path/train/HuggingFace
  generate_path=$root_path"/data/compare_results/main"
  class_name="mel"
  python $root_path/py_scripts/vis/clustering_compare_figures.py --input-image-root-path $input_path --generate-image-root-path $generate_path --output-figure-path $fig_out_path/HAM10000_compare_$class_name.png --class-name $class_name

fi

if [ "$clustering_compare_all_figures" = true ]; then
  input_path=$HAM10000_radiomics_path/train/HuggingFace
  input_center_path=$HAM10000_radiomics_path/train/HuggingFace
  generate_main_path=$root_path"/data/compare_results/main"
  generate_var_path=$root_path"/data/compare_results/VAR"
  generate_Diffusion_path=$root_path"/data/compare_results/Dreambooth/infer"
  generate_Derm_T2IM_path=$root_path"/data/compare_results/Derm-T2IM/inference"

  generate_image_paths="$input_center_path $generate_main_path $generate_var_path $generate_Diffusion_path $generate_Derm_T2IM_path"
  model_names="Center Ours VAR Diffusion Derm-T2IM"

#  # test
#  generate_image_paths="$input_path $generate_main_path $generate_var_path "
#  model_names="Center Ours VAR"


  python $root_path/py_scripts/vis/clustering_compare_all_figures.py --input-image-root-path $input_path \
      --generate-image-root-path $generate_image_paths \
      --model-names $model_names \
      --output-figure-path $fig_out_path/HAM10000_all_compare.png
fi


echo "[√] Figures generate finish!"