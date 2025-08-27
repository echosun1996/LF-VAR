#!/bin/bash
##############################
EVALUATE_MEGA_PRETRAIN=false
EVALUATE_MEGA_PRETRAIN_print=false
EVALUATE_MEGA_Finetune=false
EVALUATE_MEGA_Finetune_print=false

EVALUATE_Derm_T2IM=false
EVALUATE_Derm_T2IM_print=false

EVALUATE_VAR=false
EVALUATE_VAR_print=false

EVALUATE_Dreambooth=false
EVALUATE_Dreambooth_print=false

EVALUATE_MainBAC=false
EVALUATE_MainBAC_print=false

EVALUATE_Main=false
EVALUATE_Main_print=false

# Fixed radiomics features, generate class images
EVALUATE_Main_fixed_radiomics=false
EVALUATE_Main_fixed_radiomics_print=false

# Cross-generation (fixed radiomics features)
EVALUATE_Main_cross_generation=false
EVALUATE_Main_cross_generation_print=false

# Cross-generation (tabsyn-Tabular Data Synthesis)
EVALUATE_Main_cross_generation_tabsyn=false
EVALUATE_Main_cross_generation_tabsyn_print=true

#-----------isic 2017 external---------#
EVALUATE_MEGA_PRETRAIN_external_isic2017=false
EVALUATE_MEGA_PRETRAIN_print_external_isic2017=false
EVALUATE_MEGA_Finetune_external_isic2017=false
EVALUATE_MEGA_Finetune_print_external_isic2017=false

EVALUATE_Derm_T2IM_external_isic2017=false
EVALUATE_Derm_T2IM_print_external_isic2017=false

EVALUATE_VAR_external_isic2017=false
EVALUATE_VAR_print_external_isic2017=false

EVALUATE_Dreambooth_external_isic2017=false
EVALUATE_Dreambooth_print_external_isic2017=false


EVALUATE_Main_external_isic2017=false
EVALUATE_Main_print_external_isic2017=false

#-----------ph2 external---------#
EVALUATE_MEGA_PRETRAIN_external_PH2=false
EVALUATE_MEGA_PRETRAIN_print_external_PH2=false
EVALUATE_MEGA_Finetune_external_PH2=false
EVALUATE_MEGA_Finetune_print_external_PH2=false

EVALUATE_Derm_T2IM_external_PH2=false
EVALUATE_Derm_T2IM_print_external_PH2=false

EVALUATE_VAR_external_PH2=false
EVALUATE_VAR_print_external_PH2=false

EVALUATE_Dreambooth_external_PH2=false
EVALUATE_Dreambooth_print_external_PH2=false

EVALUATE_Main_external_PH2=false
EVALUATE_Main_print_external_PH2=false


#-----------Dermofit external---------#
EVALUATE_MEGA_PRETRAIN_external_Dermofit=false
EVALUATE_MEGA_PRETRAIN_print_external_Dermofit=false
EVALUATE_MEGA_Finetune_external_Dermofit=false
EVALUATE_MEGA_Finetune_print_external_Dermofit=false

EVALUATE_Derm_T2IM_external_Dermofit=false
EVALUATE_Derm_T2IM_print_external_Dermofit=false

EVALUATE_VAR_external_Dermofit=false
EVALUATE_VAR_print_external_Dermofit=false

EVALUATE_Dreambooth_external_Dermofit=false
EVALUATE_Dreambooth_print_external_Dermofit=false

EVALUATE_Main_external_Dermofit=false
EVALUATE_Main_print_external_Dermofit=false

##############################
metrics_dir=$ROOT_PATH/data/metrics
mkdir -p $metrics_dir

# Compare with test/train/val dataset.
COMPARE_WITH="test"
#COMPARE_WITH="train"
###############
# EVALUATE
###############
if [ "$EVALUATE_MEGA_PRETRAIN" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate on [MEGA PRETRAIN class_unconditional_generation]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/MAGE
    generate_dir=$output_dir"/class_unconditional_generation_pre_train/"$folder"/temp6.0-iter20"
    original_dir=$ROOT_PATH"/data/local/HAM10000/input/"$COMPARE_WITH"/HAM10000_img_class/"$folder
    echo "Type: "$folder
    output_path=$metrics_dir"/MEGA_ClassUnconditional_pretrain_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi
if [ "$EVALUATE_MEGA_Finetune" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate on [MEGA Finetune class_unconditional_generation]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/MAGE
    generate_dir=$output_dir"/class_unconditional_generation_finetune/"$folder"/temp6.0-iter20"
    original_dir=$ROOT_PATH"/data/local/HAM10000/input/"$COMPARE_WITH"/HAM10000_img_class/"$folder
    echo "Type: "$folder
    output_path=$metrics_dir"/MEGA_ClassUnconditional_finetune_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi

if [ "$EVALUATE_Derm_T2IM" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate on [Derm_T2IM]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/Derm-T2IM
    generate_dir=$output_dir"/inference/"$folder
    original_dir=$ROOT_PATH"/data/local/HAM10000/input/"$COMPARE_WITH"/HAM10000_img_class/"$folder
    echo "Type: "$folder
    output_path=$metrics_dir"/Derm_T2IM_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi

if [ "$EVALUATE_VAR" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate on [VAR]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/VAR
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/HAM10000/input/"$COMPARE_WITH"/HAM10000_img_class/"$folder
    echo "Type: "$folder
    output_path=$metrics_dir"/VAR_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi


if [ "$EVALUATE_Dreambooth" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate on [Dreambooth/Diffusion]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/Dreambooth/infer
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/HAM10000/input/"$COMPARE_WITH"/HAM10000_img_class/"$folder
    echo "Type: "$folder
    output_path=$metrics_dir"/Dreambooth_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi

if [ "$EVALUATE_MainBAC" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate on [MainBAC]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/main_bac
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/HAM10000/input/"$COMPARE_WITH"/HAM10000_img_class/"$folder
    echo "Type: "$folder
    output_path=$metrics_dir"/MainBAC_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi

if [ "$EVALUATE_Main" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate on [Main]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/main
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/HAM10000/input/"$COMPARE_WITH"/HAM10000_img_class/"$folder
    echo "Type: "$folder
    output_path=$metrics_dir"/Main_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi

#-----external isic2017------#
if [ "$EVALUATE_MEGA_PRETRAIN_external_isic2017" = true ]; then
  folders=("bkl" "mel" "nv")
  echo "Evaluate isic2017 on [MEGA PRETRAIN class_unconditional_generation]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/MAGE
    generate_dir=$output_dir"/class_unconditional_generation_pre_train/"$folder"/temp6.0-iter20"
    original_dir=$ROOT_PATH"/data/local/ISIC2017/input/"$COMPARE_WITH"/HuggingFace/"$folder"/ISIC2017_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_isic2017_MEGA_ClassUnconditional_pretrain_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi
if [ "$EVALUATE_MEGA_Finetune_external_isic2017" = true ]; then
  folders=("bkl" "mel" "nv")
  echo "Evaluate isic2017 on [MEGA Finetune class_unconditional_generation]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/MAGE
    generate_dir=$output_dir"/class_unconditional_generation_finetune/"$folder"/temp6.0-iter20"
    original_dir=$ROOT_PATH"/data/local/ISIC2017/input/"$COMPARE_WITH"/HuggingFace/"$folder"/ISIC2017_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_isic2017_MEGA_ClassUnconditional_finetune_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi

if [ "$EVALUATE_Derm_T2IM_external_isic2017" = true ]; then
  folders=("bkl" "mel" "nv")
  echo "Evaluate isic2017 on [Derm_T2IM]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/Derm-T2IM
    generate_dir=$output_dir"/inference/"$folder
    original_dir=$ROOT_PATH"/data/local/ISIC2017/input/"$COMPARE_WITH"/HuggingFace/"$folder"/ISIC2017_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_isic2017_Derm_T2IM_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi

if [ "$EVALUATE_VAR_external_isic2017" = true ]; then
  folders=("bkl" "mel" "nv")
  echo "Evaluate isic2017 on [VAR]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/VAR
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/ISIC2017/input/"$COMPARE_WITH"/HuggingFace/"$folder"/ISIC2017_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_isic2017_VAR_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi


if [ "$EVALUATE_Dreambooth_external_isic2017" = true ]; then
  folders=("bkl" "mel" "nv")
  echo "Evaluate isic2017 on [Dreambooth/Diffusion]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/External_Comparison/ISIC2017/Dreambooth
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/ISIC2017/input/"$COMPARE_WITH"/HuggingFace/"$folder"/ISIC2017_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_isic2017_Dreambooth_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi


if [ "$EVALUATE_Main_external_isic2017" = true ]; then
  folders=("bkl" "mel" "nv")
  echo "Evaluate isic2017 on [Main]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/External_Comparison/ISIC2017/main
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/ISIC2017/input/"$COMPARE_WITH"/HuggingFace/"$folder"/ISIC2017_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_isic2017_Main_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi


#-----external ph2------#
if [ "$EVALUATE_MEGA_PRETRAIN_external_PH2" = true ]; then
  folders=("mel" "nv")
  echo "Evaluate PH2 on [MEGA PRETRAIN class_unconditional_generation]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/MAGE
    generate_dir=$output_dir"/class_unconditional_generation_pre_train/"$folder"/temp6.0-iter20"
    original_dir=$ROOT_PATH"/data/local/PH2/input/"$COMPARE_WITH"/HuggingFace/"$folder"/PH2_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_PH2_MEGA_ClassUnconditional_pretrain_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi
if [ "$EVALUATE_MEGA_Finetune_external_PH2" = true ]; then
  folders=("mel" "nv")
  echo "Evaluate PH2 on [MEGA Finetune class_unconditional_generation]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/MAGE
    generate_dir=$output_dir"/class_unconditional_generation_finetune/"$folder"/temp6.0-iter20"
    original_dir=$ROOT_PATH"/data/local/PH2/input/"$COMPARE_WITH"/HuggingFace/"$folder"/PH2_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_PH2_MEGA_ClassUnconditional_finetune_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi

if [ "$EVALUATE_Derm_T2IM_external_PH2" = true ]; then
  folders=("mel" "nv")
  echo "Evaluate PH2 on [Derm_T2IM]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/Derm-T2IM
    generate_dir=$output_dir"/inference/"$folder
    original_dir=$ROOT_PATH"/data/local/PH2/input/"$COMPARE_WITH"/HuggingFace/"$folder"/PH2_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_PH2_Derm_T2IM_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi

if [ "$EVALUATE_VAR_external_PH2" = true ]; then
  folders=("mel" "nv")
  echo "Evaluate PH2 on [VAR]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/VAR
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/PH2/input/"$COMPARE_WITH"/HuggingFace/"$folder"/PH2_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_PH2_VAR_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi


if [ "$EVALUATE_Dreambooth_external_PH2" = true ]; then
  folders=("mel" "nv")
  echo "Evaluate PH2 on [Dreambooth/Diffusion]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/External_Comparison/PH2/Dreambooth
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/PH2/input/"$COMPARE_WITH"/HuggingFace/"$folder"/PH2_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_PH2_Dreambooth_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi


if [ "$EVALUATE_Main_external_PH2" = true ]; then
  folders=("mel" "nv")
  echo "Evaluate PH2 on [Main]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/External_Comparison/PH2/main
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/PH2/input/"$COMPARE_WITH"/HuggingFace/"$folder"/PH2_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_PH2_Main_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi


#-----external Dermofit------#
if [ "$EVALUATE_MEGA_PRETRAIN_external_Dermofit" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate Dermofit on [MEGA PRETRAIN class_unconditional_generation]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/MAGE
    generate_dir=$output_dir"/class_unconditional_generation_pre_train/"$folder"/temp6.0-iter20"
    original_dir=$ROOT_PATH"/data/local/Dermofit/input/"$COMPARE_WITH"/HuggingFace/"$folder"/Dermofit_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_Dermofit_MEGA_ClassUnconditional_pretrain_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi
if [ "$EVALUATE_MEGA_Finetune_external_Dermofit" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate Dermofit on [MEGA Finetune class_unconditional_generation]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/MAGE
    generate_dir=$output_dir"/class_unconditional_generation_finetune/"$folder"/temp6.0-iter20"
    original_dir=$ROOT_PATH"/data/local/Dermofit/input/"$COMPARE_WITH"/HuggingFace/"$folder"/Dermofit_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_Dermofit_MEGA_ClassUnconditional_finetune_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi

if [ "$EVALUATE_Derm_T2IM_external_Dermofit" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate Dermofit on [Derm_T2IM]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/Derm-T2IM
    generate_dir=$output_dir"/inference/"$folder
    original_dir=$ROOT_PATH"/data/local/Dermofit/input/"$COMPARE_WITH"/HuggingFace/"$folder"/Dermofit_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_Dermofit_Derm_T2IM_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi

if [ "$EVALUATE_VAR_external_Dermofit" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate Dermofit on [VAR]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/VAR
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/Dermofit/input/"$COMPARE_WITH"/HuggingFace/"$folder"/Dermofit_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_Dermofit_VAR_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi


if [ "$EVALUATE_Dreambooth_external_Dermofit" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate Dermofit on [Dreambooth/Diffusion]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/External_Comparison/Dermofit/Dreambooth
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/Dermofit/input/"$COMPARE_WITH"/HuggingFace/"$folder"/Dermofit_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_Dermofit_Dreambooth_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi


if [ "$EVALUATE_Main_external_Dermofit" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate Dermofit on [Main]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/External_Comparison/Dermofit/main
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/Dermofit/input/"$COMPARE_WITH"/HuggingFace/"$folder"/Dermofit_img_class"
    echo "Type: "$folder
    output_path=$metrics_dir"/external_Dermofit_Main_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi

# Fixed radiomics features
if [ "$EVALUATE_Main_fixed_radiomics" = true ]; then
  folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate on [Main_fixed_radiomics]:"
  for folder in "${folders[@]}"; do
    output_dir=$ROOT_PATH/data/compare_results/main_fixed_radiomics
    generate_dir=$output_dir"/"$folder
    original_dir=$ROOT_PATH"/data/local/HAM10000/input/"$COMPARE_WITH"/HAM10000_img_class/"$folder
    echo "Type: "$folder
    output_path=$metrics_dir"/fixed_radiomics_Main_"$folder"_"$COMPARE_WITH".txt"
    python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
  done
fi

# Cross-generation (tabsyn)
if [ "$EVALUATE_Main_cross_generation_tabsyn" = true ]; then
  folders_A=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  folders_B=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate on [Main_cross_generation_tabsyn]:"
  for folder_a in "${folders_A[@]}"; do
      for folder_b in "${folders_B[@]}"; do
        output_dir=$ROOT_PATH/data/compare_results/main_cross_infer_tabsyn
        generate_dir=$output_dir"/"$folder_a"_"$folder_b
        original_dir=$ROOT_PATH"/data/local/HAM10000/input/"$COMPARE_WITH"/HAM10000_img_class/"$folder_b
        echo "Type: "$folder_a" to "$folder_b
        output_path=$metrics_dir"/cross_generation_Main_tabsyn_"$folder_a"_"$folder_b"_"$COMPARE_WITH".txt"
        python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
    done
  done
fi



# Cross-generation (fixed radiomics features)
if [ "$EVALUATE_Main_cross_generation" = true ]; then
  folders_A=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  folders_B=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
  echo "Evaluate on [Main_cross_generation]:"
  for folder_a in "${folders_A[@]}"; do
      for folder_b in "${folders_B[@]}"; do
        output_dir=$ROOT_PATH/data/compare_results/main_cross_infer
        generate_dir=$output_dir"/"$folder_a"_"$folder_b
        original_dir=$ROOT_PATH"/data/local/HAM10000/input/"$COMPARE_WITH"/HAM10000_img_class/"$folder_b
        echo "Type: "$folder_a" to "$folder_b
        output_path=$metrics_dir"/cross_generation_Main_"$folder_a"_"$folder_b"_"$COMPARE_WITH".txt"
        python $ROOT_PATH/py_scripts/metrics/cal_metrics.py --input1 ${generate_dir} --input2 ${original_dir} --output ${output_path}
    done
  done
fi

###############
# Print
###############
if [ "$EVALUATE_MEGA_PRETRAIN_print" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate MEGA_ClassUnconditional_pretrain --compare $COMPARE_WITH
fi

if [ "$EVALUATE_MEGA_Finetune_print" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate MEGA_ClassUnconditional_finetune --compare $COMPARE_WITH
fi

if [ "$EVALUATE_Derm_T2IM_print" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate Derm_T2IM --compare $COMPARE_WITH
fi

if [ "$EVALUATE_VAR_print" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate VAR --compare $COMPARE_WITH
fi

if [ "$EVALUATE_Dreambooth_print" = true ]; then
  echo "Dreambooth/Diffusion evaluate results"
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate Dreambooth --compare $COMPARE_WITH
fi

if [ "$EVALUATE_Main_print" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate Main --compare $COMPARE_WITH
fi

if [ "$EVALUATE_MainBAC_print" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate MainBAC --compare $COMPARE_WITH
fi


#-----external isic2017------#
if [ "$EVALUATE_MEGA_PRETRAIN_print_external_isic2017" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_isic2017_MEGA_ClassUnconditional_pretrain --compare $COMPARE_WITH
fi

if [ "$EVALUATE_MEGA_Finetune_print_external_isic2017" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_isic2017_MEGA_ClassUnconditional_finetune --compare $COMPARE_WITH
fi

if [ "$EVALUATE_Derm_T2IM_print_external_isic2017" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_isic2017_Derm_T2IM --compare $COMPARE_WITH
fi

if [ "$EVALUATE_VAR_print_external_isic2017" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_isic2017_VAR --compare $COMPARE_WITH
fi

if [ "$EVALUATE_Dreambooth_print_external_isic2017" = true ]; then
  echo "Dreambooth/Diffusion evaluate results"
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_isic2017_Dreambooth --compare $COMPARE_WITH
fi

if [ "$EVALUATE_Main_print_external_isic2017" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_isic2017_Main --compare $COMPARE_WITH
fi


#-----external PH2------#
if [ "$EVALUATE_MEGA_PRETRAIN_print_external_PH2" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_PH2_MEGA_ClassUnconditional_pretrain --compare $COMPARE_WITH
fi

if [ "$EVALUATE_MEGA_Finetune_print_external_PH2" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_PH2_MEGA_ClassUnconditional_finetune --compare $COMPARE_WITH
fi

if [ "$EVALUATE_Derm_T2IM_print_external_PH2" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_PH2_Derm_T2IM --compare $COMPARE_WITH
fi

if [ "$EVALUATE_VAR_print_external_PH2" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_PH2_VAR --compare $COMPARE_WITH
fi

if [ "$EVALUATE_Dreambooth_print_external_PH2" = true ]; then
  echo "Dreambooth/Diffusion evaluate results"
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_PH2_Dreambooth --compare $COMPARE_WITH
fi

if [ "$EVALUATE_Main_print_external_PH2" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_PH2_Main --compare $COMPARE_WITH
fi


#-----external Dermofit------#
if [ "$EVALUATE_MEGA_PRETRAIN_print_external_Dermofit" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_Dermofit_MEGA_ClassUnconditional_pretrain --compare $COMPARE_WITH
fi

if [ "$EVALUATE_MEGA_Finetune_print_external_Dermofit" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_Dermofit_MEGA_ClassUnconditional_finetune --compare $COMPARE_WITH
fi

if [ "$EVALUATE_Derm_T2IM_print_external_Dermofit" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_Dermofit_Derm_T2IM --compare $COMPARE_WITH
fi

if [ "$EVALUATE_VAR_print_external_Dermofit" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_Dermofit_VAR --compare $COMPARE_WITH
fi

if [ "$EVALUATE_Dreambooth_print_external_Dermofit" = true ]; then
  echo "Dreambooth/Diffusion evaluate results"
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_Dermofit_Dreambooth --compare $COMPARE_WITH
fi

if [ "$EVALUATE_Main_print_external_Dermofit" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate external_Dermofit_Main --compare $COMPARE_WITH
fi

# Fixed radiomics features
if [ "$EVALUATE_Main_fixed_radiomics_print" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics.py --metrics_dir $metrics_dir --evaluate fixed_radiomics_Main --compare $COMPARE_WITH
fi

# Cross-generation (fixed radiomics features)
if [ "$EVALUATE_Main_cross_generation_print" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics_cross_generation.py --metrics_dir $metrics_dir --evaluate cross_generation_Main --compare $COMPARE_WITH
fi

# Cross-generation (tabsyn)
if [ "$EVALUATE_Main_cross_generation_tabsyn_print" = true ]; then
  python $ROOT_PATH/py_scripts/metrics/print_metrics_cross_generation.py --metrics_dir $metrics_dir --evaluate cross_generation_Main_tabsyn --compare $COMPARE_WITH
fi