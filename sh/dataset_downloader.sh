#!/bin/bash
####################################################
# Download Datasets
split_and_resize=false
ham_train_rate=0.8
ham_val_rate=0.1
img_resize=512
train_rate=0.9

move_class=false
prepare_VAR=false
check_dataset=false
####################################################
root_path=$(pwd)
# Suspending the script if any command fails
current_user=$(whoami)

# Check data folder
if [ ! -d "$root_path/data" ]; then
    echo "Error: 'data' folder not found in the current directory. Init first!"
    exit 1
fi
dataset_download_dir=$root_path/data/cloud/download

if [ ! -d "$dataset_download_dir"  ]; then
  mkdir -p $dataset_download_dir
fi

dataset_dir=$root_path/data/local
mkdir -p dataset_dir

############################
# Download all dataset    ##
############################
echo "Downloing original datasets..."
# Create folder for each dataset
mkdir -p $dataset_dir/HAM10000/original
mkdir -p $dataset_dir/ISIC2016/original
mkdir -p $dataset_dir/ISIC2017/original
mkdir -p $dataset_dir/ISIC2019/original

# Define function to download and verify files
download_and_verify() {
    local file=$1            # Filename
    local expected_hash=$2   # Expected hash value
    local url=$3             # Download URL

    # Check if file exists
    if [ ! -f "$file" ]; then
        wget -O "$file" "$url"
    fi

    # Calculate current file's md5 hash value
    current_hash=$(md5sum "$file" | awk '{ print $1 }')
    echo "MD5:"$current_hash

    # Compare current calculated hash value with expected hash value
    if [ "$current_hash" == "$expected_hash" ]; then
        echo $file" download finished and valied."
    else
        echo $file" hash not mathch with "$expected_hash
        rm -f "$file"  # Delete incomplete file
        exit 1
    fi
}

# HAM10000
cd $dataset_dir/HAM10000/original
## HAM10000 images
file_count=-1
if [ -d "HAM10000_images" ]; then
    file_count=$(ls -1 HAM10000_images | wc -l)
fi
if [ $file_count -ne 10015 ]; then
    cd $dataset_dir/HAM10000/original
    rm -rf HAM10000_images
    cd $dataset_download_dir

    download_and_verify "HAM10000_images_part_1.zip" "4639bfa73ab251610530a97c898e6e46" "https://drive.usercontent.google.com/download?id=1CbvFkDvngn5WqE-VJFepXRjo8rFuwJ31&export=download&authuser=0&confirm=t&uuid=5421bf6f-bccb-4be7-b8b9-5a714a6f18d8&at=APZUnTULWTkj5UhtfzWBoBzLG9ZJ%3A1716447904651"
    download_and_verify "HAM10000_images_part_2.zip" "da43d6cc50f6613013be07e8986b384b" "https://drive.usercontent.google.com/download?id=19D2BTiFHo-XFn86eH07vb6zdmAHHmEKW&export=download&authuser=0&confirm=t&uuid=843f2eca-8cf7-4f4d-a601-79e57ac788e8&at=APZUnTXOTvhRoZjQFgYhcHZF_ueh%3A1716507356071"

    cp $dataset_download_dir/HAM10000_images_part_1.zip $dataset_dir/HAM10000/original/HAM10000_images_part_1.zip
    cp $dataset_download_dir/HAM10000_images_part_2.zip $dataset_dir/HAM10000/original/HAM10000_images_part_2.zip
    cd $dataset_dir/HAM10000/original
    unzip -d HAM10000_images HAM10000_images_part_1.zip
    unzip -d HAM10000_images HAM10000_images_part_2.zip
    rm HAM10000_images_part_1.zip
    rm HAM10000_images_part_2.zip
else
    echo "[√] HAM10000 Images check pass, skipping download."
fi

mkdir -p $dataset_dir/HAM10000/input
cd $dataset_dir/HAM10000/input
if [ ! -f "./HAM10000_metadata.csv" ]; then
    cd $dataset_download_dir
    download_and_verify "HAM10000_metadata.csv" "8f85fb1aa29d80a2797247e434deb79d" "https://drive.usercontent.google.com/download?id=1wxcGLRdbx15NWwVJzlQNuxa2W12Jg8wR&export=download"
    cp $dataset_download_dir/HAM10000_metadata.csv $dataset_dir/HAM10000/input/HAM10000_metadata.csv
else
    echo "[√] HAM10000 meta check pass, skipping download."
fi

## HAM10000 GroundTruth
cd $dataset_dir/HAM10000/original
file_count=-1
if [ -d "HAM10000_GroundTruth" ]; then
    file_count=$(ls -1 HAM10000_GroundTruth | wc -l)
fi
if [ $file_count -ne 10015 ]; then
    rm -rf HAM10000_GroundTruth
    cd $dataset_download_dir
    download_and_verify "HAM10000_segmentations_lesion_tschandl.zip" "6e8d252e09cfdb0189199f15985a5b84" "https://drive.usercontent.google.com/download?id=1UoHo8rupwHPs-7TzcI18wnNR-Gvm5p_N&export=download&authuser=1&confirm=t&uuid=3c01fff3-6bcb-4123-a126-b09668c96b02&at=AENtkXZim_SlGFIGGYy9pl27XLMV%3A1730438159529"

    cp $dataset_download_dir/HAM10000_segmentations_lesion_tschandl.zip $dataset_dir/HAM10000/original/HAM10000_segmentations_lesion_tschandl.zip
    cd $dataset_dir/HAM10000/original
    unzip -j -d HAM10000_GroundTruth HAM10000_segmentations_lesion_tschandl.zip
    rm HAM10000_segmentations_lesion_tschandl.zip
else
    echo "[√] HAM10000 GroundTruth check pass, skipping download."
fi

#


#
# ISIC 2016
cd $dataset_dir/ISIC2016/original
## ISIC2016 Training images
file_count=-1
if [ -d "ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_Data" ]; then
    file_count=$(ls -1 ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_Data | wc -l)
fi
if [ $file_count -ne 900 ]; then
    rm -rf ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_Data
    cd $dataset_download_dir
    download_and_verify "ISBI2016_ISIC_Part1_Training_Data.zip" "2029f387e62dcc062b1370b1efc1f7fb" "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip"
    cp $dataset_download_dir/ISBI2016_ISIC_Part1_Training_Data.zip $dataset_dir/ISIC2016/original/ISBI2016_ISIC_Part1_Training_Data.zip
    cd $dataset_dir/ISIC2016/original
    unzip -d ISIC_2016_Training ISBI2016_ISIC_Part1_Training_Data.zip
    rm ISBI2016_ISIC_Part1_Training_Data.zip
else
    echo "[√] ISIC2016 Training Images check pass, skipping download."
fi
## ISIC2016 Training GroundTruth
cd $dataset_dir/ISIC2016/original
file_count=-1
if [ -d "ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_GroundTruth" ]; then
    file_count=$(ls -1 ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_GroundTruth | wc -l)
fi
if [ $file_count -ne 900 ]; then
    rm -rf ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_GroundTruth
    cd $dataset_download_dir
    download_and_verify "ISBI2016_ISIC_Part1_Training_GroundTruth.zip" "fbd77134298f3511479a37bac93109c7" "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip"
    cp $dataset_download_dir/ISBI2016_ISIC_Part1_Training_GroundTruth.zip $dataset_dir/ISIC2016/original/ISBI2016_ISIC_Part1_Training_GroundTruth.zip
    cd $dataset_dir/ISIC2016/original
    unzip -d ISIC_2016_Training ISBI2016_ISIC_Part1_Training_GroundTruth.zip
    rm ISBI2016_ISIC_Part1_Training_GroundTruth.zip
else
    echo "[√] ISIC2016 Training GroundTruth check pass, skipping download."
fi
## ISIC2016 Test images
file_count=-1
cd $dataset_dir/ISIC2016/original
if [ -d "ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_Data" ]; then
    file_count=$(ls -1 ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_Data | wc -l)
fi
if [ $file_count -ne 379 ]; then
    rm -rf ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_Data
    cd $dataset_download_dir
    download_and_verify "ISBI2016_ISIC_Part1_Test_Data.zip" "efebcaeaae751007401a40a60d391f93" "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip"
    cp $dataset_download_dir/ISBI2016_ISIC_Part1_Test_Data.zip $dataset_dir/ISIC2016/original/ISBI2016_ISIC_Part1_Test_Data.zip
    cd $dataset_dir/ISIC2016/original
    unzip -d ISIC_2016_Test ISBI2016_ISIC_Part1_Test_Data.zip
    rm ISBI2016_ISIC_Part1_Test_Data.zip
else
    echo "[√] ISIC2016 Test Images check pass, skipping download."
fi
## ISIC2016 Test GroundTruth
file_count=-1
cd $dataset_dir/ISIC2016/original
if [ -d "ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_GroundTruth" ]; then
    file_count=$(ls -1 ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_GroundTruth | wc -l)
fi
if [ $file_count -ne 379 ]; then
    rm -rf ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_GroundTruth
    cd $dataset_download_dir
    download_and_verify "ISBI2016_ISIC_Part1_Test_GroundTruth.zip" "492a7711a2e19b96114cab6c96bd1ad5" "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip"
    cp $dataset_download_dir/ISBI2016_ISIC_Part1_Test_GroundTruth.zip $dataset_dir/ISIC2016/original/ISBI2016_ISIC_Part1_Test_GroundTruth.zip
    cd $dataset_dir/ISIC2016/original
    unzip -d ISIC_2016_Test ISBI2016_ISIC_Part1_Test_GroundTruth.zip
    rm ISBI2016_ISIC_Part1_Test_GroundTruth.zip
else
    echo "[√] ISIC2016 Test GroundTruth check pass, skipping download."
fi

# ISIC 2017
## ISIC2017 Training images
file_count=-1
cd $dataset_dir/ISIC2017/original
if [ -d "ISIC_2017_Training/ISIC-2017_Training_Data" ]; then
    file_count=$(ls -1 ISIC_2017_Training/ISIC-2017_Training_Data | wc -l)
fi
if [ $file_count -ne 4001 ]; then
    rm -rf ISIC_2017_Training/ISIC-2017_Training_Data
    cd $dataset_download_dir
    download_and_verify "ISIC-2017_Training_Data.zip" "a14a7e622c67a358797ae59abb8a0b0c" "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip"
    cp $dataset_download_dir/ISIC-2017_Training_Data.zip $dataset_dir/ISIC2017/original/ISIC-2017_Training_Data.zip
    cd $dataset_dir/ISIC2017/original
    unzip -d ISIC_2017_Training ISIC-2017_Training_Data.zip
    rm ISIC-2017_Training_Data.zip
else
    echo "[√] ISIC2017 Training Images check pass, skipping download."
fi
## ISIC2017 Training Ground Truth
file_count=-1
cd $dataset_dir/ISIC2017/original
if [ -d "ISIC_2017_Training/ISIC-2017_Training_Part1_GroundTruth" ]; then
    file_count=$(ls -1 ISIC_2017_Training/ISIC-2017_Training_Part1_GroundTruth | wc -l)
fi
if [ $file_count -ne 2000 ]; then
    rm -rf ISIC_2017_Training/ISIC-2017_Training_Part1_GroundTruth
    cd $dataset_download_dir
    download_and_verify "ISIC-2017_Training_Part1_GroundTruth.zip" "77fdbeb6fbec4139937224416b250f4c" "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip"
    cp $dataset_download_dir/ISIC-2017_Training_Part1_GroundTruth.zip $dataset_dir/ISIC2017/original/ISIC-2017_Training_Part1_GroundTruth.zip
    cd $dataset_dir/ISIC2017/original
    wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part3_GroundTruth.csv ISIC-2017_Training_Part3_GroundTruth.csv
    unzip -d ISIC_2017_Training ISIC-2017_Training_Part1_GroundTruth.zip
    rm ISIC-2017_Training_Part1_GroundTruth.zip
else
    echo "[√] ISIC2017 Training GroundTruth check pass, skipping download."
fi
## ISIC2017 Validation images
file_count=-1
cd $dataset_dir/ISIC2017/original
if [ -d "ISIC_2017_Validation/ISIC-2017_Validation_Data" ]; then
    file_count=$(ls -1 ISIC_2017_Validation/ISIC-2017_Validation_Data | wc -l)
fi
if [ $file_count -ne 301 ]; then
    rm -rf ISIC_2017_Validation/ISIC-2017_Validation_Data
    cd $dataset_download_dir
    download_and_verify "ISIC-2017_Validation_Data.zip" "8d6419d942112f709894c0d82f6c9038" "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip"
    cp $dataset_download_dir/ISIC-2017_Validation_Data.zip $dataset_dir/ISIC2017/original/ISIC-2017_Validation_Data.zip
    cd $dataset_dir/ISIC2017/original
    unzip -d ISIC_2017_Validation ISIC-2017_Validation_Data.zip
    rm ISIC-2017_Validation_Data.zip
else
    echo "[√] ISIC2017 Validation Images check pass, skipping download."
fi
## ISIC2017 Validation Ground Truth
file_count=-1
cd $dataset_dir/ISIC2017/original
if [ -d "ISIC_2017_Validation/ISIC-2017_Validation_Part1_GroundTruth" ]; then
    file_count=$(ls -1 ISIC_2017_Validation/ISIC-2017_Validation_Part1_GroundTruth | wc -l)
fi
if [ $file_count -ne 150 ]; then
    rm -rf ISIC_2017_Validation/ISIC-2017_Validation_Part1_GroundTruth
    cd $dataset_download_dir
    download_and_verify "ISIC-2017_Validation_Part1_GroundTruth.zip" "64d3e68fa2deeb8a5e89aa8dec2efd44" "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip"
    cp $dataset_download_dir/ISIC-2017_Validation_Part1_GroundTruth.zip $dataset_dir/ISIC2017/original/ISIC-2017_Validation_Part1_GroundTruth.zip
    cd $dataset_dir/ISIC2017/original
    unzip -d ISIC_2017_Validation ISIC-2017_Validation_Part1_GroundTruth.zip
    rm ISIC-2017_Validation_Part1_GroundTruth.zip
else
    echo "[√] ISIC2017 Validation GroundTruth check pass, skipping download."
fi
## ISIC2017 Test images
file_count=-1
cd $dataset_dir/ISIC2017/original
if [ -d "ISIC_2017_Test/ISIC-2017_Test_v2_Data" ]; then
    file_count=$(ls -1 ISIC_2017_Test/ISIC-2017_Test_v2_Data | wc -l)
fi
if [ $file_count -ne 1201 ]; then
    rm -rf ISIC_2017_Test/ISIC-2017_Test_v2_Data
    cd $dataset_download_dir
    download_and_verify "ISIC-2017_Test_v2_Data.zip" "5f6a0b5e1f2972bd1f5ea02680489f09" "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip"
    cp $dataset_download_dir/ISIC-2017_Test_v2_Data.zip $dataset_dir/ISIC2017/original/ISIC-2017_Test_v2_Data.zip
    cd $dataset_dir/ISIC2017/original
    unzip -d ISIC_2017_Test ISIC-2017_Test_v2_Data.zip
    rm ISIC-2017_Test_v2_Data.zip
else
    echo "[√] ISIC2017 Test Images check pass, skipping download."
fi
## ISIC2017 Test Ground Truth
file_count=-1
cd $dataset_dir/ISIC2017/original
if [ -d "ISIC_2017_Test/ISIC-2017_Test_v2_Part1_GroundTruth" ]; then
    file_count=$(ls -1 ISIC_2017_Test/ISIC-2017_Test_v2_Part1_GroundTruth | wc -l)
fi
if [ $file_count -ne 600 ]; then
    rm -rf ISIC_2017_Test/ISIC-2017_Test_v2_Part1_GroundTruth
    cd $dataset_download_dir
    download_and_verify "ISIC-2017_Test_v2_Part1_GroundTruth.zip" "b1742de6bd257faca3b2b21a4aa3b781" "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip"
    cp $dataset_download_dir/ISIC-2017_Test_v2_Part1_GroundTruth.zip $dataset_dir/ISIC2017/original/ISIC-2017_Test_v2_Part1_GroundTruth.zip
    cd $dataset_dir/ISIC2017/original
    unzip -d ISIC_2017_Test ISIC-2017_Test_v2_Part1_GroundTruth.zip
    rm ISIC-2017_Test_v2_Part1_GroundTruth.zip
else
    echo "[√] ISIC2017 Test GroundTruth exists, skipping download."
fi

echo "✅ Datasets are already download."

############################
# Split and resize         #
############################
if [ "$split_and_resize" = true ]; then
    echo "Splitting and resizing HAM10000..."
    echo "* trianning rate: $ham_train_rate"
    echo "* validation rate: $ham_val_rate"
    echo "* resizing image to $img_resize x $img_resize"
    cd $dataset_dir

    rm -rf $dataset_dir/HAM10000/input/test
    rm -rf $dataset_dir/HAM10000/input/train
    rm -rf $dataset_dir/HAM10000/input/val
    python $root_path/py_scripts/preprocessing/HAM_split_and_resize.py --input-path $dataset_dir/HAM10000/original/  --output-path $dataset_dir/HAM10000/input/ --image-resize $img_resize $img_resize --train-rate $ham_train_rate --val-rate $ham_val_rate
    echo "[√] HAM10000 split and resize done."

    echo "Resizing ISIC2016..."
    rm -rf $dataset_dir/data/ISIC2016/input/test
    rm -rf $dataset_dir/data/ISIC2016/input/train
    python $root_path/py_scripts/preprocessing/resize_images.py --input-path $dataset_dir/ISIC2016/original/ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_Data --output-path $dataset_dir/ISIC2016/input/train/ISIC2016_img --image-resize $img_resize $img_resize
    python $root_path/py_scripts/preprocessing/resize_images.py --input-path $dataset_dir/ISIC2016/original/ISIC_2016_Training/ISBI2016_ISIC_Part1_Training_GroundTruth --output-path $dataset_dir/ISIC2016/input/train/ISIC2016_seg --image-resize $img_resize $img_resize --is-seg True
    echo "[√] ISIC2016 training resize done."
    python $root_path/py_scripts/preprocessing/resize_images.py --input-path $dataset_dir/ISIC2016/original/ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_Data --output-path $dataset_dir/ISIC2016/input/test/ISIC2016_img --image-resize $img_resize $img_resize
    python $root_path/py_scripts/preprocessing/resize_images.py --input-path $dataset_dir/ISIC2016/original/ISIC_2016_Test/ISBI2016_ISIC_Part1_Test_GroundTruth --output-path $dataset_dir/ISIC2016/input/test/ISIC2016_seg --image-resize $img_resize $img_resize --is-seg True
    echo "[√] ISIC2016 test resize done."

    echo "Resizing ISIC2017..."
    rm -rf $dataset_dir/data/ISIC2017/input/train
    rm -rf $dataset_dir/data/ISIC2017/input/test
    python $root_path/py_scripts/preprocessing/resize_images.py --input-path $dataset_dir/ISIC2017/original/ISIC_2017_Training/ISIC-2017_Training_Data --output-path $dataset_dir/ISIC2017/input/train/ISIC2017_img --image-resize $img_resize $img_resize
    python $root_path/py_scripts/preprocessing/resize_images.py --input-path $dataset_dir/ISIC2017/original/ISIC_2017_Training/ISIC-2017_Training_Part1_GroundTruth --output-path $dataset_dir/ISIC2017/input/train/ISIC2017_seg --image-resize $img_resize $img_resize --is-seg True
    echo "[√] ISIC2017 training resize done."
    python $root_path/py_scripts/preprocessing/resize_images.py --input-path $dataset_dir/ISIC2017/original/ISIC_2017_Validation/ISIC-2017_Validation_Data --output-path $dataset_dir/ISIC2017/input/val/ISIC2017_img --image-resize $img_resize $img_resize
    python $root_path/py_scripts/preprocessing/resize_images.py --input-path $dataset_dir/ISIC2017/original/ISIC_2017_Validation/ISIC-2017_Validation_Part1_GroundTruth --output-path $dataset_dir/ISIC2017/input/val/ISIC2017_seg --image-resize $img_resize $img_resize --is-seg True
    echo "[√] ISIC2017 val resize done."
    python $root_path/py_scripts/preprocessing/resize_images.py --input-path $dataset_dir/ISIC2017/original/ISIC_2017_Test/ISIC-2017_Test_v2_Data --output-path $dataset_dir/ISIC2017/input/test/ISIC2017_img --image-resize $img_resize $img_resize
    python $root_path/py_scripts/preprocessing/resize_images.py --input-path $dataset_dir/ISIC2017/original/ISIC_2017_Test/ISIC-2017_Test_v2_Part1_GroundTruth --output-path $dataset_dir/ISIC2017/input/test/ISIC2017_seg --image-resize $img_resize $img_resize --is-seg True
    echo "[√] ISIC2017 test resize done."

    echo "✅ Datasets split and resize finish."
fi
############################
# Move to class folder    ##
############################
if [ "$move_class" = true ]; then
    echo "Start to move images to the class folder..."

    train_val_img_dir="$dataset_dir/HAM10000/input/train_val/HAM10000_img_class/"
    mkdir -p "$train_val_img_dir"
    train_val_seg_dir="$dataset_dir/HAM10000/input/train_val/HAM10000_seg_class/"
    mkdir -p "$train_val_seg_dir"

    # HAM10000 train img
    output_dir="$dataset_dir/HAM10000/input/train/HAM10000_img_class/"
    python $root_path/py_scripts/preprocessing/class_copy.py --data-name HAM10000_train_img --csv-file $dataset_dir/HAM10000/input/HAM10000_metadata.csv --data-path $dataset_dir/HAM10000/input/train/HAM10000_img/ --out-path $output_dir
    file_count=$(find "$output_dir" -type f | wc -l)
    if [ "$file_count" -ne 8012 ]; then
      echo "Error：Folder: $output_dir files num: $file_count, != 8012"
      exit 1
    fi
    rsync -a "$output_dir" "$train_val_img_dir"

    # HAM10000 train seg
    output_dir="$dataset_dir/HAM10000/input/train/HAM10000_seg_class/"
    python $root_path/py_scripts/preprocessing/class_copy.py --data-name HAM10000_train_seg --csv-file $dataset_dir/HAM10000/input/HAM10000_metadata.csv --data-path $dataset_dir/HAM10000/input/train/HAM10000_seg/ --out-path $output_dir
    file_count=$(find "$output_dir" -type f | wc -l)
    if [ "$file_count" -ne 8012 ]; then
      echo "Error：Folder: $output_dir files num: $file_count, != 8012"
      exit 1
    fi
    rsync -a "$output_dir" "$train_val_seg_dir"

    # HAM10000 val img
    output_dir="$dataset_dir/HAM10000/input/val/HAM10000_img_class/"
    python $root_path/py_scripts/preprocessing/class_copy.py --data-name HAM10000_val_img --csv-file $dataset_dir/HAM10000/input/HAM10000_metadata.csv --data-path $dataset_dir/HAM10000/input/val/HAM10000_img/ --out-path $output_dir
    file_count=$(find "$output_dir" -type f | wc -l)
    if [ "$file_count" -ne 1001 ]; then
      echo "Error：Folder: $output_dir files num: $file_count, != 1001"
      exit 1
    fi
    rsync -a "$output_dir" "$train_val_img_dir"
    # Check $train_val_img_dir
    file_count=$(find "$train_val_img_dir" -type f | wc -l)
    if [ "$file_count" -ne 9013 ]; then
      echo "Error：Folder: $train_val_img_dir files num: $file_count, != 9013"
      exit 1
    fi


    # HAM10000 val seg
    output_dir="$dataset_dir/HAM10000/input/val/HAM10000_seg_class/"
    python $root_path/py_scripts/preprocessing/class_copy.py --data-name HAM10000_val_seg --csv-file $dataset_dir/HAM10000/input/HAM10000_metadata.csv --data-path $dataset_dir/HAM10000/input/val/HAM10000_seg/ --out-path $output_dir
    file_count=$(find "$output_dir" -type f | wc -l)
    if [ "$file_count" -ne 1001 ]; then
      echo "Error：Folder: $output_dir files num: $file_count, != 1001"
      exit 1
    fi
    rsync -a "$output_dir" "$train_val_seg_dir"
    # Check $train_val_seg_dir
    file_count=$(find "$train_val_seg_dir" -type f | wc -l)
    if [ "$file_count" -ne 9013 ]; then
      echo "Error：Folder: $train_val_seg_dir files num: $file_count, != 9013"
      exit 1
    fi

    # HAM10000 test img
    output_dir="$dataset_dir/HAM10000/input/test/HAM10000_img_class/"
    python $root_path/py_scripts/preprocessing/class_copy.py --data-name HAM10000_test_img --csv-file $dataset_dir/HAM10000/input/HAM10000_metadata.csv --data-path $dataset_dir/HAM10000/input/test/HAM10000_img/ --out-path $output_dir
    file_count=$(find "$output_dir" -type f | wc -l)
    if [ "$file_count" -ne 1002 ]; then
      echo "Error：Folder: $output_dir files num: $file_count, != 1002"
      exit 1
    fi

    # HAM10000 test seg
    output_dir="$dataset_dir/HAM10000/input/test/HAM10000_seg_class/"
    python $root_path/py_scripts/preprocessing/class_copy.py --data-name HAM10000_test_seg --csv-file $dataset_dir/HAM10000/input/HAM10000_metadata.csv --data-path $dataset_dir/HAM10000/input/test/HAM10000_seg/ --out-path $output_dir
    file_count=$(find "$output_dir" -type f | wc -l)
    if [ "$file_count" -ne 1002 ]; then
      echo "Error：Folder: $output_dir files num: $file_count, != 1002"
      exit 1
    fi

    # Modify directory hierarchy structure
    folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
    target_dirs=("train_val" "train" "test" "val")
    for target_dir in "${target_dirs[@]}"; do
        folder_temp=$dataset_dir/HAM10000/input/$target_dir/
        rm -r $folder_temp/HuggingFace 2>/dev/null
        for folder in "${folders[@]}"; do
            # Create new hierarchy structure
            mkdir -p $folder_temp/HuggingFace/$folder/HAM10000_img_class
            mkdir -p $folder_temp/HuggingFace/$folder/HAM10000_seg_class
            # Move corresponding content to new structure
            cp $folder_temp/HAM10000_img_class/$folder/* $folder_temp/HuggingFace/$folder/HAM10000_img_class/ 2>/dev/null
            cp $folder_temp/HAM10000_seg_class/$folder/* $folder_temp/HuggingFace/$folder/HAM10000_seg_class/ 2>/dev/null
        done
    done

    python $root_path/py_scripts/preprocessing/HAM_meta_generate.py --root-path $dataset_dir/HAM10000/input

    echo "✅ Dataset class move finish."

fi

############################
# Copy files for VAR    ##
############################

# Define function for file copying
copy_files_for_VAR() {
    local source_base=$1
    local target_base=$2

    # Iterate through subdirectories under HuggingFace directory
    for dir in "$source_base"/*/HAM10000_img_class; do
        # Get corresponding subdirectory name
        local sub_dir=$(basename "$(dirname "$dir")")
        # Build target path
        local target_dir="$target_base/$sub_dir"
        # Create target directory if it doesn't exist
        if [ ! -d "$target_dir" ]; then
            mkdir -p "$target_dir"
        fi
        # Copy files from source directory to target directory
        cp -r "$dir/"* "$target_dir/"
        echo "Copied files from $dir to $target_dir"
    done
}

if [ "$prepare_VAR" = true ]; then
  # Define source and target directories
  source_base=$dataset_dir/HAM10000/input/test/"HuggingFace"
  target_base=$dataset_dir/HAM10000/input/test/"VAR"
  copy_files_for_VAR "$source_base" "$target_base"

  source_base=$dataset_dir/HAM10000/input/val/"HuggingFace"
  target_base=$dataset_dir/HAM10000/input/val/"VAR"
  copy_files_for_VAR "$source_base" "$target_base"

  source_base=$dataset_dir/HAM10000/input/train/"HuggingFace"
  target_base=$dataset_dir/HAM10000/input/train/"VAR"
  copy_files_for_VAR "$source_base" "$target_base"

fi
############################
# Check resize dataset    ##
############################
if [ "$check_dataset" = true ]; then
    echo "Checking image size..."
    cd $root_path
    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name HAM10000_train_img --data-path $dataset_dir/HAM10000/input/train/HAM10000_img/ --image-size $img_resize $img_resize --image-num 8012
    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name HAM10000_train_seg --data-path $dataset_dir/HAM10000/input/train/HAM10000_seg/ --image-size $img_resize $img_resize --image-num 8012

    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_img --data-path $dataset_dir/HAM10000/input/val/HAM10000_img/ --image-size $img_resize $img_resize --image-num 1001
    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name HAM10000_val_seg --data-path $dataset_dir/HAM10000/input/val/HAM10000_seg/ --image-size $img_resize $img_resize --image-num 1001

    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_img --data-path $dataset_dir/HAM10000/input/test/HAM10000_img/ --image-size $img_resize $img_resize --image-num 1002
    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name HAM10000_test_seg --data-path $dataset_dir/HAM10000/input/test/HAM10000_seg/ --image-size $img_resize $img_resize --image-num 1002

    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_train_img --data-path $dataset_dir/ISIC2016/input/train/ISIC2016_img/ --image-size $img_resize $img_resize --image-num 900
    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_train_seg --data-path $dataset_dir/ISIC2016/input/train/ISIC2016_seg/ --image-size $img_resize $img_resize --image-num 900

    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_img --data-path $dataset_dir/ISIC2016/input/test/ISIC2016_img/ --image-size $img_resize $img_resize --image-num 379
    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name ISIC2016_test_seg --data-path $dataset_dir/ISIC2016/input/test/ISIC2016_seg/ --image-size $img_resize $img_resize --image-num 379

    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_train_img --data-path $dataset_dir/ISIC2017/input/train/ISIC2017_img/ --image-size $img_resize $img_resize --image-num 2000
    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_train_seg --data-path $dataset_dir/ISIC2017/input/train/ISIC2017_seg/ --image-size $img_resize $img_resize --image-num 2000

    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_val_val --data-path $dataset_dir/ISIC2017/input/val/ISIC2017_img/ --image-size $img_resize $img_resize --image-num 150
    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_val_val --data-path $dataset_dir/ISIC2017/input/val/ISIC2017_seg/ --image-size $img_resize $img_resize --image-num 150

    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_img --data-path $dataset_dir/ISIC2017/input/test/ISIC2017_img/ --image-size $img_resize $img_resize --image-num 600
    python $root_path/py_scripts/preprocessing/check_image_size.py --data-name ISIC2017_test_seg --data-path $dataset_dir/ISIC2017/input/test/ISIC2017_seg/ --image-size $img_resize $img_resize --image-num 600

    echo "✅ Dataset check pass."
fi
