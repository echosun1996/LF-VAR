import sys
from tqdm import tqdm
import os
from PIL import Image
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default=None, help='dataset path')
parser.add_argument('--output-dir', default=None, type=str)

args = parser.parse_args()

if args.input_dir is None:
    print("Error: input_dir is required.")
    sys.exit(1)
    
if args.output_dir is None:
    print("Error: output_dir is required.")
    sys.exit(1)


input_path = os.path.join(args.input_dir,"PH2Dataset/PH2 Dataset images")

output_path = args.output_dir
out_ground_truth_path = os.path.join(output_path,"PH2_GroundTruth")
out_data_path = os.path.join(output_path,"PH2_Data")
if not os.path.exists(out_ground_truth_path):
    os.makedirs(out_ground_truth_path)
if not os.path.exists(out_data_path):
    os.makedirs(out_data_path)

folder_names = os.listdir(input_path)
for folder_name in tqdm(folder_names):
    if folder_name.startswith("IMD"):
        data_folder_path = os.path.join(input_path, folder_name)
        image_path = os.path.join(data_folder_path , folder_name+"_Dermoscopic_Image" ,folder_name +".bmp")
        mask_path = os.path.join(data_folder_path , folder_name+"_lesion" ,folder_name +"_lesion.bmp")
        
        
        output_file_name = folder_name.replace("IMD", "IMD_")
        out_image_name = output_file_name + ".jpg"
        out_image_path = os.path.join(out_data_path, out_image_name)
        out_mask_name = output_file_name + "_segmentation" + ".png"    
        out_mask_path = os.path.join(out_ground_truth_path, out_mask_name)

        
        im = Image.open(image_path)
        im.save(out_image_path)
        
        im = Image.open(mask_path)
        im.save(out_mask_path)