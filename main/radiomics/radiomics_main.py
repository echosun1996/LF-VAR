import os.path
from sys import meta_path

import radiomics_funcs
import argparse
from torch_fidelity import calculate_metrics


parser = argparse.ArgumentParser(description="Radiomics Main")
parser.add_argument("--root-path", type=str, required=True,
                    help="Path for the folder path with 'HAM10000_images' folder in it.")
parser.add_argument("--meta-path", type=str, required=True)
parser.add_argument("--img-folder-name", type=str, required=False)
parser.add_argument("--seg-folder-name", type=str, required=False)
args = parser.parse_args()
root_path = args.root_path
meta_path = args.meta_path


img_folder_name = args.img_folder_name
seg_folder_name = args.seg_folder_name

out_folder_path = os.path.join(root_path, "radiomics", "1.Original")

if not os.path.exists(os.path.join(out_folder_path,"radiomics.csv")):
    print("Start to extract all radiomics features")
    radiomics_funcs.extra_main(root_path=root_path,img_folder_name=img_folder_name,seg_folder_name=seg_folder_name)

radiomics_funcs.process_csv_folder(os.path.join(root_path, "radiomics"),metapath=meta_path)