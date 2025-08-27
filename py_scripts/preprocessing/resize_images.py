import os
from tqdm import tqdm
from PIL import Image
from rich import print
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-img-path", type=str, default=None, help="input image path")
parser.add_argument("--input-path", type=str, default=None, help="input path")
parser.add_argument("--output-path", type=str, default=None, help="output path")
parser.add_argument("--image-resize", nargs=2, default=[1024, 1024], type=int)
parser.add_argument("--is-seg", type=bool, default=False, help="is segmentation folder")


input_folder = parser.parse_args().input_path
output_folder = parser.parse_args().output_path
image_resize = parser.parse_args().image_resize

print("* input_folder", input_folder)
print(
    "* files_num: ",
    len(os.listdir(input_folder)),
    "; is_seg: ",
    parser.parse_args().is_seg,
    sep="",
)

os.makedirs(output_folder, exist_ok=True)


def copy_files(source_folder, target_folder, file_list, resize, seg=False):
    for file_name in tqdm(file_list):
        if str(file_name).find("_superpixels") != -1 or str(file_name).endswith(".csv"):
            continue
        source_path = os.path.join(source_folder, file_name)
        if seg:
            file_name = str(file_name).replace("_Segmentation", "_segmentation")
        target_path = os.path.join(target_folder, file_name)
        if not seg:
            image = Image.open(source_path)
            resized_image = image.resize(resize)
        else:
            mask = Image.open(source_path).convert("L")
            resized_image = mask.resize(resize, Image.Resampling.NEAREST)
        resized_image.save(target_path)


copy_files(
    input_folder,
    output_folder,
    file_list=os.listdir(input_folder),
    resize=image_resize,
    seg=parser.parse_args().is_seg,
)