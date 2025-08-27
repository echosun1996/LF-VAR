import os
import random
from tqdm import tqdm
import argparse
from PIL import Image
from rich import print

parser = argparse.ArgumentParser()
parser.add_argument("--input-path", type=str, default=None, help="input path")
parser.add_argument("--output-path", type=str, default=None, help="output path")
parser.add_argument("--image-resize", nargs=2, default=[1024, 1024], type=int)
parser.add_argument("--train-rate", type=float, default=0.8, help="train rate")
parser.add_argument("--val-rate", type=float, default=0.1, help="val rate")

img_folder = os.path.join(parser.parse_args().input_path, "HAM10000_images")
seg_folder = os.path.join(parser.parse_args().input_path, "HAM10000_GroundTruth")


image_resize = parser.parse_args().image_resize


taget_folder = parser.parse_args().output_path

img_files = sorted(os.listdir(img_folder))

random.seed(0)
random.shuffle(img_files)


train_folder = os.path.join(taget_folder, "train")
train_img_folder = os.path.join(train_folder, "HAM10000_img")
train_seg_folder = os.path.join(train_folder, "HAM10000_seg")
os.makedirs(train_img_folder, exist_ok=True)
os.makedirs(train_seg_folder, exist_ok=True)

val_folder = os.path.join(taget_folder, "val")
val_img_folder = os.path.join(val_folder, "HAM10000_img")
val_seg_folder = os.path.join(val_folder, "HAM10000_seg")
os.makedirs(val_img_folder, exist_ok=True)
os.makedirs(val_seg_folder, exist_ok=True)


test_folder = os.path.join(taget_folder, "test")
test_img_folder = os.path.join(test_folder, "HAM10000_img")
test_seg_folder = os.path.join(test_folder, "HAM10000_seg")
os.makedirs(test_img_folder, exist_ok=True)
os.makedirs(test_seg_folder, exist_ok=True)


total_files = len(img_files)
train_count = int(total_files * parser.parse_args().train_rate)
val_count = int(total_files * parser.parse_args().val_rate)
test_count = total_files - train_count - val_count
print("* train_count:", train_count)
print("* val_count:", val_count)
print("* test_count:", test_count)


def copy_files(source_folder, target_folder, file_list, resize, seg=False):
    for file_name in tqdm(file_list):
        if seg:
            file_name = file_name.split(".")[0] + "_segmentation.png"
        source_path = os.path.join(source_folder, file_name)
        target_path = os.path.join(target_folder, file_name)
        if not seg:
            image = Image.open(source_path)
            resized_image = image.resize(resize)
        else:
            mask = Image.open(source_path).convert("L")
            resized_image = mask.resize(resize, Image.Resampling.NEAREST)
        resized_image.save(target_path)


print("* Start process train files...")
copy_files(img_folder, train_img_folder, img_files[:train_count], resize=image_resize)
copy_files(
    seg_folder, train_seg_folder, img_files[:train_count], resize=image_resize, seg=True
)

print("* Start process val files...")
copy_files(
    img_folder,
    val_img_folder,
    img_files[train_count : train_count + val_count],
    resize=image_resize,
)
copy_files(
    seg_folder,
    val_seg_folder,
    img_files[train_count : train_count + val_count],
    resize=image_resize,
    seg=True,
)

print("* Start process test files...")
copy_files(
    img_folder,
    test_img_folder,
    img_files[train_count + val_count :],
    resize=image_resize,
)
copy_files(
    seg_folder,
    test_seg_folder,
    img_files[train_count + val_count :],
    resize=image_resize,
    seg=True,
)