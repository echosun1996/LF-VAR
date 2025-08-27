from math import remainder
from tqdm import tqdm
import os
import numpy as np
from rich import print
from rich.progress import Progress
import sys
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data-name", type=str, default=None, help="dataset name")
parser.add_argument("--data-path", type=str, default=None, help="dataset path")
parser.add_argument("--image-num", default=None, type=int)
parser.add_argument("--image-size", nargs=2, default=[1024, 1024], type=int)

args = parser.parse_args()

dir_path = args.data_path


seg = False
files = os.listdir(dir_path)
for file in files:
    if not file.endswith(".jpg") and not file.endswith("_segmentation.png"):
        files.remove(file)
    if file.endswith("_segmentation.png"):
        seg = True

file_num = len(files)
if file_num != args.image_num:
    for file in files:
        if file.endswith(".jpg") or file.endswith("_segmentation.png"):
            continue
        print(file)
    print(
        f"[bold red]Error: [/bold red] For {args.data_name}, args.image_num({args.image_num}) does not match the real file_num({file_num})."
    )
    sys.exit(1)

min_h = 100000000
min_w = 100000000
max_h = -1
max_w = -1

with Progress() as progress:
    task = progress.add_task(
        f"[green]{args.data_name} - [blue]{len(files)}", total=len(files)
    )
    for file in files:
        progress.advance(task)
        with Image.open(os.path.join(dir_path, file)) as img:
            w, h = img.size
        if h < min_h:
            min_h = h
        if w < min_w:
            min_w = w
        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w

    if min_h != max_h or min_h != args.image_size[0]:
        print(
            f"[bold red]Error: [/bold red] For {args.data_name}, min_h({min_h}) != max_h({max_h}) != args.image_size[0]({args.image_size[0]})"
        )
        sys.exit(1)
    if min_w != max_w or min_w != args.image_size[1]:
        print(
            f"[bold red]Error: [/bold red] For {args.data_name}, min_w({min_w}) != max_w({max_w}) != args.image_size[1]({args.image_size[1]})"
        )
        sys.exit(1)