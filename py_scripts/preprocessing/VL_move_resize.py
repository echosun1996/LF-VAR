import pandas as pd
import os
import shutil
from PIL import Image
import argparse
from tqdm import tqdm
import numpy as np
def resize_and_move_images(csv_file, source_dir, output_dir, resize_width, resize_height, train_rate=0.8):
    df = pd.read_csv(csv_file)

    total_files = len(df)
    train_target = int(total_files * train_rate)
    train_count = 0
    val_count = 0
    for index, row in tqdm(df.iterrows(), total=total_files):
        image_name = row['Image_Path']
        infection = row['Infection']
        image_type = row['Type']
        source_folder = row['source_folder']

        if train_count < train_target and np.random.rand() < train_rate:
            set_type = "train"
            train_count += 1
        else:
            set_type = "val"
            val_count += 1

        source_path = os.path.join(source_dir, image_name)

        if not os.path.exists(source_path):
            print(f"Source file does not exist: {source_path}")
            exit(1)

        target_dir = os.path.join(output_dir, set_type, image_type, infection)
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, source_folder + "_" + os.path.splitext(image_name)[0] + '.jpg')

        with Image.open(source_path) as img:
            resized_img = img.resize((resize_width, resize_height))
            resized_img = resized_img.convert("RGB")
            resized_img.save(target_path, format='JPEG')
    print("Train Count:",train_count, "Val Count:",val_count, "Total:",total_files)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and move images based on CSV metadata.")
    parser.add_argument("-file_path", required=True, help="Path to the CSV file with metadata.")
    parser.add_argument("-source", required=True, help="Source directory of the images.")
    parser.add_argument("-output", required=True, help="Output directory for resized images.")
    parser.add_argument("-resize", nargs=2, type=int, required=True, help="Resize width and height.")
    parser.add_argument("-train_rate", type=float, default=0.8, help="Proportion of images for training set.")

    args = parser.parse_args()

    resize_and_move_images(args.file_path, args.source, args.output, args.resize[0], args.resize[1], args.train_rate)