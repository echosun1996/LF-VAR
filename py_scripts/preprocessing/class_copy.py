import os
import shutil
import pandas as pd
import argparse
from tqdm import tqdm

def move_files_by_filenames(data_name, csv_file, data_path, out_path):
    try:
        metadata = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if 'image_id' not in metadata.columns or 'dx' not in metadata.columns:
        print("CSV file must contain 'image_id' and 'dx' columns.")
        return

    if "HAM" in data_name:
        id_to_dx = dict(zip(metadata['image_id'], metadata['dx']))
    else:
        print(f"Not valid dataset name:[{data_name}]!")
        exit(-1)
    for filename in tqdm(os.listdir(data_path),desc=data_name):
        file_path = os.path.join(data_path, filename)
        if not os.path.isfile(file_path):
            continue

        file_id, ext = os.path.splitext(filename)

        if file_id.find("_segmentation"):
            file_id = file_id.replace("_segmentation", "")

        if file_id not in id_to_dx:
            print(f"File {filename} not found in metadata.")
            exit(-1)

        dx = id_to_dx[file_id]

        dst_dir = os.path.join(out_path, dx)
        dst_file = os.path.join(dst_dir, filename)

        os.makedirs(dst_dir, exist_ok=True)

        try:
            shutil.copy(file_path, dst_file)
        except Exception as e:
            print(f"Error moving file {file_path} to {dst_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Move files into class directories based on CSV metadata and existing filenames.")
    parser.add_argument('--data-name', type=str, required=True, help="Name of the data set.")
    parser.add_argument('--csv-file', type=str, required=True, help="Path to the CSV metadata file.")
    parser.add_argument('--data-path', type=str, required=True, help="Path to the source data directory.")
    parser.add_argument('--out-path', type=str, required=True, help="Path to the output directory.")

    args = parser.parse_args()

    move_files_by_filenames(args.data_name ,args.csv_file, args.data_path, args.out_path)

if __name__ == "__main__":
    main()