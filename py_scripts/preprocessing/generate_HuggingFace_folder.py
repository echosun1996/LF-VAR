import os
import argparse
import pandas as pd
import shutil


def copy_files(meta_path, root_path, output_path, dataset:str):
    df = pd.read_csv(meta_path)



    for _, row in df.iterrows():
        class_name = row['class']
        img_src = os.path.join(root_path, row['img_path'])
        seg_src = os.path.join(root_path, row['seg_path'])

        img_output_base = os.path.join(output_path, class_name, dataset + "_img_class")
        seg_output_base = os.path.join(output_path, class_name, dataset + "_seg_class")
        os.makedirs(img_output_base, exist_ok=True)
        os.makedirs(seg_output_base, exist_ok=True)



        img_dst = os.path.join(img_output_base, os.path.basename(img_src))
        seg_dst = os.path.join(seg_output_base, os.path.basename(seg_src))

        if os.path.exists(img_src):
            shutil.copy2(img_src, img_dst)
        else:
            print(f"Warning: Image file not found: {img_src}")
            exit(-1)

        if os.path.exists(seg_src):
            shutil.copy2(seg_src, seg_dst)
        else:
            print(f"Warning: Segmentation file not found: {seg_src}")
            exit(-1)

    print(f"File organization complete for dataset: {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HuggingFace folder structure.")
    parser.add_argument("--meta-path", type=str, required=True, help="Path to the metadata CSV file.")
    parser.add_argument("--root-path", type=str, required=True, help="Root path for dataset files.")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory for organized files.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")

    args = parser.parse_args()
    copy_files(args.meta_path, args.root_path, args.output_path, args.dataset)