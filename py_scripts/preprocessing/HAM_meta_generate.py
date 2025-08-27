import argparse
import os
import csv


parser = argparse.ArgumentParser(description="Move files into class directories based on CSV metadata and existing filenames.")
parser.add_argument('--root-path', type=str, required=True, help="Path to the HAM10000 root directory.")
args = parser.parse_args()

root_path = args.root_path


class_prompts = {
    "akiec": "An image of a skin area with actinic keratoses or intraepithelial carcinoma.",
    "bcc": "An image of a skin area with basal cell carcinoma.",
    "bkl": "An image of a skin area with benign keratosis-like lesions.",
    "df": "An image of a skin area with dermatofibroma.",
    "mel": "An image of a skin area with melanoma.",
    "nv": "An image of a skin area with melanocytic nevi.",
    "vasc": "An image of a skin area with a vascular lesion."
}

root_dirs = {
    str(os.path.join(args.root_path,"test/HuggingFace")): "test",
    str(os.path.join(args.root_path,"train_val/HuggingFace")): "train_val"
}

output_file = str(os.path.join(args.root_path,"metadata.csv"))

metadata = []

for root_dir, dataset_split in root_dirs.items():
    for class_name, prompt in class_prompts.items():
        img_dir = os.path.join(root_dir, class_name, "HAM10000_img_class")
        seg_dir = os.path.join(root_dir, class_name, "HAM10000_seg_class")

        if os.path.exists(img_dir) and os.path.exists(seg_dir):
            img_files = set(f.replace(".jpg", "")  for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

            seg_files = {f.replace("_segmentation", "").replace(".png", "") for f in os.listdir(seg_dir) if
                f.lower().endswith(('.png', '.jpg', '.jpeg'))}
            matched_files = img_files & seg_files

            for file_name in matched_files:
                img_path = os.path.join(img_dir, file_name+".jpg")
                seg_path = os.path.join(seg_dir, file_name+"_segmentation.png")
                metadata.append([img_path, seg_path, class_name, prompt, dataset_split])


with open(output_file, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["img_path", "seg_path", "class", "prompt", "dataset_split"])
    writer.writerows(metadata)

print(f"Metadata file '{output_file}' created successfully with {len(metadata)} entries.")