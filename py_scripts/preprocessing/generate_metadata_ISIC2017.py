import pandas as pd
import argparse
import os
import json

def get_class_from_labels(melanoma, seborrheic_keratosis):
    
    if melanoma == 1 and seborrheic_keratosis == 0:
        return 'mel'
    elif melanoma == 0 and seborrheic_keratosis == 1:
        return 'bkl'
    elif melanoma == 0 and seborrheic_keratosis == 0:
        return 'nv'
    else:
        return None

def check_file_exists(root_path,img_path, seg_path):
    
    if not os.path.exists(os.path.join(root_path,img_path)):
        print(f"Warning: Image file not found: {os.path.join(root_path,img_path)}")
        return False
    if not os.path.exists(os.path.join(root_path,seg_path)):
        print(f"Warning: Segmentation file not found: {os.path.join(root_path,seg_path)}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate metadata file from ISIC labels')
    parser.add_argument('--root_path', type=str, required=True, help='Root file path')
    parser.add_argument('--in_csv_path', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--out_csv_path', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--keys', type=str, nargs='+', help='List of class keys')
    parser.add_argument('--values', type=str, nargs='+', help='List of corresponding prompts')
    args = parser.parse_args()
    root_path = args.root_path

    if not os.path.exists(args.in_csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {args.in_csv_path}")

    out_dir = os.path.dirname(args.out_csv_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")

    if len(args.keys) != len(args.values):
        raise ValueError("Number of keys must match number of values")
    
    prompts = dict(zip(args.keys, args.values))

    df = pd.read_csv(args.in_csv_path)

    new_data = []
    skipped_files = 0
    total_files = 0
    
    for _, row in df.iterrows():
        image_id = row['image_id']
        class_name = get_class_from_labels(row['melanoma'], row['seborrheic_keratosis'])
        total_files += 1
        
        if class_name is not None and class_name in prompts:
            img_path = f"data/local/ISIC2017/input/test/ISIC2017_img/{image_id}.jpg"
            seg_path = f"data/local/ISIC2017/input/test/ISIC2017_seg/{image_id}_segmentation.png"
            
            if check_file_exists(root_path,img_path, seg_path):
                new_data.append({
                    'img_path': img_path,
                    'seg_path': seg_path,
                    'class': class_name,
                    'prompt': prompts[class_name],
                    'dataset_split': 'test'
                })
            else:
                print(f"Warning: Missing files for row {row}")
                skipped_files += 1


    if skipped_files > 0:
        print(f"Warning: {skipped_files} files skipped due to missing files")
        print(f"Total files processed: {total_files}")
        print(f"Files skipped due to missing files: {skipped_files}")
        print(f"Files successfully processed: {len(new_data)}")
        print(f"Generated metadata file saved to {args.out_csv_path}")
        exit(-1)
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(args.out_csv_path, index=False)
    print("Metadata file generated successfully!")
if __name__ == '__main__':
    main()