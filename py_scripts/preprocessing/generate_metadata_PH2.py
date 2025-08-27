import pandas as pd
import argparse
import os
import re

def parse_ph2_dataset(file_path):
    
    data = []
    with open(file_path, 'r') as f:
        next(f)
        
        for line in f:
            if not line.strip() or '||' not in line:
                continue
                
            parts = [part.strip() for part in line.split('||')]
            if len(parts) >= 4:
                image_id = parts[1].strip()
                image_id = re.sub(r'(IMD)(\d+)', r'\1_\2', image_id)
                
                diagnosis_match = re.search(r'\d+', parts[3]) if len(parts) > 4 else None
                diagnosis = int(diagnosis_match.group()) if diagnosis_match else None
                
                if image_id and diagnosis is not None:
                    data.append({
                        'image_id': image_id,
                        'diagnosis': diagnosis
                    })
    return pd.DataFrame(data)

def get_class_name(diagnosis):
    
    mapping = {
        0: 'nv',
        2: 'mel'
    }
    return mapping.get(diagnosis)

def check_file_exists(root_path, img_path, seg_path):
    
    if not os.path.exists(os.path.join(root_path, img_path)):
        print(f"Warning: Image file not found: {os.path.join(root_path, img_path)}")
        return False
    if not os.path.exists(os.path.join(root_path, seg_path)):
        print(f"Warning: Segmentation file not found: {os.path.join(root_path, seg_path)}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate metadata file from PH2 dataset')
    parser.add_argument('--root_path', type=str, required=True, help='Root path of the project')
    parser.add_argument('--in_path', type=str, required=True, help='Input PH2 dataset file path')
    parser.add_argument('--out_path', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--keys', type=str, nargs='+', help='List of class keys')
    parser.add_argument('--values', type=str, nargs='+', help='List of corresponding prompts')
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        raise FileNotFoundError(f"Input file not found: {args.in_path}")

    out_dir = os.path.dirname(args.out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")

    if len(args.keys) != len(args.values):
        raise ValueError("Number of keys must match number of values")
    prompts = dict(zip(args.keys, args.values))

    df = parse_ph2_dataset(args.in_path)

    new_data = []
    skipped_files = 0
    total_files = 0

    for _, row in df.iterrows():
        total_files += 1
        class_name = get_class_name(row['diagnosis'])
        
        if class_name is not None and class_name in prompts:
            img_path = f"data/local/PH2/input/test/PH2_img/{row['image_id']}.jpg"
            seg_path = f"data/local/PH2/input/test/PH2_seg/{row['image_id']}_segmentation.png"
            
            if check_file_exists(args.root_path, img_path, seg_path):
                new_data.append({
                    'img_path': img_path,
                    'seg_path': seg_path,
                    'class': class_name,
                    'prompt': prompts[class_name],
                    'dataset_split': 'test'
                })
            else:
                skipped_files += 1
        else:
            print(f"Skipping image {row['image_id']} with diagnosis {row['diagnosis']} (not in supported classes)")
            skipped_files += 1

    new_df = pd.DataFrame(new_data)
    new_df.to_csv(args.out_path, index=False)
    
    print(f"Total files processed: {total_files}")
    print(f"Files skipped: {skipped_files}")
    print(f"Files successfully processed: {len(new_data)}")
    print(f"Generated metadata file saved to {args.out_path}")

if __name__ == '__main__':
    main() 