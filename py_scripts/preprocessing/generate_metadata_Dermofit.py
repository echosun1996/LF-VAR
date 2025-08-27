import pandas as pd
import argparse
import os

def parse_lesion_list(file_path):
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split()]
            if len(parts) >= 3:
                data.append({
                    'index': parts[0],
                    'image_id': parts[1],
                    'diagnosis': parts[2]
                })
    return pd.DataFrame(data)

def get_class_name(diagnosis):
    
    mapping = {
        'BCC': 'bcc',
        'ML': 'mel',
        'SK': 'bkl',
        'AK': 'akiec',
        'VASC': 'vasc',
        'DF': 'df',
        'IEC': 'akiec',
        'MEL': 'nv'
    }
    return mapping.get(diagnosis)

def check_file_exists(root_path,img_path, seg_path):
    
    if not os.path.exists(os.path.join(root_path,img_path)):
        print(f"Warning: Image file not found: {os.path.join(root_path,img_path)}")
        return False
    if not os.path.exists(os.path.join(root_path,seg_path)):
        print(f"Warning: Segmentation file not found: {os.path.join(root_path,seg_path)}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate metadata file from Dermofit lesion list')
    parser.add_argument('--root_path', type=str, required=True, help='Root path of the project')
    parser.add_argument('--in_path', type=str, required=True, help='Input lesion list file path')
    parser.add_argument('--out_path', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--keys', type=str, nargs='+', help='List of class keys')
    parser.add_argument('--values', type=str, nargs='+', help='List of corresponding prompts')
    args = parser.parse_args()
    root_path = args.root_path
    if not os.path.exists(args.in_path):
        raise FileNotFoundError(f"Input file not found: {args.in_path}")

    out_dir = os.path.dirname(args.out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")

    if len(args.keys) != len(args.values):
        raise ValueError("Number of keys must match number of values")
    prompts = dict(zip(args.keys, args.values))

    df = parse_lesion_list(args.in_path)

    new_data = []
    skipped_files = 0
    total_files = 0

    for _, row in df.iterrows():
        total_files += 1
        class_name = get_class_name(row['diagnosis'])
        
        if class_name is not None and class_name in prompts:
            img_path = f"data/local/Dermofit/input/test/Dermofit_img/{row['image_id']}.jpg"
            seg_path = f"data/local/Dermofit/input/test/Dermofit_seg/{row['image_id']}_segmentation.png"
            
            if check_file_exists(root_path,img_path, seg_path):
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
            print(f"Skipping diagnosis {row['diagnosis']} (not in supported classes)")
            skipped_files += 1

    new_df = pd.DataFrame(new_data)
    new_df.to_csv(args.out_path, index=False)
    
    print(f"Total files processed: {total_files}")
    print(f"Files skipped: {skipped_files}")
    print(f"Files successfully processed: {len(new_data)}")
    print(f"Generated metadata file saved to {args.out_path}")

if __name__ == '__main__':
    main() 