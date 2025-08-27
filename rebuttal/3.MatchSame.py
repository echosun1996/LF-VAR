import pandas as pd
import os
from collections import Counter
import glob

metadata_file = '/mnt/SkinGenerativeModel/code/data/local/Dermofit/input/metadata.csv'
df = pd.read_csv(metadata_file)

class_counts = Counter(df['class'])
print("Current class distribution:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

target_count = 400

base_dir = '/mnt/SkinGenerativeModel/code/data/compare_results/main'

augmented_df = df.copy()

all_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

for class_name in all_classes:
    current_count = class_counts.get(class_name, 0)
    deficit = target_count - current_count
    
    if deficit > 0:
        print(f"\nProcessing {class_name}: need {deficit} more images")
        
        pattern = os.path.join(base_dir, class_name, '*.png')
        available_images = glob.glob(pattern)
        
        if not available_images:
            print(f"Warning: No images found for {class_name} in {pattern}")
            continue
        
        selected_images = available_images[:deficit]
        
        new_rows = []
        for img_path in selected_images:
            new_row = {
                'img_path': img_path.replace("/mnt/SkinGenerativeModel/code/", ""),
                'seg_path': '',
                'class': class_name,
                'prompt': f'An image of a skin area with {class_name}.',
                'dataset_split': 'test'
            }
            new_rows.append(new_row)
        
        new_df = pd.DataFrame(new_rows)
        augmented_df = pd.concat([augmented_df, new_df], ignore_index=True)
        
        print(f"Added {len(new_rows)} images for {class_name}")
    else:
        print(f"{class_name} already has {current_count} images (>= {target_count})")

final_class_counts = Counter(augmented_df['class'])
print("\nFinal class distribution:")
for class_name, count in final_class_counts.items():
    print(f"{class_name}: {count}")

output_file = '/mnt/SkinGenerativeModel/code/data/local/Dermofit/input/metadata_rebuttal.csv'
augmented_df.to_csv(output_file, index=False)
print(f"\nAugmented metadata saved to {output_file}")