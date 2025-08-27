import pandas as pd
import os
from collections import Counter
import glob

metadata_file = '/mnt/SkinGenerativeModel/code/data/local/Dermofit/input/metadata.csv'
df = pd.read_csv(metadata_file)

all_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

balanced_df = pd.DataFrame()

base_dir = '/mnt/SkinGenerativeModel/code/data/compare_results/main'

for class_name in all_classes:
    class_df = df[df['class'] == class_name]
    
    if len(class_df) >= 50:
        sampled_original = class_df.sample(n=50, random_state=42)
    else:
        sampled_original = class_df
        print(f"Warning: {class_name} has only {len(class_df)} images in original metadata")
    
    balanced_df = pd.concat([balanced_df, sampled_original], ignore_index=True)
    
    pattern = os.path.join(base_dir, class_name, '*.png')
    available_images = glob.glob(pattern)
    
    if not available_images:
        print(f"Warning: No images found for {class_name} in {pattern}")
        continue
    
    selected_images = available_images[:50]
    
    if len(selected_images) < 50:
        print(f"Warning: Only {len(selected_images)} supplementary images available for {class_name}")
    
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
    balanced_df = pd.concat([balanced_df, new_df], ignore_index=True)
    
    print(f"Class {class_name}: {len(sampled_original)} original + {len(new_rows)} supplementary = {len(sampled_original) + len(new_rows)} total")

final_class_counts = Counter(balanced_df['class'])
print("\nFinal class distribution:")
for class_name, count in final_class_counts.items():
    print(f"{class_name}: {count}")

output_file = '/mnt/SkinGenerativeModel/code/data/local/Dermofit/input/metadata_rebuttal_samesize.csv'

balanced_df.to_csv(output_file, index=False)
print(f"\nBalanced metadata saved to {output_file}")

print(f"\nTotal images in balanced dataset: {len(balanced_df)}")
print(f"Images per class (target=100): {dict(final_class_counts)}")