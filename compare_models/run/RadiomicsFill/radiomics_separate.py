import argparse
import os
import pandas as pd

def get_file_stems(directory):
    
    stems = set()
    if not os.path.exists(directory):
        raise ValueError(f"Directory not found: {directory}")
    for fname in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, fname)):
            stem = os.path.splitext(fname)[0]
            stems.add(stem)
    return stems

def save_if_needed(df, file_stems, save_path, split_name):
    
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        if len(existing_df) == len(file_stems):
            print(f"[{split_name}] {save_path} already exists and row count matches ({len(file_stems)}). Skipping save.")
            return
        else:
            print(f"[{split_name}] {save_path} exists but row count mismatch ({len(existing_df)} vs {len(file_stems)}). Overwriting...")
    else:
        print(f"[{split_name}] {save_path} does not exist. Saving...")

    df.to_csv(save_path, index=False)
    print(f"Saved {split_name} set with {len(df)} samples to {save_path}")

def main(args):
    df = pd.read_csv(args.data_path)

    df = df.drop(columns=[col for col in ['feature_class', 'id'] if col in df.columns])

    train_dir = os.path.join(args.input_path, "train/HAM10000_img")
    val_dir = os.path.join(args.input_path, "val/HAM10000_img")
    test_dir = os.path.join(args.input_path, "test/HAM10000_img")

    train_files = get_file_stems(train_dir)
    val_files = get_file_stems(val_dir)
    test_files = get_file_stems(test_dir)

    if 'file_name' not in df.columns:
        raise ValueError("CSV must contain a 'file_name' column.")

    df['file_stem'] = df['file_name'].apply(lambda x: os.path.splitext(x)[0])

    train_df = df[df['file_stem'].isin(train_files)].drop(columns=['file_stem'])
    val_df = df[df['file_stem'].isin(val_files)].drop(columns=['file_stem'])
    test_df = df[df['file_stem'].isin(test_files)].drop(columns=['file_stem'])

    os.makedirs(args.output_path, exist_ok=True)

    save_if_needed(train_df, train_files, os.path.join(args.output_path, "radiomics_final_train.csv"), "train")
    save_if_needed(val_df, val_files, os.path.join(args.output_path, "radiomics_final_val.csv"), "val")
    save_if_needed(test_df, test_files, os.path.join(args.output_path, "radiomics_final_test.csv"), "test")

    trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    trainval_files = train_files.union(val_files)
    save_if_needed(trainval_df, trainval_files, os.path.join(args.output_path, "radiomics_final_trainval.csv"), "trainval")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to radiomics_final.csv')
    parser.add_argument('--input_path', type=str, required=True, help='Directory containing train/val/test folders')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save separated CSVs')
    args = parser.parse_args()
    main(args)