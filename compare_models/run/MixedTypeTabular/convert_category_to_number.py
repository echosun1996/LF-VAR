import pandas as pd
import argparse

def main(path):
    mapping = {
        "akiec": 1,
        "bcc": 2,
        "bkl": 3,
        "df": 4,
        "mel": 5,
        "nv": 6,
        "vasc": 7
    }

    df = pd.read_csv(path)

    if "category" not in df.columns:
        raise ValueError(f"'category' column not found in file {path}.")

    if pd.api.types.is_numeric_dtype(df["category"]) and "file_name" not in df.columns:
        print(f"'category' column in file {path} is already numeric, skipping.")
        return

    if "file_name" in df.columns:
        df.drop(columns=["file_name"], inplace=True)
        print(f"'file_name' column in file {path} has been deleted.")
        
    df["category"] = df["category"].map(mapping).fillna(df["category"])

    df["category"] = df["category"].astype(int)

    df.to_csv(path, index=False)
    print(f"Successfully processed and saved: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert category strings to numbers.")
    parser.add_argument("--path", type=str, required=True, help="Path to the CSV file.")
    args = parser.parse_args()

    main(args.path)