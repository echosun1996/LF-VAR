import pandas as pd
import argparse

def process_csv(file_path):
    df = pd.read_csv(file_path)

    df['Infection'] = None
    df['Type'] = None

    for i, text in df['Text'].items():
        if "of " in text and "." in text:
            df.at[i, 'Infection'] = text.split("of ")[-1].split(".")[0].strip()
        elif "as " in text:
            df.at[i, 'Infection'] = text.split("as ")[-1].split(".")[0].strip()

        if "clinical" in text:
            df.at[i, 'Type'] = "clinical"
        elif "dermoscopic" in text:
            df.at[i, 'Type'] = "dermoscopic"
    print("All Rows:", df.shape[0], "Columns:", df.shape[1])
    output_file = file_path.replace(".csv", "_processed.csv")
    df.to_csv(output_file, index=False)


    filtered_df = df['Infection'].value_counts()
    filtered_df = filtered_df[filtered_df >= 500]

    result_df = df[df['Infection'].isin(filtered_df.index)]

    result_df.to_csv(file_path.replace(".csv", "_filtered.csv"), index=False)

    pivot_table = result_df.pivot_table(index='Infection', columns='Type', aggfunc='size', fill_value=0)
    pivot_table['sum'] = pivot_table.sum(axis=1)

    print("Infection >= 500:")
    print(pivot_table)
    print("After filter Rows:", df.shape[0], "Columns:", df.shape[1])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process merged_meta.csv to add Infection and Type columns")
    parser.add_argument('-file_path', required=True, help="Path to the merged_meta.csv file")
    args = parser.parse_args()

    process_csv(args.file_path)