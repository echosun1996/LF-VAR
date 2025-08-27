import pandas as pd
import numpy as np

def analyse_isic_classes(csv_file_path):
    
    df = pd.read_csv(csv_file_path)
    
    class_stats = df['class'].value_counts()
    class_percentages = df['class'].value_counts(normalize=True) * 100
    
    stats_df = pd.DataFrame({
        'Class': class_stats.index,
        'Count': class_stats.values,
        'Percentage': class_percentages.values
    })
    
    total_row = pd.DataFrame({
        'Class': ['Total'],
        'Count': [len(df)],
        'Percentage': [100.0]
    })
    
    stats_df = pd.concat([stats_df, total_row], ignore_index=True)
    
    stats_df['Percentage'] = stats_df['Percentage'].round(2)
    
    return stats_df, df

def generate_detailed_stats(df):
    
    detailed_stats = {}
    
    if 'dataset_split' in df.columns:
        split_stats = df['dataset_split'].value_counts()
        detailed_stats['dataset_split'] = split_stats.to_dict()
    
    if 'dataset_split' in df.columns:
        split_class_dist = pd.crosstab(df['class'], df['dataset_split'])
        detailed_stats['class_by_split'] = split_class_dist
    
    return detailed_stats

if __name__ == "__main__":
    csv_path = "/mnt/SkinGenerativeModel/code/data/local/HAM10000/input/metadata.csv"
    
    try:
        stats_table, original_df = analyse_isic_classes(csv_path)
        
        print("ISIC Dataset Class Statistics")
        print("=" * 30)
        print(stats_table.to_string(index=False))
        print()
        
        detailed_stats = generate_detailed_stats(original_df)
        
        if 'dataset_split' in detailed_stats:
            print("\nDataset Split Statistics")
            print("=" * 30)
            for split, count in detailed_stats['dataset_split'].items():
                print(f"{split}: {count}")
        
        if 'class_by_split' in detailed_stats:
            print("\nClass Distribution by Dataset Split")
            print("=" * 30)
            print(detailed_stats['class_by_split'])
        
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_path}'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")