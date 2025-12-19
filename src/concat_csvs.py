
import pandas as pd
import os
import glob

def concatenate_csvs(output_file='outputs/all_predictions.csv'):
    """
    Concatenates all prediction CSVs in the outputs directory.
    """
    csv_files = ['outputs/train_predictions.csv', 'outputs/val_predictions.csv', 'outputs/test_predictions.csv']
    dfs = []
    
    for f in csv_files:
        if os.path.exists(f):
            print(f"Reading {f}...")
            df = pd.read_csv(f)
            # Add a 'set' column to distinguish
            if 'train' in f:
                df['set'] = 'train'
            elif 'val' in f:
                df['set'] = 'validation'
            elif 'test' in f:
                df['set'] = 'test'
            dfs.append(df)
        else:
            print(f"Warning: {f} not found.")
            
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        print(f"Concatenated CSV saved to {output_file}")
    else:
        print("No CSVs found to concatenate.")

if __name__ == "__main__":
    concatenate_csvs()
