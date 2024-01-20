import json
import os
from datasets import Dataset
import pandas as pd
import argparse

def process_json_files(directory, all_data):
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                all_data.extend(data)

def main():
    parser = argparse.ArgumentParser(description="Collect all translation json files.")
    parser.add_argument('--input_dir', type=str, 
                        help='Folder containing json files or subfolders with json files.')
    parser.add_argument('--output_dir', type=str,
                        help='Target directory of the Huggingface Dataset.')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    all_data = []

    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isfile(item_path) and item.endswith('.json'):
            process_json_files(input_dir, all_data)
            break  # No need to continue if json files are found directly in input_dir
        elif os.path.isdir(item_path):
            process_json_files(item_path, all_data) # Process json files in each folder

    javanese_data = [record for record in all_data if record['lang'] == 'jav_Latn']  # Filter out non-Javanese records

    dataset = Dataset.from_pandas(pd.DataFrame(javanese_data)) 
    dataset.push_to_hub(output_dir)

if __name__ == "__main__":
    main()