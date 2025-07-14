from pathlib import Path
import argparse
import pandas as pd
import json
import datasets

JSON_SAVE_PATH = './data/json_dataset/'

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, default='./data/en_train_data_SM4H_2025_clean.csv', help='set path to file with data')
args = parser.parse_args()

data_name = args.data_path.removesuffix('.csv')

data_path = Path(args.data_path)

assert(data_path.exists())

df = pd.read_csv(data_path.absolute())

PROMPT = '' #TODO: set a prompt

dataset = []
for _, row in df.iterrows():
    if row['label'] == 1:
        dataset.append({'content': PROMPT, 'role': 'user'})
        dataset.append({'content': row['text'], 'role': 'user'})
        
with open(f'{JSON_SAVE_PATH}/{data_name}.csv', 'w') as json_file:
    json.dump(dataset, json_file)
    
def import_dataset_from_json(json_file_path: str):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        dataset = datasets.Dataset.from_dict(data)
    return dataset