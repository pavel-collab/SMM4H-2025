from pathlib import Path
import argparse
import pandas as pd
import json
import datasets
import os

#TODO: take a path with positive samples automatic
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, default='./data/en_train_data_SM4H_2025_clean.csv', help='set path to file with data')
parser.add_argument('-p', '--path', type=str, default='./data/json_dataset/', help='set path to save json dataset')
args = parser.parse_args()

data_name = args.data_path.removesuffix('.csv')

data_path = Path(args.data_path)

assert(data_path.exists())

df = pd.read_csv(data_path.absolute())

PROMPT = 'Generate an example of a comment or tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug.'

# чуть позже это мы уберем для оптимизации. Будем фильтровать данные на предобработке
dataset = []
for _, row in df.iterrows():
    if row['label'] == 1:
        sample = {}
        convarsation = []
        convarsation.append({'content': PROMPT, 'role': 'user'})
        convarsation.append({'content': row['text'], 'role': 'assistant'})
        sample['conversations'] = convarsation
        dataset.append(sample)
        
save_path = Path(args.path)

if not save_path.exists():
    os.mkdir(save_path.absolute())
                
#!? Bug: почему-то json файл пишется не в директорию json_dataset, а в ее родительскую. Пока не разобрался                
with open(f'{save_path.absolute()}/{data_name}.json', 'w') as json_file:
    json.dump(dataset, json_file)
    
def import_dataset_from_json(json_file_path: str):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        dataset = datasets.Dataset.from_list(data)
    return dataset