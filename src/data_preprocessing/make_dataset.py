from pathlib import Path
import argparse
import pandas as pd
import json
import datasets
import os

from utils import ParsedFileName, LANGUAGES

JSON_SAVE_DIRNAME_TEMPLATE = 'json_datasets'

#TODO: take a path with positive samples automatic
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set path to root dir with data')
parser.add_argument('--language', type=str, default='en', help='set language of raw positive dataset')
args = parser.parse_args()

lang = args.language
assert(lang in LANGUAGES)

root_data_dir_path = Path(args.data_path)
assert(root_data_dir_path.exists())
splited_data_path = Path(f'{root_data_dir_path.absolute()}/splited_samples/')
assert(splited_data_path.exists())

splited_data_files = os.listdir(splited_data_path.absolute())
if len(splited_data_files) == 0:
    raise Exception(f'there are no files in {splited_data_path.absolute()}')

target_file_info = None
for filename in splited_data_files:
    file_info = ParsedFileName(f'{splited_data_path.absolute()}/{filename}')
    if file_info.lang == lang and file_info.positive:
        target_file_info = file_info
        break
    
assert(target_file_info is not None)
assert(target_file_info.file_extension == '.csv')

df = pd.read_csv(target_file_info.filepath.absolute())

PROMPT = 'Generate an example of a comment or tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug.'

# чуть позже это мы уберем для оптимизации. Будем фильтровать данные на предобработке
dataset = []
for _, row in df.iterrows():
    sample = {}
    convarsation = []
    convarsation.append({'content': PROMPT, 'role': 'user'})
    convarsation.append({'content': row['text'], 'role': 'assistant'})
    sample['conversations'] = convarsation
    dataset.append(sample)
        
save_file_path = Path(f'{root_data_dir_path.absolute()}/{JSON_SAVE_DIRNAME_TEMPLATE}/{target_file_info.filename}_json.json')

if not save_file_path.parent.exists():
    os.mkdir(save_file_path.parent.absolute())
                               
with open(save_file_path.absolute(), 'w') as json_file:
    json.dump(dataset, json_file)
    
def import_dataset_from_json(json_file_path: str):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        dataset = datasets.Dataset.from_list(data)
    return dataset