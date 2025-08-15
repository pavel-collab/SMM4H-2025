from pathlib import Path
import argparse
import pandas as pd
import json
import datasets
import os

from utils import ParsedFileName, LANGUAGES

#! ATTENTION: depends on file position in project tree
root_dir = Path(__file__).resolve().parent.parent

'''
В этом скрипте мы используем исходные данные (положительные примеры)
для формирования json датасета для обучения unsolth модели генерации.
Генератор предложений unsloth при обучении принимает датасет datasets.Dataset
данных. Если переводить этот формат в json, он будет иметь следующую структуру

[
    {
        "conversations": [
            {
                "content": "prompt 1",
                "role": "user"
            },
            {
                "content": "answer 1",
                "role": "assistant"
            }
        ]
    },
    ...
]
'''

JSON_SAVE_DIRNAME_TEMPLATE = 'json_datasets'

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', help='set path to dir with data')
parser.add_argument('--language', type=str, default='en', help='set language of raw positive dataset')
args = parser.parse_args()

lang = args.language
assert(lang in LANGUAGES)

target_file_info = ParsedFileName(args.data_path)
    
assert(target_file_info is not None)
assert(target_file_info.file_extension == '.csv')

df = pd.read_csv(target_file_info.filepath.absolute())
# Перемешиваем примеры, чтобы в последствии не было проблем с обучением модели
df = df.sample(frac=1)

PROMPT = 'Generate an example of a comment or tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug.'

# Формируем структуру датасета в формате json
dataset = []
for _, row in df.iterrows():
    sample = {}
    convarsation = []
    convarsation.append({'content': PROMPT, 'role': 'user'})
    convarsation.append({'content': row['text'], 'role': 'assistant'})
    sample['conversations'] = convarsation
    dataset.append(sample)

save_json_path = Path(f'{root_dir}/data/{JSON_SAVE_DIRNAME_TEMPLATE}')
if not save_json_path.exists():
    os.mkdir(save_json_path.absolute())
   
# сохраняем данные в json формате; позже мы импортируем их в скрипте для обучения модели и создадим объект datasets.Dataset
save_file_path = Path(f'{save_json_path.absolute()}/{lang}_json_dataset.json')
                               
with open(save_file_path.absolute(), 'w') as json_file:
    json.dump(dataset, json_file)