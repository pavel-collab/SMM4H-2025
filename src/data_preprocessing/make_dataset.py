from pathlib import Path
import argparse
import pandas as pd
import json
import datasets
import os

from utils import ParsedFileName, LANGUAGES

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

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#! Переделали логику скрипта. Теперь мы используем не разделенные файлы с положительными примерами, а
#! конкретный указанный файл с данными.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

JSON_SAVE_DIRNAME_TEMPLATE = 'json_datasets'

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set path to dir with data')
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
        
#TODO: нужно переделать путь, куда сохраняется json-датасет. Сейкас пока захардкодили
# сохраняем данные в json формате; позже мы импортируем их в скрипте для обучения модели и создадим объект datasets.Dataset
save_file_path = Path(f'./data/{JSON_SAVE_DIRNAME_TEMPLATE}/{target_file_info.filename}_json.json')

if not save_file_path.parent.exists():
    os.mkdir(save_file_path.parent.absolute())
                               
with open(save_file_path.absolute(), 'w') as json_file:
    json.dump(dataset, json_file)
    
def import_dataset_from_json(json_file_path: str):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        dataset = datasets.Dataset.from_list(data)
    return dataset