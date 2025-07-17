from llama_cpp import Llama
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm
import json
import random
from utils import debug_print, ParsedFileName, LANGUAGES
import os
import numpy as np

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
Пока что этот конкретный пайплайн генерации справляется не особо. Модель генерирует общие предупредления об ADE
без конкретных случаев, которые ей передаются в качестве примера.
'''
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

'''
В данном скрипте мы используем модуль llama_cpp для генерации данных.
llama_cpp локально запускает модели из .gguf файлов, поэтому предварительно необходимо
скачать нужные файлы с hugging face или сохранить предобученные модели в этом формате.
'''

SAVE_GENERATIONS_PATH_TEMPLATE = 'generations'
'''
При few-shot генерации мы подаем модели на вход примеры запросов и выходных данных. Для
этого мы используем json датасет, сформированный из первоначальных сырых данных (см make_dataset.py).
Приблема в том, что размер контекстного окна модели (см параметр max_model_len) ограничен, в то время
как примеров генерации множество. Поэтому усстанавливаем ограничение на количество примеров из
json файла, которые мы показываем модели в качестве примера.

Zero-shot генерация? Вроде как передаем примеры, но результаты генерации, как будто примеров не было вообще.
'''
CONVERSATION_LIMIT = 10

'''
Поскольку данный способ запуска моделей требует явного обращения к локальной модели в формате .gguf,
необходимо явно указать путь к файлу в системе (скачайте предварительно модель GGUF формата).
'''
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, default='./saved_models/mistral-7b-instruct-v0.1.Q4_K_M.gguf', help='set a path to generation model gguf file')
parser.add_argument('-n', '--num_generations', type=int, default=5, help='set number of output generations')
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set path to root dir with data')
parser.add_argument('--language', type=str, default='en', help='set language of raw positive dataset')
args = parser.parse_args()

num_generations = args.num_generations
assert(num_generations > 0)

model_path = Path(args.model_path)
if not  model_path.exists():
    raise FileExistsError(f'model path {model_path.absolute()} is not correct')

model_name = model_path.name.removesuffix('.gguf')

lang = args.language
assert(lang in LANGUAGES)

# находим файл с закотовленным json датасетом см скрипт make_dataset.py
root_data_dir_path = Path(args.data_path)
assert(root_data_dir_path.exists())
json_data_path = Path(f'{root_data_dir_path.absolute()}/json_datasets/')
assert(json_data_path.exists())

json_data_files = os.listdir(json_data_path.absolute())
if len(json_data_files) == 0:
    raise Exception(f'there are no files in {json_data_path.absolute()}')

target_file_info = None
for filename in json_data_files:
    file_info = ParsedFileName(f'{json_data_path.absolute()}/{filename}')
    if file_info.lang == lang:
        target_file_info = file_info
        break
    
assert(target_file_info is not None)
assert(target_file_info.file_extension == '.json')

# Инициализация модели
llm = Llama(
    model_path=str(model_path.absolute()),  # Путь к GGUF модели
    n_ctx=2048,  # Размер контекста
    n_threads=4  # Количество потоков для обработки
)

prompt = f"""
Generate one tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug.
Don't use images, don't use emojies.
"""

system_prompt = {
        "role": "system",
        "content": "You are a helpful medical assistant"
    }

conversations = [system_prompt]
# import prepared json dataset made with make_dataset.py
with open(target_file_info.filepath.absolute(), 'r') as json_file:
    data = json.load(json_file)

# перемешиваем данные внутри json датасета (хотя, в целом, это не обязательно, т к там только положительные примеры)
random.shuffle(data)

for item in data[:CONVERSATION_LIMIT]:
    conversations.extend(item["conversations"])

llm.create_chat_completion(
      messages = conversations
)

generations = []
for _ in tqdm(range(num_generations)):
    # Генерация текста
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are helpfull assistent"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,  # Максимальное количество токенов в ответе
        temperature=1.0,  # "Творческость" ответа (0-1)
        top_p = 0.95, 
        top_k = 64,
        stop=["\n"]       # Символы, при которых генерация останавливается
    )

    generations.append(output["choices"][0]["message"]["content"].strip().replace("\"", ""))

assert(len(generations) > 0)
assert(len(generations) == num_generations)

labels = np.ones((len(generations))).astype(int)
df = pd.DataFrame({'text': generations, 'label': labels})

# Сохраняем в CSV
df.to_csv(f'{root_data_dir_path.absolute()}/{SAVE_GENERATIONS_PATH_TEMPLATE}/{model_name}_generation.csv', index=False, encoding='utf-8')