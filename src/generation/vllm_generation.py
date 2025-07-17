from vllm import LLM, SamplingParams
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from utils import debug_print, ParsedFileName, LANGUAGES
import json
import random

'''
В данном скрипте генерируем и сохраняем синтетические данные
с помощью инструмента vllm.
'''

SAVE_GENERATIONS_PATH_TEMPLATE = 'generations'

'''
При few-shot генерации мы подаем модели на вход примеры запросов и выходных данных. Для
этого мы используем json датасет, сформированный из первоначальных сырых данных (см make_dataset.py).
Приблема в том, что размер контекстного окна модели (см параметр max_model_len) ограничен, в то время
как примеров генерации множество. Поэтому усстанавливаем ограничение на количество примеров из
json файла, которые мы показываем модели в качестве примера.

Few-shot генерация
'''
CONVERSATION_LIMIT = 10

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n_samples', type=int, default=10, help='number of generated samples')
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set path to root dir with data')
parser.add_argument('--language', type=str, default='en', help='set language of raw positive dataset')
args = parser.parse_args()

root_data_path = Path(args.data_path)
assert(root_data_path.exists())

model_name = 'Qwen/Qwen2.5-0.5B-Instruct'

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

# Загружаем LLM
llm = LLM(model=model_name,
        #   tokenizer_mode="auto",
        #   dtype="auto",
          gpu_memory_utilization=0.9,
          max_num_seqs=128,
          max_model_len=2048)

# llm = LLM(model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
#           tokenizer_mode="auto",
#           dtype="auto",
#           gpu_memory_utilization=0.9,
#           max_num_seqs=128,
#           max_model_len=2048)

n_samples = args.n_samples

# Промпт с уточнением стиля
prompt = f"""
Generate one tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug.
Don't use images, don't use emojies.
"""

'''
Судя по всему, при использовании llm.chat()
модель не обращает внимание на параметр n_samples и
генерирует один единственный пример.
'''
sampling_params = SamplingParams(
            temperature=0.9,  # randomness of the sampling
            top_k=500, 
            top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
            max_tokens=256, # Maximum number of tokens to generate per output sequence.
            # n=n_samples, # Number of output sequences to return for each prompt.
            # stop=['\n\n', '. '],      # List of strings that stop the generation when they are generated.
            skip_special_tokens=True, # Whether to skip special tokens in the output.
        )

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

generations = []
for _ in tqdm(range(n_samples)):
    outputs = llm.chat(conversations,
                    sampling_params=sampling_params,
                    use_tqdm=False)

    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        generations.append(f"{generated_text}")

df = pd.DataFrame(generations, columns=['text'])

save_path = Path(f'{root_data_path.absolute()}/{SAVE_GENERATIONS_PATH_TEMPLATE}/')
if not save_path.exists():
    os.mkdir(save_path.absolute())
    
assert(save_path.exists())

#TODO: add label 1
df.to_csv(f'{save_path.absolute()}/{model_name.replace('/', '-')}_generation.csv', index=False) 