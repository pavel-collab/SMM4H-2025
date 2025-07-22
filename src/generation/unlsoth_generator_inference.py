from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from pathlib import Path
import argparse
import pandas as pd
import os
from tqdm import tqdm
import re
import numpy as np

'''
Эта функция парсинга НЕ универсальная. Она написана под парсинг ответа модели
unsloth/gemma-3-4b-it с chat_template gemma-3.
Другие модели и шаблоны возможно потребуют другой функции парсинга 
'''
def parce_model_answer(model_response: str):
    model_answer = re.search(r'<start_of_turn>model\n(.*?)<end_of_turn>', model_response, re.DOTALL)
    return model_answer.group(1).strip()

'''
В данном скрипте мы запускаем на инференсе натренированную в скрипте unsloth_generator_inference.py
модель. Напомним, что в скрипте с тренировкой мы сохраняли локально только LoRa адаптеры, а не всю модель.
Для импорта модели используем механизм unsloth. Указываем название модели, chat_template, количество генераций и
путь к общему каталогку с данными (навигация внутри этого каталога прописана в самой программе).

Скрипт генерирует предложения по заданному пользовательскому запросу.

Few-shot генерация
'''

SAVE_MODEL_PATH = './saved_models/' # каталог с сохраненной моделью
SAVE_GENERATIONS_PATH_TEMPLATE = 'generations' # подкаталог в каталоге с данными, куда сохраняем нагенерированные данные

# Промпт для генерации данных
PROMPT = '''
Generate one tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug.
Don't use images or emojies, but you can write a hashtags.
Write the [TWEET] with folowing template:

Tweet: [TWEET]
'''

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str, default='unsloth/gemma-3-4b-it', help='set open source model name')
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set path to root dir with data')
#TODO: make a map from model to chat_template and check availability
parser.add_argument('--chat_template', type=str, default='gemma-3', help='set a chat template for model')
parser.add_argument('-n', '--num_generations', type=int, default=100, help='set number of generation samples')
args = parser.parse_args()

root_data_path = Path(args.data_path)
assert(root_data_path.exists())

model_name = args.model_name
chat_template = args.chat_template

lora_adapters_save_path = Path(f'{SAVE_MODEL_PATH}/{args.model_name.replace('/', '-')}-lora-adapters')
assert(lora_adapters_save_path.exists())

# Загрузка модели с адаптерами
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = str(lora_adapters_save_path.absolute()),  # Путь к адаптерам
    max_seq_length = 2048,    # Должно совпадать с обучением
    dtype = None,             # Авто-выбор (float16, bfloat16)
    load_in_4bit = True,      # Для экономии памяти
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = chat_template,
)

messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : PROMPT,
    }]
}]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)

# задаем количество примеров, которое надо нагенерировать
num_generations = args.num_generations
assert(num_generations > 0)

generations = []

#! При генерации большого количества примеров возможно имеет смысл делать периодическую выгрузку данных в csv файл
#! Если задать количество генераций, условно 10000000, то можно просто перегрузить оперативную память и
#! потерять сгенерированные примеры. Так что стоит выбрать оптимальные размер батча для генерации и записывать
#! сгенерированные данные на диск.
# генерация синтетических данных
for _ in tqdm(range(num_generations)):
    outputs = model.generate(
        **tokenizer(text=[text], return_tensors = "pt").to("cuda"),
        max_new_tokens = 256, # Increase for longer outputs!
        # Recommended Gemma-3 settings!
        temperature = 1.0, 
        top_p = 0.95, 
        top_k = 64,
    )
    
    # парсим ответ модели, убираем ненужные артефакты генерации
    parced_response = parce_model_answer(tokenizer.batch_decode(outputs)[0])
    parced_response = parced_response[len("Tweet: "):]
    parced_response.strip().replace("\"", "") # убираем кавычки
    generations.append(parced_response)
 
labels = np.ones((len(generations))).astype(int)
# Сохраняем данные в csv файл
df = pd.DataFrame({'text': generations, 'label': labels})

save_path = Path(f'{root_data_path.absolute()}/{SAVE_GENERATIONS_PATH_TEMPLATE}/')
if not save_path.exists():
    os.mkdir(save_path.absolute())
    
assert(save_path.exists())

#! Обратите внимание, что этот метод перезаписывает документ, то есть контент будет потерян
df.to_csv(f'{save_path.absolute()}/{args.model_name.replace('/', '-')}_generation.csv', index=False)