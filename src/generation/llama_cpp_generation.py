from llama_cpp import Llama
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

'''
В данном скрипте мы используем модуль llama_cpp для генерации данных.
llama_cpp локально запускает модели из .gguf файлов, поэтому предварительно необходимо
скачать нужные файлы с hugging face или сохранить предобученные модели в этом формате.
'''

SAVE_GENERATIONS_PATH_TEMPLATE = 'generations'

'''
Поскольку данный способ запуска моделей требует явного обращения к локальной модели в формате .gguf,
необходимо явно указать путь к файлу в системе (скачайте предварительно модель GGUF формата).
'''
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, default='./saved_models/mistral-7b-instruct-v0.1.Q4_K_M.gguf', help='set a path to generation model gguf file')
parser.add_argument('-n', '--num_generations', type=int, default=5, help='set number of output generations')
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set path to root dir with data')
args = parser.parse_args()

num_generations = args.num_generations
assert(num_generations > 0)

model_path = Path(args.model_path)
if not  model_path.exists():
    raise FileExistsError(f'model path {model_path.absolute()} is not correct')

model_name = model_path.name.removesuffix('.gguf')

root_data_path = Path(args.data_path)
assert(root_data_path.exists())

# Инициализация модели
llm = Llama(
    model_path=str(model_path.absolute()),  # Путь к GGUF модели
    n_ctx=2048,  # Размер контекста
    n_threads=4  # Количество потоков для обработки
)

prompt = f"""
Generate an example of a comment or tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug.
Don't use general phrases, give an example of tweet or comment.
Make a brief answer. Only text of the unswer without introduction phrases.
"""

generations = []
for _ in tqdm(range(num_generations)):
    # Генерация текста
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are helpfull assistent"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,  # Максимальное количество токенов в ответе
        temperature=0.7,  # "Творческость" ответа (0-1)
        stop=["\n"]       # Символы, при которых генерация останавливается
    )

    generations.append(output["choices"][0]["message"]["content"])

assert(len(generations) > 0)
assert(len(generations) == num_generations)

df = pd.DataFrame(generations, columns=['text'])

# Сохраняем в CSV
df.to_csv(f'{root_data_path.absolute()}/{SAVE_GENERATIONS_PATH_TEMPLATE}/{model_name}_generation.csv', index=False, encoding='utf-8')