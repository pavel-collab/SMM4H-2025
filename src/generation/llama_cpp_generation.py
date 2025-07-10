from llama_cpp import Llama
from pathlib import Path
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, default='mistral-7b-instruct-v0.1.Q4_K_M.gguf', help='set a path to generation model gguf file')
parser.add_argument('-n', '--num_generations', type=int, default=5, help='set number of output generations')
args = parser.parse_args()

num_generations = args.num_generations
assert(num_generations > 0)

model_path = Path(args.model_path)
if not  model_path.exists():
    raise FileExistsError(f'model path {model_path.absolute()} is not correct')

model_name = model_path.removesuffix('.gguf')

# Инициализация модели (скачайте предварительно модель GGUF формата)
llm = Llama(
    model_path=model_path.absolute(),  # Путь к GGUF модели
    n_ctx=2048,  # Размер контекста
    n_threads=4  # Количество потоков для обработки
)

prompt = f"""
Generate an example of a comment or tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug.
Don't use general phrases, give an example of tweet or comment.
Make a brief answer. Only text of the unswer without introduction phrases.
"""

generations = []
for _ in range(num_generations):
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
df.to_csv(f'./data/generated/generation_{model_name}.csv', index=False, encoding='utf-8')