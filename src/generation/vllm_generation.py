from vllm import LLM, SamplingParams
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

'''
В данном скрипте генерируем и сохраняем синтетические данные
с помощью инструмента vllm.
'''

SAVE_GENERATIONS_PATH_TEMPLATE = 'generations'

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n_samples', type=int, default=10, help='number of generated samples')
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set path to root dir with data')
args = parser.parse_args()

root_data_path = Path(args.data_path)
assert(root_data_path.exists())

model_name = 'Qwen/Qwen2.5-0.5B-Instruct'

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
Generate an example of a comment or tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug.
Don't use general phrases, give an example of tweet or comment.
Make a brief answer. Only text of the unswer without introduction phrases.
"""

sampling_params = SamplingParams(
            temperature=0.9, top_k=500, top_p=0.9, max_tokens=512, n=n_samples
        )

# Генерация
outputs = llm.generate(prompt, sampling_params)

#TODO полная строка генерации outputs[0].outputs[0] -- разобраться что значат эти индексы
#TODO в генерациях остаются артефакты генерации в начале предложения, придумать, как убрать их
generations = [output.text.strip() for output in tqdm(outputs[0].outputs)]   

df = pd.DataFrame(generations, columns=['text'])

save_path = Path(f'{root_data_path.absolute()}/{SAVE_GENERATIONS_PATH_TEMPLATE}/')
if not save_path.exists():
    os.mkdir(save_path.absolute())
    
assert(save_path.exists())

#TODO: add label 1
df.to_csv(f'{save_path.absolute()}/{model_name.replace('/', '-')}_generation.csv', index=False) 