from transformers import pipeline
import torch
import pandas as pd
import argparse
from pathlib import Path

'''
В этом скрипте мы будем фильтровать предложения по токсичности.
Идея в том, чтобы проверить гипотизу о том, что в положительных примерах 
содержится больше токсичной и нецензурной лексики. Будем использовать
модель unitary/multilingual-toxic-xlm-roberta. На hugging face для этой
модели есть пример использования с нестандартными кастомными классами,
но пока что попробуем обойтись стандарным api через pipeline.
'''

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, default='./data/splited_samples/fr_positive_splited_data_SMM4H_2025_clean.csv', help='Path to the csv file with samples')
args = parser.parse_args()

data_path = Path(args.data_path)
assert(data_path.exists())

df = pd.read_csv(data_path.absolute())
texts = df['text'].tolist()

# Опционально: проверка доступности GPU
device = 0 if torch.cuda.is_available() else -1
print(f"Используемое устройство: {'GPU' if device == 0 else 'CPU'}")

# Создание пайплайна для классификации текста
classifier = pipeline(
    "text-classification",
    model="unitary/multilingual-toxic-xlm-roberta",
    device=device,  # автоматическое использование GPU если доступно
    truncation=True  # важно для длинных текстов
)

# Выполнение классификации
results = classifier(texts)
toxity_score = [res['score'] for res in results]
toxity = [1 if score > 0.85 else 0 for score in toxity_score]

df['toxity_score'] = toxity_score
df['toxity'] = toxity

#! Пока что сохраняем локально, потом посмотрим, куда это сохранить
df.to_csv(f'{data_path.name.removesuffix('.csv')}_toxity_result.csv')