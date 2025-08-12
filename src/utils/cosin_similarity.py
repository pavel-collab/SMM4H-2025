from sentence_transformers import SentenceTransformer, util
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''
Первая базовая идея -- будем брать набор целевых предложений и сравнивать их с предложениями из исходного датасета.
Для каждого целевого предложения будем рисовать barplot с вычисленными метриками похожести. Таким образом, будет столько
графиков, сколько целевых предложений. Далее можно будет поисследовать это распределение и понять, насколько идея сранвения
предложений состоятельна. Если будет малый разброс значений похожести -- значит эвристика работает хорошо. Если распределение с
большой дисперсией по значениям -- значит, эвристика работает не очень хорошо.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_path', type=str, default='./data/splited_samples/en_positive_splited_data_SMM4H_2025_clean.csv', help='set a path to the file with positive samples')
args = parser.parse_args()

# model_name = 'paraphrase-multilingual-mpnet-base-v2'
model_name = 'distiluse-base-multilingual-cased-v1'

# В этом массиве у нас будут целевые предложения для исследования
# Они будут сравниваться с предложениями из исходного датасета и высчитываться метрика похожести
TARGET_SAMPLES = [
    'banana? hot milk? and randomly lettuce! all contain sleepy bye chems. all i have is trazodone which means dopey all day tomo'
]

num_samples = len(TARGET_SAMPLES)

raw_positive_data_path = Path(args.raw_data_path)
assert(raw_positive_data_path.exists())

df = pd.read_csv(raw_positive_data_path.absolute())
texts = df['text'].tolist()

# Модель, понимающая русский и английский
model = SentenceTransformer(model_name)

def get_cosin_similarity(sent1, sent2, model):
    emb1 = model.encode(sent1, convert_to_tensor=True)
    emb2 = model.encode(sent2, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(emb1, emb2)
    return similarity

target_samples_similarity_scorses = []

for sent in TARGET_SAMPLES:
    sent_similarity_scorses = []
    for raw_sent in texts:
        cosin_score = get_cosin_similarity(sent, raw_sent, model)
        sent_similarity_scorses.append(cosin_score)
    target_samples_similarity_scorses.append(sent_similarity_scorses)

num_rows = num_samples // 3

fig, axes = plt.subplots(num_samples, num_rows, figsize=(10, 5 * num_samples))

for i, data in enumerate(target_samples_similarity_scorses):
    mean = np.mean(data)
    variance = np.var(data)
    std_dev = np.sqrt(variance)

    # Построение барплота
    sns.histplot(data, bins=20, kde=True, ax=axes[i], color='blue', stat="density")

    # Описание графика
    axes[i].axvline(mean, color='red', linestyle='--', label='Среднее')
    axes[i].axvline(mean + std_dev, color='orange', linestyle='--', label='Среднее + Дисперсия')
    axes[i].axvline(mean - std_dev, color='orange', linestyle='--', label='Среднее - Дисперсия')
    axes[i].set_title(f'График для массива {i+1}')
    axes[i].legend()

plt.tight_layout()
plt.show()