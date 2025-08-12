import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

'''
В этом скрипте мы будем использвать библиотеку nltk для оценки
метрик похожести предложений. Мы будем брать массив целевых предложений и
массив предложений из исходного датасета. Мы будем сравнивать 
каждое целевое предложение с каждым из исходного датасета по трем метрикам:
bleu, rougel, meteor. Затем каждый массив оценок (для каждого целевого предлжения
должно быть 3 массива, потому что у нас 3 метрики) будут отрисованы в виде гистограммы.
идея анализа гистограмм такая же, как в скрипте cosin_similarity. Если будет большая дисперсия 
в значениях, значит оценка несостоятельная. Если дисперсия маленькая, значит имеет смысл 
применять эту оценку.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_path', type=str, default='./data/splited_samples/en_positive_splited_data_SMM4H_2025_clean.csv', help='set a path to the file with positive samples')
args = parser.parse_args()

def calculate_bleu(reference, candidate):
    return sentence_bleu([reference], candidate)

def calculate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(" ".join(candidate), " ".join(reference))  
    return scores[0]['rouge-l']['f']  

def calculate_meteor(reference, candidate):
    return meteor_score([reference], candidate)

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

nltk.download('wordnet')

target_samples_bleu_scorses = []
#! временно заморозили эти метрики, пока отладим скрипт на bleu
# target_samples_rouge_scorses = []
# target_samples_meteor_scorses = []

for sent in TARGET_SAMPLES:
    sent_bleu_scorses = []
    for raw_sent in texts:
        bleu_score = calculate_bleu(raw_sent, sent)
        sent_bleu_scorses.append(bleu_score)
    target_samples_bleu_scorses.append(sent_bleu_scorses)

num_rows = num_samples // 3

fig, axes = plt.subplots(num_samples, num_rows, figsize=(10, 5 * num_samples))

for i, data in enumerate(target_samples_bleu_scorses):
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