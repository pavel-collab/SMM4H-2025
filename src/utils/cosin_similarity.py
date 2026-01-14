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
    'banana? hot milk? and randomly lettuce! all contain sleepy bye chems. all i have is trazodone which means dopey all day tomo',
    "how anyone can say there is no withdrawal from prozac withdrawal is beyond me.  their answer is to go back on it and up the dose.",
    "so i took 2 vyvanse this morning..i think i'm about to have a heart attack?????",
    "i need to start taking my viibryd again but it makes me feel so nauseous ughhh",
    "Venlafaxine gave me horrendous nightmares, & coming of it was horrific.!",
    "last night i was a mermaid and then a wizard who had to save the world while being hunted by the japanese... err thank you #trazodone ?",
    "my medication gives me weird &amp; bizarre dreams. not all bad though. last night i had dinner with the rolling st"
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
        cosin_score = float (get_cosin_similarity(sent, raw_sent, model).to("cpu"))
        sent_similarity_scorses.append(cosin_score) # a vector of similarity scores for one sent from TARGET_SAMPLES (1, texts.size)
    target_samples_similarity_scorses.append(sent_similarity_scorses) # a matrix of scores (TARGET_SAMPLES.size, texts.size)


for i, data in enumerate(target_samples_similarity_scorses):
    # data is a vector (1, texts.size)
    
    mean = np.mean(data)
    variance = np.var(data)
    std_dev = np.sqrt(variance)

    # Построение барплота
    sns.histplot(data, bins=20, kde=True, color='blue', stat="density")

    # Описание графика
    plt.axvline(mean, color='red', linestyle='--', label='Среднее')
    plt.axvline(mean + std_dev, color='orange', linestyle='--', label='Среднее + Дисперсия')
    plt.axvline(mean - std_dev, color='orange', linestyle='--', label='Среднее - Дисперсия')
    plt.title(f'График для массива {i+1}')
    plt.legend()

    plt.tight_layout()
    plt.show()