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

# В этом массиве у нас будут целевые предложения для исследования
# Они будут сравниваться с предложениями из исходного датасета и высчитываться метрика похожести

# TARGET_SAMPLES = [
#     "Incidence and risk of denosumab-related hypocalcemia in cancer patients: a systematic review and pooled analys...",
#     "I'm gonna take Cymbalta just for the side effects.",
#     "I actually watched that newscast about breast cancer &amp; chemo. apparently early to mid-stage cancer patients may not need chemo. Just estrogen therapy,and the usual removal procedure.",
#     "seroquel. please. ""drake is the man of my dreams""",
#     "Asthma is the talk on Froome. My diagnosis is COPD and he was taking Advair mid-stage.",
#     "Anyone notice that Xarelto commercials are almost identical to Viagra commercials? I just like commercials with sexy older men.",
#     "Reduces heart attacks and do not increase any stroke or clotting. So overall,estrogen-only hormone replacement therapy is very protective from breast cancer. It is safe than originally reported and much more safe than people perceive",
#     "It's Viibryd. I can't afford it so I get samples from my psych and we had to bump my appt.",
#     "Mom texted ""Mark in HollyWood"" ... I still have it in the air Who Ya Got?",
#     "SSRI Birth Injury Drugs:  Celexa, Effexor, Lexapro, Paxil, Prozac, and Zoloft. Call now to speak with an attorney.",
# ]

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

nltk.download('wordnet')

target_samples_bleu_scorses = []
#! временно заморозили эти метрики, пока отладим скрипт на bleu
# target_samples_rouge_scorses = []
# target_samples_meteor_scorses = []

for sent in TARGET_SAMPLES:
    sent_bleu_scorses = []
    for raw_sent in texts:
        bleu_score = calculate_bleu(raw_sent, sent)
        # bleu_score = calculate_rouge(raw_sent, sent)
        sent_bleu_scorses.append(bleu_score)
    target_samples_bleu_scorses.append(sent_bleu_scorses)

for i, data in enumerate(target_samples_bleu_scorses):
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