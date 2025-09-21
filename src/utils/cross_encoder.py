from sentence_transformers import CrossEncoder
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

'''
Здесь нужно будет написать все то же самое, что в скрипте cosin_similarity,
только функция get_similarity будет другой. И другая модель.
'''

model_name = 'sentence-transformers/stsb-mpnet-base-v2'

model = CrossEncoder(model_name)

def get_similarity(sent1, sent2, model):
    score = model.predict([(sent1, sent2)])
    return score

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_path', type=str, default='./data/splited_samples/en_positive_splited_data_SMM4H_2025_clean.csv', help='set a path to the file with positive samples')
args = parser.parse_args()

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

target_samples_similarity_scorses = []

for sent in TARGET_SAMPLES:
    sent_similarity_scorses = []
    for raw_sent in texts:
        cosin_score = float (get_similarity(sent, raw_sent, model)[0])      
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