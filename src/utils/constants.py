import torch
from dotenv import load_dotenv
import os

'''
Перед запуском необходимо задать в текущем каталоге файл .env и прописать в нем абсолютный 
путь к файлу с тренировочными данными. Например:

TRAIN_DATA_FILE=~/data/en_train_data_SMM4H_2025_clean.csv
'''

# Загружаем переменные из .env файла
load_dotenv()
train_csv_file = os.getenv("TRAIN_DATA_FILE")

n_classes = 2

batch_size = 8
num_epoches = 3

class ClassWeights:
    class_weights=torch.tensor([1.0, 1.0])

clw = ClassWeights()