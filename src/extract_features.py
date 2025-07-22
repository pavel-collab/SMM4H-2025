from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from utils.utils import get_train_eval_dataset
import json

train_dataset, _ = get_train_eval_dataset()

texts, labels = train_dataset['text'], train_dataset['label']

# 1. Векторизация текста
vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1,2),
    analyzer='word'  # можно поменять на 'char_wb', если хочется символы
)
X = vectorizer.fit_transform(texts)

# 2. Быстрая логистическая регрессия
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# 3. Получение признаков
feature_names = vectorizer.get_feature_names_out()

results = {}

# Для каждого класса выводим топ-10 самых влиятельных слов
#! при бинарной классификации логистическая регрессия распознает только положительный класс, а отрицательный выбирается методом исключения
# for i, class_label in enumerate(model.classes_):
i = class_label = 0
top10 = np.argsort(model.coef_[i])[-15:]
results[class_label] = [feature_names[idx] for idx in reversed(top10)]
print(f"\nClass: {class_label}")
for idx in reversed(top10):
    print(f"{feature_names[idx]} : {model.coef_[i][idx]:.4f}")
        
# Запись результатов в JSON файл
with open('top_features.json', 'w') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)