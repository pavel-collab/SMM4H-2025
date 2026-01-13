from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
# from sklearn.ensemble import StackingClassifier

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from utils.utils import (fix_random_seed,
                         get_device,
                         get_train_eval_dataset)

fix_random_seed()
device = get_device()

train_dataset, test_dataset = get_train_eval_dataset(use_generation=False,
                                                     get_class_weight_flag=True)

X_train = train_dataset['text']
y_train = train_dataset['label']
X_test = test_dataset['text']
y_test = test_dataset['label']

# 2. Базовая модель 1: TF-IDF + LogisticRegression
model_tfidf_lr = make_pipeline(
    TfidfVectorizer(max_features=5000),
    LogisticRegression(max_iter=200)
)

# 3. Базовая модель 2: Hugging Face (DistilBERT)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Подготовка датасета для HF
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Модель
hf_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Обучение HF модели
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_dir="./logs",
    learning_rate=5e-5
)

trainer = Trainer(
    model=hf_model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# Функция для получения вероятностей из HF модели
def hf_predict_proba(texts):
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = hf_model(**encodings)
        probs = torch.softmax(outputs.logits, dim=1).numpy()
    return probs

# 4. Формируем вход для мета-модели
# Обучаем первую модель
model_tfidf_lr.fit(X_train, y_train)

# Предсказания на train для meta-learner
train_pred1 = model_tfidf_lr.predict_proba(X_train)
train_pred2 = hf_predict_proba(X_train)

# Объединяем признаки
import numpy as np
train_meta = np.hstack([train_pred1, train_pred2])

# Обучаем meta-learner
meta_model = LogisticRegression()
meta_model.fit(train_meta, y_train)

# 5. Оценка на тесте
test_pred1 = model_tfidf_lr.predict_proba(X_test)
test_pred2 = hf_predict_proba(X_test)
test_meta = np.hstack([test_pred1, test_pred2])

final_pred = meta_model.predict(test_meta)

print(classification_report(y_test, final_pred))
