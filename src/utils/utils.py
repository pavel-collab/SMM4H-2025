from utils.constants import *
import numpy as np
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             confusion_matrix, 
                             classification_report)
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    validation_accuracy = accuracy_score(predictions, labels)
    validation_precision = precision_score(predictions, labels)
    validation_recall = recall_score(predictions, labels)
    validation_f1_micro = f1_score(predictions, labels, average='micro')
    validation_f1_macro = f1_score(predictions, labels, average='macro')

    return {
        'accuracy': validation_accuracy,
        'precision': validation_precision,
        'recall': validation_recall,
        'f1_micro': validation_f1_micro,
        'f1_macro': validation_f1_macro
    }
    
def fix_random_seed(seed=20):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
def get_train_eval_dataset(get_class_weight_flag=False):
    '''
    Можно было бы сначала формировать датасет, а потом только делить его на 
    train и test. Но тут задумка в том, что в части для валидации нет сгенерированных данных.
    '''
    df = pd.read_csv(train_csv_file)
    
    # df['text'] = df['text'].apply(preprocess_text)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=20)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    if get_class_weight_flag:
        clw.class_weights = get_class_weights(train_df)
    
    return train_dataset, val_dataset


def print_device_info():
    print(f"[DEBUG] Torch sees ", torch.cuda.device_count(), 'GPU(s)')
    print(f"[DEBUG] Accelerate is using device: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    print()
    

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def evaleate_model(model, trainer, tokenized_val_dataset, device):
    model.to(device)
    model.eval()
    predictions = trainer.predict(tokenized_val_dataset)
    
    #! in some models predictions.predictions is a complex tupple, not a numpy array 
    if isinstance(predictions.predictions, tuple):
        target_predictions = predictions.predictions[0]
    else:
        target_predictions = predictions.predictions
    
    preds = np.argmax(target_predictions, axis=-1)
    true_lables = tokenized_val_dataset['label']
    cm = confusion_matrix(true_lables, preds)
    report = classification_report(true_lables, preds)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    # Вычисление взвешенной F1-меры для текущей модели
    micro_f1 = f1_score(true_lables, preds, average='micro')
    return cm, report, accuracy, micro_f1

def plot_confusion_matrix(cm, classes, model_name=None, save_file_path=None):
    with plt.style.context('default'):  
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        if model_name:
            assert save_file_path is not None
            plt.title(f"Confusion Matrix for {model_name}")
        else:
            plt.title("Confusion Matrix")
        
        if save_file_path is None:
            plt.show()
        else:
            # Verify that model_name exists before saving
            assert model_name, "model_name must be provided when save_file_path is not None"
            plt.savefig(f"{save_file_path}/confusion_matrix_{model_name}.jpg")
            return f"{save_file_path}/confusion_matrix_{model_name}.jpg"
        
def get_class_weights(train_df):
    # Подсчитать количество изображений в каждом классе для обучающего набора данных
    train_class_counts = np.zeros(n_classes)
    for idx, row in train_df.iterrows():
        label = row['label']
        train_class_counts[label] += 1
        
    # посчитаем веса для каждого класса
    class_weights = (sum(train_class_counts.tolist()) / (n_classes * train_class_counts)).tolist()
    class_weights = torch.tensor(class_weights)
    return class_weights