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
    
def get_train_eval_dataset(use_generation=False, get_class_weight_flag=False):
    '''
    Можно было бы сначала формировать датасет, а потом только делить его на 
    train и test. Но тут задумка в том, что в части для валидации нет сгенерированных данных.
    '''
    df = pd.read_csv(train_csv_file)
    
    # df['text'] = df['text'].apply(preprocess_text)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=20)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # if use_generation:
    #     generated_df = pd.read_csv(generated_csv_file)
    #     generated_df = generated_df.rename(columns={'Question': 'text'})

    #     train_df = pd.concat([train_df, generated_df], ignore_index=True)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # if get_class_weight_flag:
    #     clw.class_weights = get_class_weights(train_df)
    
    return train_dataset, val_dataset


def print_device_info():
    print(f"[DEBUG] Torch sees ", torch.cuda.device_count(), 'GPU(s)')
    print(f"[DEBUG] Accelerate is using device: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    print()
    

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device