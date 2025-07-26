from unsloth import FastLanguageModel
import re
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
from utils.utils import (fix_random_seed,
                         overload_dataset_by_instruction,
                         get_metrics)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str, default="unsloth-Qwen3-1.7B-unsloth-bnb-4bit", help='set open source model name')
#TODO: находить автоматически нужные данные. Пока что задаем путь к данным явно
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set path to root dir with data')
parser.add_argument('-n', '--num_generations', type=int, default=100, help='set number of generation samples')
args = parser.parse_args()

fix_random_seed()

'''
Данная функция парсинга может отличаться для различных моделей.
В частности может отличаться последний токен, поэтому распознование
регулярных выражений для разных backbone моделей может не работать.
Конкретно эта функция заточена под модель unsloth/mistral-7b-instruct-v0.3-bnb-4bit.
Для того, чтобы подогнать функцию под другую модель, возьмите обученную модель и
посмотрите на формат выхода 

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)
'''
def parse_output(output):
    # re_match = re.search(r'### Response:\n(.*?)<\|end▁of▁sentence\|>', output, re.DOTALL)
    re_match = re.search(r'### Response:\n(.*?)<\|im_end\|>', output, re.DOTALL)
    if re_match:
        response = re_match.group(1).strip()
        return response
    else:
        return ''

# Load saved model LoRa adapters
model_name = args.model_name

data_path = Path(args.data_path)
assert(data_path.exists())

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"{model_name.replace('/', '-')}",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True
)

# Make a prediction on test set
_, val = get_train_eval_dataset_pd()
public_set = val

FastLanguageModel.for_inference(model)

prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

public_set["instruction"] = "Classify this math problem into two topics: with Adverse Drug Events and without. Adverse Drug Events are negative medical side effects associated with a drug"
public_set.rename(columns = {"text": "input"}, inplace=True)

'''
Здесь мы собираем в одном датасете всю информацию:
- исходные примеры
- ответ модели
- ответ модели, который прошел через парсер
- текстовую метку
- числовую метку
'''
raw_outputs = []
for i in tqdm(range(len(public_set))):
  inputs = tokenizer(
  [
      prompt.format(
          public_set.iloc[0]["instruction"], 
          public_set.iloc[i]["input"], 
          "",
      )
  ], return_tensors = "pt", truncation = True, max_length = 2048).to("cuda")

  outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
  raw_outputs.append(tokenizer.batch_decode(outputs))
  
public_set["raw_outputs"] = [raw_output[0] for raw_output in raw_outputs]
public_set["parsed_outputs"] = public_set["raw_outputs"].apply(parse_output)

label_map = {
    0: "without Adverse Drug Events",
    1: "with Adverse Drug Events"
}

label2id = {v:k for k,v in label_map.items()}
public_set["predicted_label"] = public_set["parsed_outputs"].map(label2id)

#! Дамп датафрейма с классификацией для дебага, потом можно будет убрать или включать по флагу
if True:
    tmp_dataset_path = Path('./tmp')
    if not tmp_dataset_path.exists():
        os.mkdir(tmp_dataset_path.absolute())
        
    public_set.to_csv('./tmp/public_set_classification.csv', index=False)
    
true_labels = public_set['label'] 
preds = public_set["predicted_label"]

try:
    print(f"EVALUATE MODEL {model_name}")
    #TODO: need refactor
    cm, validation_accuracy, validation_precision, validation_recall, validation_f1_micro, validation_f1_macro = get_metrics(preds, true_labels)
    
    print(f"METRICS FOR THIS MODEL:\n")
    print(
        f"Accuracy: {validation_accuracy}\n",
        f"Precision: {validation_precision}\n",
        f"Recall: {validation_recall}\n",
        f"F1 micro: {validation_f1_micro}\n",
        f"F1 macro: {validation_f1_macro}\n"
    )
    
    plot_confusion_matrix(cm, classes=range(n_classes), model_name=model_name, save_file_path='./images')
    
    if output_file_path is not None:
        file_create = output_file_path.exists()
        
        with open(output_file_path.absolute(), 'a') as fd:
            if not file_create:
                fd.write("model,accuracy\n")
            fd.write(f"{model_name},{validation_f1_micro}\n")
except Exception as ex:
    print(f"ERROR during evaluating model {model_name}: {ex}")