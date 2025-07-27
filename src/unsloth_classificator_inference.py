from unsloth import FastLanguageModel
import re
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
from utils.utils import (fix_random_seed,
                         get_metrics,
                         get_train_eval_dataset_pd,
                         dump_classification_metrics)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str, default="unsloth-mistral-7b-instruct-v0.3-bnb-4bit", help='set open source model name')
#TODO: находить автоматически нужные данные. Пока что задаем путь к данным явно
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set path to root dir with data')
args = parser.parse_args()

DUMP_METRICS_FILEPATH = 'evaluation_report.csv'

fix_random_seed()

#TODO: it can be moved to utils
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
    re_match = re.search(r'### Response:\n(.*?)<\/s>', output, re.DOTALL)
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
    #TODO: final_checkpoint это костыль -- исправить
    model_name = f"./results/{model_name.replace('/', '-')}_results",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True
)

#TODO: it can be made as a function call
# Make a prediction on test set
train = pd.read_csv(data_path.absolute())
_, val = train_test_split(train, test_size=0.2, random_state=20)
public_set = val

FastLanguageModel.for_inference(model)
prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

public_set["instruction"] = "Classify this example into two topics: with Adverse Drug Events and without Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug"
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
    
true_labels = public_set['label'] 
preds = public_set["predicted_label"]
 
print(f"EVALUATE MODEL {model_name}")
cm, validation_report, accuracy, micro_f1 = get_metrics(preds, true_labels)

metrics = {
    'accuracy': accuracy,
    'micro_f1': micro_f1,
    'classification_report': validation_report
}
dump_classification_metrics(model_name, metrics, csv_file=DUMP_METRICS_FILEPATH, use_generation=False)