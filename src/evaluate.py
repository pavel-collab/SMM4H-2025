from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
import argparse
from pathlib import Path
from utils.constants import *
from utils.utils import (get_device, get_train_eval_dataset,
                         evaleate_model, plot_confusion_matrix,
                         dump_classification_metrics)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', help='set a path to the model that we want to evaluate')
parser.add_argument('-t', '--tokenizer_path', type=str, default=None, help='set path to saved tokenizer')
parser.add_argument('-o', '--output', help='set a path to the output filename, programm will write a model name and final accuracy')
args = parser.parse_args()

DUMP_METRICS_FILEPATH = 'evaluation_report.csv'

if args.output is not None and args.output != "":
    output_file_path = Path(args.output)
else: 
    output_file_path = None

model_file_path = Path(args.model_path)
assert(model_file_path.exists())

if args.tokenizer_path is None:
    tokenizer_file_path = Path(f"{model_file_path.parent.absolute()}/tokenizer")
else:
    tokenizer_file_path = Path(args.tokenizer_path)
assert(tokenizer_file_path.exists())

model_name = model_file_path.parent.name

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path.absolute())
# Загрузка модели
model = AutoModelForSequenceClassification.from_pretrained(model_file_path.absolute(), num_labels=n_classes)

# детектируем девайс
device = get_device()
model.to(device)

_, val_dataset = get_train_eval_dataset(use_generation=False,
                            get_class_weight_flag=False)

# Токенизация данных
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

baseline_trainer = Trainer(
    model=model,
    eval_dataset=tokenized_val_dataset,
)


print(f"EVALUATE MODEL {model_name}")
cm, validation_report, accuracy, micro_f1 = evaleate_model(model, baseline_trainer, tokenized_val_dataset, device)

metrics = {
    'accuracy': accuracy,
    'micro_f1': micro_f1,
    'classification_report': validation_report
}
dump_classification_metrics(model_name, metrics, csv_file=DUMP_METRICS_FILEPATH, use_generation=False)

# print("Metrics for current model:")
# print(f'Test accuracy: {accuracy:.4f}')
# print(validation_report)
# print(f'Test F1 micro: {micro_f1:.4f}')
# plot_confusion_matrix(cm, classes=range(n_classes), model_name=model_name, save_file_path='./images')
