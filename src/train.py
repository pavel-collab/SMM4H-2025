import argparse
from utils.constants import *
from utils.utils import (fix_random_seed,
                         get_device,
                         get_train_eval_dataset,
                         print_device_info,
                         compute_metrics)
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification,
                          Trainer, 
                          TrainingArguments)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str, default='bert-base-uncased', help='set backbone model name')
# parser.add_argument('--use_generation', action='store_true', help='if we using generation data for train')
args = parser.parse_args()

model_name = args.model_name

fix_random_seed()
device = get_device()

train_dataset, eval_dataset = get_train_eval_dataset(get_class_weight_flag=True)

texts = train_dataset['text']
labels = train_dataset['label']

print_device_info()

try:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.to(device)
    
    # Токенизация данных
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
            output_dir=f"./results/{model_name.replace('/', '-')}_results",
            eval_strategy="steps",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epoches,
            weight_decay=0.01,
            logging_dir=f"./logs/{model_name.replace('/', '-')}_logs",  
            save_steps=1000, # сохранение чекпоинтов модели каждые 1000 шагов# директория для логов TensorBoard
            logging_steps=100,
            save_total_limit=5, # Сохранять только последние 5 чекпоинтов
            fp16=True,
            gradient_accumulation_steps=2
        )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics
    )
    
    try:
        trainer.train()
    except Exception as ex:
        print(f"[ERROR] with training {model_name}: {ex}")
        
    tokenizer.save_pretrained(f"./results/{model_name.replace('/', '-')}_results/tokenizer")
        
except KeyboardInterrupt:
    print(f"[STOP] training with keyboard interrupt")