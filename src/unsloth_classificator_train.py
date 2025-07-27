'''
Важно, чтобы импорт библиотеки unsloth происходил до импорта библиотек 
trl, transformers и peft, иначе можно столкнуться с бредовыми и
неожиданными ошибками.
'''
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import argparse
from pathlib import Path
import os
from utils.utils import (fix_random_seed,
                         get_train_eval_dataset_pd,
                         print_device_info)

parser = argparse.ArgumentParser()
#! Пока что указываем путь к файлу с данными для обучения; чуть позже, когда будет понятна структура данных в каталоге -- переделаем
#TODO: добавить язык датасета и автоматически искать нужный датасет по корневому пути + язык
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set path to file with training data')
parser.add_argument('-m', '--model_name', type=str, default='unsloth/mistral-7b-instruct-v0.3-bnb-4bit', help='set open source model name')
args = parser.parse_args()

model_name = args.model_name

data_path = Path(args.data_path)
assert(data_path.exists())

fix_random_seed()

# Импортируем модель и токенизатор
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Add LoRa adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
print_device_info()

tmp_dataset_path = Path('./tmp')
if not tmp_dataset_path.exists():
    os.mkdir(tmp_dataset_path.absolute())
    
#TODO: it can be made as a function call
# construct a dataset with training prompt for training
label_map = {
    0: "without Adverse Drug Events",
    1: "with Adverse Drug Events"
}

train = pd.read_csv(data_path.absolute())
train, _ = train_test_split(train, test_size=0.2, random_state=20)

train["instruction"] = "Classify this example tweet into two topics: with Adverse Drug Events and without. Adverse Drug Events are negative medical side effects associated with a drug"
train["label"] = train["label"].map(label_map)

train = train.rename(columns={"label": "output", "text": "input"})
train.to_csv("./tmp/train_updated.csv", index=False)
    
#TODO: перегрузить название файла языком датасета (en, ru, ...)
dataset = load_dataset("csv", data_files=f"{tmp_dataset_path.absolute()}/train_updated.csv", split="train")

# Prepare training prompt
prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# data formating before training
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

dataset = dataset.map(formatting_prompts_func, batched=True,)

# Setup trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 300, #642, #TODO: chenge llearning steps number
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = f"./results/{model_name.replace('/', '-')}_checkpoints",
        #TODO: почему здесь не сохраняются логи обучения
        #! У нас выставлен параметр max_steps, а значит, есть вероятность, что обучение не имеет даже одной эпохи.
        #! между тем, логирование выставлено по эпохам, поэтому, вероятнее всего, не создается лог
        logging_dir=f"./logs/{model_name.replace('/', '-')}_logs", 
        report_to = "none"
    ),
)

# Start training
trainer_stats = trainer.train()

#! ATTENTION: here we're saving only LoRa adapters. Not original model itelf
# save model and tokenizer
model.save_pretrained(f"./results/{model_name.replace('/', '-')}_results")
tokenizer.save_pretrained(f"./results/{model_name.replace('/', '-')}_results")