import pandas as pd
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported, FastLanguageModel
from sklearn.model_selection import train_test_split

model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"

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

#TODO: need to refactor
# Load training data
label_map = {
    0: "without Adverse Drug Events",
    1: "with Adverse Drug Events"
}


train = pd.read_csv("../data/en_train_data_SMM4H_2025_clean.csv")
train, val = train_test_split(train, test_size=0.2, random_state=20)

train["instruction"] = "Classify this math problem into two topics: with Adverse Drug Events and without. Adverse Drug Events are negative medical side effects associated with a drug"
train["label"] = train["label"].map(label_map)
train = train.rename(columns={"label": "output", "text": "input"})
train.to_csv("train_updated.csv", index=False)

dataset = load_dataset("csv", data_files="train_updated.csv", split="train")

# Prepare data
prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

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

dataset = dataset.map(formatting_prompts_func, batched = True,)

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
        max_steps = 642,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = f"{model_name.replace('/', '-')}_outputs",
        report_to = "none"
    ),
)

# Start training
trainer_stats = trainer.train()

#! ATTENTION: here we're saving only LoRa adapters. Not original model itelf
# save model and tokenizer
model.save_pretrained(f"{model_name.replace('/', '-')}")
tokenizer.save_pretrained(f"{model_name.replace('/', '-')}")