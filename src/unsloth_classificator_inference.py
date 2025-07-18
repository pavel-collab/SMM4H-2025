from unsloth import FastLanguageModel
import re
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_output(output):
    # re_match = re.search(r'### Response:\n(.*?)<\|end▁of▁sentence\|>', output, re.DOTALL)
    re_match = re.search(r'### Response:\n(.*?)<\|im_end\|>', output, re.DOTALL)
    if re_match:
        response = re_match.group(1).strip()
        return response
    else:
        return ''

# Load saved model LoRa adapters
model_name = "unsloth-Qwen3-1.7B-unsloth-bnb-4bit"

max_seq_length = 2048
dtype = None # None for auto detection.
load_in_4bit = True # 4bit quantization to reduce memory usage. 

# if False:
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"{model_name.replace('/', '-')}",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True
)

#TODO: need to refactor
# Make a prediction on test set

train = pd.read_csv("./data/en_train_data_SMM4H_2025_clean.csv")
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

public_set["instruction"] = "Classify this math problem into two topics: with Adverse Drug Events and without. Adverse Drug Events are negative medical side effects associated with a drug"
public_set.rename(columns = {"text": "input"}, inplace=True)

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
  
public_set["parsed_outputs"] = public_set["raw_outputs"].apply(parse_output)

label_map = {
    0: "without Adverse Drug Events",
    1: "with Adverse Drug Events"
}

label2id = {v:k for k,v in label_map.items()}

public_set["predicted_label"] = public_set["parsed_outputs"].map(label2id)

