from vllm import LLM, SamplingParams
from transformers import pipeline
import gc
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pipeline', action='store_true', help='use raw pipeline to classify generationed result')
parser.add_argument('-m', '--model_path', help='set a path to the model that we want to evaluate')
parser.add_argument('-n', '--n_samples', type=int, default=10, help='number of generated samples')
parser.add_argument('-o', '--output_path', type=str, default='./data/', help='path to directory where we will save output file with generated data')
args = parser.parse_args()

# Загружаем LLM
llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct',
        #   tokenizer_mode="auto",
        #   dtype="auto",
          gpu_memory_utilization=0.9,
          max_num_seqs=128,
          max_model_len=2048)
# llm = LLM(model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
#           tokenizer_mode="auto",
#           dtype="auto",
#           gpu_memory_utilization=0.9,
#           max_num_seqs=128,
#           max_model_len=2048)

target_label = args.target_label
n_samples = args.n_samples

# Промпт с уточнением стиля
prompt = f"""
Generate an example of a comment or tweet with Adverse Drug Events. Adverse Drug Events are negative medical side effects associated with a drug.
Don't use general phrases, give an example of tweet or comment.
Make a brief answer. Only text of the unswer without introduction phrases.
"""

sampling_params = SamplingParams(
            temperature=0.9, top_k=500, top_p=0.9, max_tokens=512, n=n_samples
        )

# Генерация
outputs = llm.generate(prompt, sampling_params)

accepted_samples_number = 0
for output in tqdm(outputs[0].outputs):
    #TODO полная строка генерации outputs[0].outputs[0] -- разобраться что значат эти индексы
    #TODO в генерациях остаются артефакты генерации в начале предложения, придумать, как убрать их
    generated = output.text.strip()

    # освободим память перед применением новой нейронки
    gc.collect()
    torch.cuda.empty_cache()

    model_file_path = Path(args.model_path)
    assert(model_file_path.exists())

    tokenizer_file_path = Path(f"{model_file_path.parent.absolute()}/tokenizer")
    assert(tokenizer_file_path.exists())

    model_name = model_file_path.parent.name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path.absolute())
    # Загрузка модели
    model = AutoModelForSequenceClassification.from_pretrained(model_file_path.absolute())
    
    # Токенизация данных
    inputs = tokenizer(generated, 
                        return_tensors='pt', 
                        padding=True, 
                        truncation=True, 
                        max_length=256, 
                        add_special_tokens = True)

    # Прогон текста через модель
    with torch.no_grad():  # Отключаем градиенты для оптимизации
        outputs = model(**inputs)

    # Извлечение предсказаний
    logits = outputs.logits
    result_lable = torch.argmax(logits, dim=-1)
    propabilities = F.softmax(logits, dim=1)
    score = propabilities[0][result_lable].detach().item()
    result = result_lable.detach().item()
        
    output_file_path = Path(args.output_path)
    output_file = Path(f"{output_file_path.absolute()}/{target_label.replace(' ', '-')}_generated_samples.csv")
    file_create = output_file.exists()

    if result == 1 and score > 0.7:
        # clean result before insert in file
        cleaned_generation = generated.replace(',', '').replace('\n', ' ')
        
        with open(output_file.absolute(), 'a') as fd:
            if not file_create:
                fd.write("Question,label\n")
            fd.write(f"{cleaned_generation},{result_lable.detach().item()}\n")
        accepted_samples_number += 1

gc.collect()
torch.cuda.empty_cache()
    
print("SUCCESSFUL GENERATION COMPLITION.")
print(f"NUMBER OF ACCEPTED SAMPLES [{accepted_samples_number}/{n_samples}]")