from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import json
import datasets
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from pathlib import Path
from utils import debug_print, FileInfo, LANGUAGES
import argparse
import os

'''
В этом скрипте мы берем open source модель и дообучаем ее через механизм LoRa
и библиотеку unsloth. В качестве обучающего датасета мы берем известные примеры
твитов (из первичного датасета или полученные через перевод или аугментацию).
После обучения адаптеры LoRa сохраняются локально.
'''

# путь, по которому сохраняем адаптеры модели
SAVE_MODEL_PATH = './saved_models/'

def main():
    parser = argparse.ArgumentParser()
    # в качестве data_path указываем путь к корневой директории с данными
    parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set path to root dir with data')
    parser.add_argument('--language', type=str, default='en', help='set language of raw positive dataset')
    parser.add_argument('-m', '--model_name', type=str, default='unsloth/gemma-3-4b-it', help='set open source model name')
    #TODO: make a map from model to chat_template and check availability
    parser.add_argument('--chat_template', type=str, default='gemma-3', help='set a chat template for model')
    args = parser.parse_args()

    lang = args.language
    assert(lang in LANGUAGES)

    # находим файл с закотовленным json датасетом см скрипт make_dataset.py
    root_data_dir_path = Path(args.data_path)
    assert(root_data_dir_path.exists())
    json_data_path = Path(f'{root_data_dir_path.absolute()}/json_datasets/')
    assert(json_data_path.exists())

    json_data_files = os.listdir(json_data_path.absolute())
    if len(json_data_files) == 0:
        raise Exception(f'there are no files in {json_data_path.absolute()}')

    #! Временно для тестирования
    #TODO: исправить
    target_file_info = FileInfo(f'{json_data_path.absolute()}/{json_data_files[0]}')
    # for filename in json_data_files:
    #     file_info = FileInfo(f'{json_data_path.absolute()}/{filename}')
    #     if file_info.lang == lang:
    #         target_file_info = file_info
    #         break
        
    assert(target_file_info is not None)
    assert(target_file_info.file_extension == '.json')

    model_name = args.model_name

    '''
    We now have to apply the chat template onto the conversations, and save it to text. 
    We remove the <bos> token using removeprefix('<bos>') since we're finetuning. 
    The Processor will add this token before training and the model expects only one.
    '''
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
        return { "text" : texts, }

    debug_print(f'Import model {model_name}')

    # import model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048, # Choose any for long context!
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
    )

    debug_print('Get PEFT model')

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # SHould leave on always!

        r = 8,           # Larger = higher accuracy, but might overfit
        lora_alpha = 8,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
    )

    # import chat template for learning model in few-shot way
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = args.chat_template,
    )

    debug_print('Import dataset')

    # import prepared json dataset made with make_dataset.py
    with open(target_file_info.filepath.absolute(), 'r') as json_file:
        data = json.load(json_file)
        dataset = datasets.Dataset.from_list(data)

    dataset = dataset.map(formatting_prompts_func, batched = True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 2000,
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use this for WandB etc
        ),
    )

    '''
    We also use Unsloth's train_on_completions method to only train on the assistant outputs and ignore the loss on the user's inputs. 
    This helps increase accuracy of finetunes!
    '''
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )

    #TODO: make a utilities for compuse available host resources
    '''
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    #----------------------------------------------------------------------------------------

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    '''

    debug_print('Start to train')

    trainer_stats = trainer.train()

    debug_print('Save model LoRa adapters')

    '''
    Сохраняем модель локально. Заметим, что в данном случае
    сохраняются только LoRa адаптеры, а не вся модель целиком.
    '''
    # replace / to - in model_name because we don't want create unesessary directories
    # also we don't use model_name variable itself because method replace changes the object
    model.save_pretrained(f"{SAVE_MODEL_PATH}/{args.model_name.replace('/', '-')}-lora-adapters")  # Local saving
    tokenizer.save_pretrained(f"{SAVE_MODEL_PATH}/{args.model_name.replace('/', '-')}-lora-adapters")

    #TODO: save in different formats: gguf, vllm
    
if __name__ == '__main__':
    main()