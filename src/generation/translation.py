from transformers import MarianMTModel, MarianTokenizer
import argparse
import pandas as pd
from pathlib import Path
import os
from utils import ParsedFileName
from tqdm import tqdm

'''
В этому скрипте мы используем имеющиеся данные на разных языках для перевода их на другие языки и
получения дополнительных данных. При запуске указывается путь к корневому каталогу с данными
(навигация внутри этого каталога прописана в самой программе), а также язык, с которого будет делаться перевод.

Замечание: на данный момент переводчик транслирует ТОЛЬКО НА АНГЛИЙСКИЙ ЯЗЫК
'''

SAVE_TRANSLATED_DIR_TEMPLATE = 'translated'

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set path to root dir with data')
parser.add_argument('--language', type=str, default='ru', help='set language of raw positive dataset')
args = parser.parse_args()

# Карта языков: код языка Hugging Face MarianMT и его человекочитаемое название
language_codes = {
    "ru": "Russian",
    "fr": "French",
    "es": "Spanish",
    "de": "German"
}

lang = args.language
assert(lang in language_codes.keys())

# Находим в корневом каталоге файл с положительными примерами на языке, с которого будем делать перевод
root_data_dir_path = Path(args.data_path)
assert(root_data_dir_path.exists())
splited_data_path = Path(f'{root_data_dir_path.absolute()}/splited_samples/')
assert(splited_data_path.exists())

splited_data_files = os.listdir(splited_data_path.absolute())
if len(splited_data_files) == 0:
    raise Exception(f'there are no files in {splited_data_path.absolute()}')

target_file_info = None
for filename in splited_data_files:
    file_info = ParsedFileName(f'{splited_data_path.absolute()}/{filename}')
    if file_info.lang == lang and file_info.positive:
        target_file_info = file_info
        break
    
assert(target_file_info is not None)
assert(target_file_info.file_extension == '.csv')

def get_model_and_tokenizer(src_lang: str):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-en' # перевод на английский язык
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_texts(texts, tokenizer, model, batch_size=8):
    translated = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs)
        translated_batch = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translated.extend(translated_batch)
    return translated

if __name__ == "__main__":
    all_translations = []
    
    df = pd.read_csv(target_file_info.filepath.absolute())
    
    texts = df['text'].tolist()
    texts_by_lang = {
        lang: texts
    }

    for lang_code, texts in texts_by_lang.items():
        print(f"Translating from {language_codes[lang_code]}...")
        tokenizer, model = get_model_and_tokenizer(lang_code)
        translated = translate_texts(texts, tokenizer, model)
        all_translations.extend(translated)

    output_file_path = Path(f'{target_file_info.data_root_dir}/{SAVE_TRANSLATED_DIR_TEMPLATE}/translated_samples_{lang}_en.csv')
    file_create = output_file_path.exists()
    
    if not output_file_path.parent.exists():
        os.mkdir(output_file_path.parent.absolute())

    # написано так, чтобы в один csv файл можно было записать переводы текстов с рахных языков
    with open(output_file_path.absolute(), 'a') as fd:
        if not file_create:
            fd.write("text,label\n")
        for t in all_translations:
            cleaned_generation = t.replace(',', '').replace('\n', ' ')
            fd.write(f"{cleaned_generation},1\n")
