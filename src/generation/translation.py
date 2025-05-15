from transformers import MarianMTModel, MarianTokenizer
import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', help='path to train data')
parser.add_argument('--language', type=str, default='en', help='set language')
parser.add_argument('-o', '--output_path', type=str, default='./data/', help='path to directory where we will save output file with generated data')
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

def get_model_and_tokenizer(src_lang: str):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_texts(texts, tokenizer, model, batch_size=8):
    translated = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs)
        translated_batch = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translated.extend(translated_batch)
    return translated

# Пример использования
if __name__ == "__main__":
    all_translations = []
    
    data_path = Path(args.data_path)
    assert(data_path.exists())
    
    df = pd.read_csv(data_path.absolute())
    
    texts = df[df['language'] == lang].to_list()
    texts_by_lang = {
        lang: texts
    }

    for lang_code, texts in texts_by_lang.items():
        print(f"Translating from {language_codes[lang_code]}...")
        tokenizer, model = get_model_and_tokenizer(lang_code)
        translated = translate_texts(texts, tokenizer, model)
        all_translations.extend(translated)

    output_file_path = Path(args.output_path)
    output_file = Path(f"{output_file_path.absolute()}/translated_samples_{lang}_en.csv")
    file_create = output_file.exists()

    for t in all_translations:
        cleaned_generation = t.replace(',', '').replace('\n', ' ')
        with open(output_file.absolute(), 'a') as fd:
            if not file_create:
                fd.write("text,label\n")
            fd.write(f"{cleaned_generation},1\n")