from transformers import pipeline
import pandas as pd
import argparse
from pathlib import Path
from utils import ParsedFileName, LANGUAGES
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', help='path to root directory with data')
parser.add_argument('--language', type=str, default='ru', help='set language of raw positive dataset')
args = parser.parse_args()

SAVE_AUGMENTED_TEMPLATE = 'aurmented'

def main():

    data_path = Path(args.data_path)
    assert(data_path.exists())
    
    lang = args.language
    assert(lang in LANGUAGES)

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

    paraphraser = pipeline("text2text-generation", 
                        model="ramsrigouthamg/t5_paraphraser", 
                        tokenizer="t5-base")

    prompt = "paraphrase: {text} </s>"

    df = pd.read_csv(target_file_info.filepath.absolute())

    accepted_sample_num = 0
    for _, row in df.iterrows():
        outputs = paraphraser(prompt.format(text = row['text']), 
                            max_length=128, 
                            num_return_sequences=1, # number of perefrase generations 
                            do_sample=True,
                            temperature=1.1,
                            top_k=500,
                            top_p=0.95)
        
        output_file_path = Path(f'{data_path.absolute()}/{SAVE_AUGMENTED_TEMPLATE}/{lang}_augmented_samples.csv')
        file_create = output_file_path.exists()
        
        if not output_file_path.parent.exists():
            os.mkdir(output_file_path.parent.absolute())

        for out in outputs:
            generated = out['generated_text']
            
            # clean result before insert in file
            cleaned_generation = generated.replace(',', '').replace('\n', ' ')
            
            with open(output_file_path.absolute(), 'a') as fd:
                if not file_create:
                    fd.write("text,label\n")
                fd.write(f"{cleaned_generation},1\n")

        accepted_sample_num += 1
        if accepted_sample_num == args.n_samples:
            break
        
if __name__ == '__main__':
    main()