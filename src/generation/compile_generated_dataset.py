import pandas as pd
import os
import argparse
from pathlib import Path
from utils import ParsedFileName, LANGUAGES

SAVE_GENERATIONS_PATH_TEMPLATE = 'generations' # подкаталог в каталоге с данными, куда сохраняем нагенерированные данные
SAVE_TRANSLATED_DIR_TEMPLATE = 'translated'
SAVE_NEW_DATASET_DIR_TEMPLATE = 'new_datasets'

CHUNK_SIZE = 1000

'''
В данном подходе логика следующая:

Мы сначала делаем перевод всех положительных примеров со всех языков на целевой язык и объединяем датасеты.
Таким образом, мы получаем датасет на целевом языке, где у нас просто больше положительных примеров, но они 
все еще из исходного датасета (просто с других языков). 

Затем на этом датасете мы обучеем генератор. С помощью генератора мы генерируем новые положительные примеры
в достаточном количестве и объединяем теперь уже исходный датасет, переведенные примеры и сгенерированные примеры.

На этом новом полном датасете мы обучаем классификатор и смотрим на метрики качества.
'''

#TODO: это костыль из-за того, что исходные файлы с данными не хранятся ни в какой выделенной директории, поэтому тут нельзя их найти через listdir
#TODO: нужно переделать так, чтобы программа автоматически находила нужный файл
source_data_files = {
    "en": "en_train_data_SMM4H_2025_clean.csv",
    "ru": "ru_train_data_SMM4H_2025_clean.csv",
    "fr": "fr_train_data_SMM4H_2025_clean.csv",
    "de": "de_train_data_SMM4H_2025_clean.csv"
}

'''
Данный скрипт объединяет все файлы с нагенерированными данными в один новый датасет и записывает его в файл.
Заметим, что в новый файл запишутся только сгенерированные данные БЕЗ данных из изначального датасета.
Сделано это для того, чтобы на этапе обучения классификатора можно было выбрать -- использовать дополнительные
сгенерированные данные или нет.
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default='./data/', help='set path to root dir with data')
    parser.add_argument('--add_generated', action='store_true', help='if we want to add generated examples to dataset')
    parser.add_argument('--add_translated', action='store_true', help='if we want to add translated examples to dataset')
    parser.add_argument('--language', type=str, default="en", help='language of new dataset')
    args = parser.parse_args()
    
    language = args.language
    assert(language in LANGUAGES)
    
    if not args.add_generated and not args.add_translated:
        print("[ERROR] there are no addition data for new dataset")
        return

    root_data_path = Path(args.data_path)
    assert(root_data_path.exists())
    
    result_file_path = Path(f'{root_data_path.absolute()}/{SAVE_NEW_DATASET_DIR_TEMPLATE}/new_dataset.csv')
    if not result_file_path.parent.exists():
        os.mkdir(result_file_path.parent.absolute())
    
    generations_path = None
    translations_path = None
    
    if args.add_generated:
        generations_path = Path(f'{root_data_path.absolute()}/{SAVE_GENERATIONS_PATH_TEMPLATE}')

    if args.add_translated:
        translations_path = Path(f'{root_data_path.absolute()}/{SAVE_TRANSLATED_DIR_TEMPLATE}')
       
    # Потому что исходный датасет объединяется всегда 
    n_dataframes = 1
        
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #! Пока что здесь не учитываетс вариативность языков. Делаем все вручную
    #TODO: задача: чтобы скрипт был универсален для всех языков (сейчас парааметр language бесполезен -- исправить)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if generations_path is not None:
        generation_data_files = os.listdir(generations_path.absolute())
        
        for file_name in generation_data_files:
            file_path = Path(f'{generations_path.absolute()}/{file_name}')
            '''
            Чистаем из файла чанками; если файл большой, это
            поможет избежать переполнения памяти.
            '''
            for chunk in pd.read_csv(file_path.absolute(), chunksize=CHUNK_SIZE, lineterminator='\n'):
                chunk.to_csv(result_file_path, mode='a', header=not os.path.exists(result_file_path), index=False)
            n_dataframes += 1
            
    if translations_path is not None:
        translation_data_files = os.listdir(translations_path.absolute())
        
        for file_name in translation_data_files:
            file_path = Path(f'{translations_path.absolute()}/{file_name}')
            '''
            Чистаем из файла чанками; если файл большой, это
            поможет избежать переполнения памяти.
            '''
            for chunk in pd.read_csv(file_path.absolute(), chunksize=CHUNK_SIZE, lineterminator='\n'):
                chunk.to_csv(result_file_path, mode='a', header=not os.path.exists(result_file_path), index=False)
            n_dataframes += 1
                
    source_train_data_file = Path(f"{root_data_path.absolute()}/{source_data_files[language]}")
    for chunk in pd.read_csv(source_train_data_file.absolute(), chunksize=CHUNK_SIZE, lineterminator='\n'):
        chunk.to_csv(result_file_path, mode='a', header=not os.path.exists(result_file_path), index=False)
            
    print(f'Объединено {n_dataframes} файлов.')
    
if __name__ == '__main__':
    main()