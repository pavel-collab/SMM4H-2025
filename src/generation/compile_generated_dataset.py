import pandas as pd
import os
import argparse
from pathlib import Path
from utils import LANGUAGES
import yaml

#! ATTENTION: depends on file position in project tree
root_dir = Path(__file__).resolve().parent.parent

SAVE_NEW_DATASET_DIR_TEMPLATE = 'compiled_datasets'

CHUNK_SIZE = 1000

def read_yaml_file(file_path):
    """
    Читает данные из YAML файла и возвращает их в виде словаря.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Ошибка при парсинге YAML: {e}")
        return None

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

'''
Все параметры прописываются в конфигурационном файле config.yaml. 
Структура файла следующая:

params:
  lang: en
  files:
    - "file_1"
    - "file_2"
    - "file_3"
    - "file_4"

где lang -- язык датасета, с которым ведется работа,
а files -- список файлов, которые необходимо обхединить. 
Необходимо указать абсолютный путь к файлам!
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-с', '--config', type=str, default='./config.yaml', help='set path to config with file to compilation')
    args = parser.parse_args()

    config_path = Path(args.config)
    assert(config_path.exists())
    config_data = read_yaml_file(config_path.absolute())
    
    language = config_data.get('lang')
    assert(language in LANGUAGES)
    
    files_list = config_data.get('files')
    if files_list is None or len(files_list) == 0:
        print('[ERROR] there are no files for compilation')
        return
    
    #! Есть идея перегрузить название файла, напрмер, временной меткой. Вопрос, насколько это будет полезно и актуально
    result_file_path = Path(f'{root_dir}/data/{SAVE_NEW_DATASET_DIR_TEMPLATE}/compiled_dataset_{language}.csv')
    if not result_file_path.parent.exists():
        os.mkdir(result_file_path.parent.absolute())
            
    # Потому что исходный датасет объединяется всегда 
    n_dataframes = 1

    for file in files_list:
        file_path = Path(file)
        if not file_path.exists():
            print(f"[ERROR] file {file_path.absolute()} is not exist")
            continue
            
        for chunk in pd.read_csv(file_path.absolute(), chunksize=CHUNK_SIZE, lineterminator='\n'):
            chunk.to_csv(result_file_path, mode='a', header=not os.path.exists(result_file_path), index=False)
        n_dataframes += 1
        
    # Прибавляем к объедененному датасету данные исходного датасета            
    source_train_data_file = Path(f"{root_dir.absolute()}/data/{source_data_files[language]}")
    assert(source_train_data_file.exists())
    for chunk in pd.read_csv(source_train_data_file.absolute(), chunksize=CHUNK_SIZE, lineterminator='\n'):
        chunk.to_csv(result_file_path, mode='a', header=not os.path.exists(result_file_path), index=False)
            
    print(f'Объединено {n_dataframes} файлов.')
    
if __name__ == '__main__':
    main()