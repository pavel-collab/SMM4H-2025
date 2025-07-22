import pandas as pd
import os
import argparse
from pathlib import Path

SAVE_GENERATIONS_PATH_TEMPLATE = 'generations' # подкаталог в каталоге с данными, куда сохраняем нагенерированные данные
SAVE_TRANSLATED_DIR_TEMPLATE = 'translated'
SAVE_NEW_DATASET_DIR_TEMPLATE = 'new_datasets'

CHUNK_SIZE = 1000

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
    args = parser.parse_args()
    
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
        
    n_dataframes = 0
        
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
            
    print(f'Объединено {n_dataframes} файлов.')
    
if __name__ == '__main__':
    main()