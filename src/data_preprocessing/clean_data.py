import pandas as pd
import argparse
from pathlib import Path

LANGUAGES = ['en', 'ru', 'fr', 'de']
TEMPLATE_TRAIN_DATA_FILE_NAME = 'train_data_SMM4H_2025_clean.csv'
TEMPLATE_EVAL_DATA_FILE_NAME = 'dev_data_SMM4H_2025_clean.csv'
TEMPLATE_SPLIT_DATA = 'splited_data_SMM4H_2025_clean.csv'

SPLIT_SAMPLES_DIR_NAME = "splited_samples"

def extract_language_df(raw_df, lang: str):
    lang_df = raw_df[raw_df['language'] == lang].reset_index()
    lang_df = lang_df.drop(columns=['index',
                                    'language'])
    return lang_df

def split_samples(lang_df):
    positive_df = lang_df[lang_df['label'] == 1].reset_index()
    negative_df = lang_df[lang_df['label'] == 0].reset_index()

    positive_df = positive_df.drop(columns=['index'])
    negative_df = negative_df.drop(columns=['index'])
    return positive_df, negative_df

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_path', help='path to raw training data')
    parser.add_argument(
        '-e',
        '--eval',
        help='indicate if cleaning dataset is eval',
        action='store_true',
        default=False)
    parser.add_argument('-s', '--split_samples', help='split samples to the positive and negative for each of language', action='store_true', default=False)

    args = parser.parse_args()

    if args.data_path is None or args.data_path == '':
        print('[err] there is no training data path; please set the path to training data and try again')
        return

    data_path = Path(args.data_path)

    if not data_path.exists():
        print(f"[err] {data_path.name} is not exists")
        return

    data_directory_path = data_path.parent

    raw_df = pd.read_csv(data_path.absolute())

    df = raw_df.drop(columns=['id',
                              'file_name',
                              'origin',
                              'type',
                              'split'])

    if args.eval:    
        for lang in LANGUAGES:
            lang_df = extract_language_df(df, lang)
            file_name = f"{data_directory_path}/{lang}_{TEMPLATE_EVAL_DATA_FILE_NAME}"
            lang_df.to_csv(file_name, index=False)
        
        df = df.drop(columns=['language'])
        df.to_csv(
            f"{data_directory_path}/{TEMPLATE_EVAL_DATA_FILE_NAME}",
            index=False)    
        return
    
    if args.split_samples:
        path_to_splited_data = Path(data_directory_path.absolute().name + "/" + SPLIT_SAMPLES_DIR_NAME)
        path_to_splited_data.mkdir(parents=True, exist_ok=True)

        for lang in LANGUAGES:
            lang_df = extract_language_df(df, lang)
            positive_df, negative_df = split_samples(lang_df)
            file_name_positive = f"{path_to_splited_data}/{lang}_positive_{TEMPLATE_SPLIT_DATA}"
            file_name_negative = f"{path_to_splited_data}/{lang}_negative_{TEMPLATE_SPLIT_DATA}"
            positive_df.to_csv(file_name_positive)
            negative_df.to_csv(file_name_negative)
        return 

    for lang in LANGUAGES:
        lang_df = extract_language_df(df, lang)
        file_name = f"{data_directory_path}/{lang}_{TEMPLATE_TRAIN_DATA_FILE_NAME}"
        lang_df.to_csv(file_name, index=False)

    df.drop(columns=['language'])
    df.to_csv(
        f"{data_directory_path}/{TEMPLATE_TRAIN_DATA_FILE_NAME}",
        index=False)


if __name__ == '__main__':
    main()
