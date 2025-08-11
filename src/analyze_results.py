import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import os

'''
В этом скрипте мы анализируем результаты классификации
моделей, записанные в файл evaluation_report.csv и строим сравнительные 
графики для разных метрик модели
'''

image_save_path = Path('./images')

#! Поскольку в общем случае в датафрейме у нас есть метки моделей (использовалась ли генерация или нет)
#! Нужно заранее отфильтровать нужную часть датафрейма
def make_simple_diagram(df):
    # Создаем столбчатую диаграмму
    plt.figure(figsize=(10, 6))  # Устанавливаем размер фигуры
    ax = sns.barplot(x='model_name', y='label_1_f1', data=df)

    # Настраиваем подписи столбцов
    plt.xticks(rotation=45, ha="right")  # Поворачиваем подписи на 45 градусов и выравниваем по правому краю

    # Добавляем подписи осей и заголовок
    plt.xlabel("Модель")
    plt.ylabel("F1-score")
    plt.title("Диаграмма точности моделей")

    # Показываем диаграмму
    plt.tight_layout()  # Обеспечивает, что все элементы диаграммы помещаются в область отображения
    plt.show()    

def make_compared_diagram_f1(df, save_image=False):
    # Создаем столбчатую диаграмму с группировкой
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='model_name', 
        y='label_1_f1', 
        hue='use_generation',  
        data=df,
        palette={True: 'skyblue', False: 'salmon'}  # Цвета для наглядности
    )

    plt.xticks(rotation=45, ha="right")
    
    plt.title('Сравнение качества моделей: С использованием синтетических данных и без')
    plt.xlabel('Модель')
    plt.ylabel('F1-score')
    plt.legend(title='Маркер', labels=['С синтетическими данными', 'Без синтетических данных'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if not save_image:
        plt.show()
    else:
        assert(image_save_path.exists())
        plt.savefig(f'{image_save_path}/compare_model_barplot_f1.png')

#! Здесь используется precision именно для метки 1
def make_compared_diagram_precision(df, save_image=False):
    # Создаем столбчатую диаграмму с группировкой
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='model_name',
        y='label_1_precision',
        hue='use_generation',
        data=df,
        palette={True: 'skyblue', False: 'salmon'}  # Цвета для наглядности
    )

    plt.xticks(rotation=45, ha="right")
    
    plt.title('Сравнение качества моделей: С использованием синтетических данных и без')
    plt.xlabel('Модель')
    plt.ylabel('Precision')
    plt.legend(title='Маркер', labels=['С синтетическими данными', 'Без синтетических данных'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if not save_image:
        plt.show()
    else:
        assert(image_save_path.exists())
        plt.savefig(f'{image_save_path}/compare_model_barplot_precision.png')

#! Здесь используется precision именно для метки 1
def make_compared_diagram_recall(df, save_image=False):
    # Создаем столбчатую диаграмму с группировкой
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='model_name',
        y='label_1_recall',
        hue='use_generation',
        data=df,
        palette={True: 'skyblue', False: 'salmon'}  # Цвета для наглядности
    )

    plt.xticks(rotation=45, ha="right")

    plt.title('Сравнение качества моделей: С использованием синтетических данных и без')
    plt.xlabel('Модель')
    plt.ylabel('Recall')
    plt.legend(title='Маркер', labels=['С синтетическими данными', 'Без синтетических данных'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if not save_image:
        plt.show()
    else:
        assert(image_save_path.exists())
        plt.savefig(f'{image_save_path}/compare_model_barplot_recall.png')

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filepath', type=str, default='./evaluation_report.csv', help='set path to csv file aith results')
parser.add_argument('-s', '--save', action='store_true', help='if we need to save images')
args = parser.parse_args()

filepath = Path(args.filepath)
assert(filepath.exists())

df = pd.read_csv(filepath.absolute())

if args.save:
    if not image_save_path.exists():
        os.mkdir(image_save_path.absolute())

# Строим графики для моделей без синтетики
make_simple_diagram(df[df['use_generation'] == 0])

make_compared_diagram_f1(df, save_image=args.save)
make_compared_diagram_precision(df, save_image=args.save)
make_compared_diagram_recall(df, save_image=args.save)
