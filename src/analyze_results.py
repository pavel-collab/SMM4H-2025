import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

'''
В этом скрипте мы анализируем результаты классификации
моделей, записанные в файл evaluation_report.csv и строим сравнительные 
графики для разных метрик модели
'''

#TODO: добавить больше метрик классификации и графиков, добавить сохранение диаграм

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

def make_compared_diagram(df):
    # Создаем столбчатую диаграмму с группировкой
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='model_name', 
        y='label_1_f1', 
        #! Здесь может быть прикол. Мы в датафрейме храним маркер в числовом виде, а по идее он дожен подаваться в логическом виде.
        #! Не знаю, произойдет ли приведение типов под капотом.
        hue='use_generation',  
        data=df,
        palette={True: 'skyblue', False: 'salmon'}  # Цвета для наглядности
    )
    
    plt.title('Сравнение качества моделей: С использованием синтетических данных и без')
    plt.xlabel('Модель')
    plt.ylabel('F1-score')
    plt.legend(title='Маркер', labels=['С синтетическими данными', 'Без синтетических данных'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filepath', type=str, default='./evaluation_report.csv', help='set path to csv file aith results')
args = parser.parse_args()

filepath = Path(args.filepath)
assert(filepath.exists())

df = pd.read_csv(filepath.absolute())

make_simple_diagram(df)

make_compared_diagram(df)