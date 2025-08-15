## Предобработка данных

Очищаем данные от лишних столбцов, делим данные на выборки по языкам, выделяем положительные примеры для генерации.
```
./script/clean_data.sh
```

Выводим основную информацию о данных из обучающего датасета: количество примеров, средняя длина текстового примера,
наличие специальных текстовых форматов (эмоджи, латех, html) и др, наиболее частые токены.
```
python3 ./src/data_preprocessing/check_data.py -d ./data/en_train_data_SMM4H_2025_clean.csv
```

Вывод основной информации о тексте и визуализация этой информации в графиках.
```
mkdir images
python3 ./src/data_preprocessing/check_feature_visualize.py -d ./data/en_train_data_SMM4H_2025_clean.csv
```

## Обучение классификаторов

Запустит автоматическую тренировку нескольких классификаторов на базе открытых моделей (с малым числом параметров)
```
./scripts/run.sh
```

Запустит автоматическую валидацию моделей, обученных предыдущим скриптом
```
./scripts/evaluate_models.sh
```

## Обучение генератора

Для начала нам нужно составить специальный датасет в формате json для обучения
```
./scripts/make_json_datasets.sh
```

Обученим модель генерации через библиотеку unsloth. LoRa адаптеры обученной модели будут сохранены локально
```
python3 unsloth_generator_train.py -d ./data
```

Запускаем обученную модель на инференсе. Сгенерированные данные будут записаны в подкаталог каталога data
```
python3 unsloth_generator_inference.py -d ./data
```

Запуск генератора через vllm
```
python3 vllm_generation.py -n 100 -d ./data
```

Запуск генератора через llama_cpp (на данный момент показывает не очень
хорошие результаты + требуется предварительно скачать локально модель в формате .gguf)
```
python3 llama_cpp_generation.py -m ./saved_models/mistral-7b-instruct-v0.1.Q3_K_M.gguf -n 10 -d ./data/
```

Запуск модели для перевода примеров (указываем, с какого языка хотим перевести, на данный момент перевод всегда осуществляется на английский)
```
python3 translation.py -d ./data --language ru
```

Для того, чтобы перевести положительные примеры на целевой язык -- запустите скрипт
Вниимание! На данный момент работа этого bash-скрипта не протестирована, целевым языком является английский.
```
./srcipts/translate_samples.sh
```

Для того, чтобы сформировать новый комбинированный датасет
```
python3 ./src/generation/compile_generated_dataset.py --add_translated -add_generated --language en
```

## Текущее состояние и порядок действий

```
python3 ./src/generation/translation.py -d ./data --language ru
python3 ./src/generation/translation.py -d ./data --language fr
python3 ./src/generation/translation.py -d ./data --language de

python3 ./src/generation/compile_generated_dataset.py --add_translated --language en

python3 ./src/data_preprocessing/make_dataset.py -d ./data/new_datasets/new_dataset.csv

python3 ./src/generation/unsloth_generator_train.py -d ./data

python3 ./src/generation/unlsoth_generator_inference.py -d ./data -n 450
После обучения в каталоге results будут сохранены натренированные чекпоинты моделей, а в каталоге logs -- информация об обучении,
которую можно визуализировать через tensorboard.

Визуализация логов через tensorboard
```
tensorboard --logdir=./logs
```

Анализ результатов классификации
```
python3 ./src/analyze_results.py
```

## Обучение через биюлиотеку unsloth

Тренировка классификатора
```
python3 ./src/unsloth_classificator_train.py -d ./data/en_train_data_SMM4H_2025_clean.csv
```

Валидация обученного классификатора
```
python3 ./src/unsloth_classificator_inference.py -d ./data/en_train_data_SMM4H_2025_clean.csv
```