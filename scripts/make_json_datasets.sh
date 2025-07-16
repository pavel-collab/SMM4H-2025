#!/bin/bash

languages=("ru" "en" "fr" "de")

for lang in "${languages[@]}"; do
    python3 ./src/data_preprocessing/make_dataset.py -d ./data/ --language $lang
done