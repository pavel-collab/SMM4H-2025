#!/bin/bash

# Пока что целевой язык -- английский
languages=("ru" "de" "fr")

# Проход по массиву с помощью цикла
for lang in "${languages[@]}"; do
    python3 ./src/generation/translation.py -d ./data --language $lang
done