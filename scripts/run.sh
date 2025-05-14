#!/bin/bash

#  backbode models
models=("bert-base-uncased" "distilbert-base-uncased" "roberta-base" "albert-base-v2" "xlnet-base-cased" "google/electra-base-discriminator" "facebook/bart-base" "microsoft/deberta-base",)

# Проход по массиву с помощью цикла
for model in "${models[@]}"; do
    python3 ./src/train.py -m $model
done