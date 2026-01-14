#!/bin/bash

python3 ./src/data_preprocessing/clean_data.py -d ./data/train_data_SMM4H_2025_Task_1.csv
python3 ./src/data_preprocessing/clean_data.py -d ./data/train_data_SMM4H_2025_Task_1.csv -s
# python3 ./src/data_preprocessing/clean_data.py -d ./data/dev_data_SMM4H_2025_Task_1.csv -e