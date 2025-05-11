```
./script/clean_data.sh
```

```
python3 ./src/data_preprocessing/check_data.py -d ./data/en_train_data_SMM4H_2025_clean.csv
```

```
mkdir images
python3 ./src/data_preprocessing/check_feature_visualize.py -d ./data/en_train_data_SMM4H_2025_clean.csv
```