import torch

n_classes = 2

train_csv_file = './data/en_train_data_SMM4H_2025_clean.csv'

batch_size = 8
num_epoches = 3

class ClassWeights:
    class_weights=torch.tensor([1.0, 1.0])

clw = ClassWeights()