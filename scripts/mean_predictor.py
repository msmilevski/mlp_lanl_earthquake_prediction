import numpy as np
import os
import pandas as pd
import torch.nn as nn
import torch

def compute_mean():
    train_data_file = "data/only_train.csv"
    
    sample_size = 150000
    train_sample_n = 3334   # We have about 629 million rows in total, so dedicating 3334 * 150 000  = 500 100 000 for training
    
    chunks_processed = 0
    time_sum = 0
    for chunk in pd.read_csv(train_data_file, chunksize=sample_size):
        time_sum += chunk.time_to_failure.mean()

        chunks_processed += 1

        if chunks_processed % 20 == 0:
            print("{0} chunks processed, {1} remaining".format(chunks_processed, train_sample_n - chunks_processed))

    return time_sum / chunks_processed      # 5.594792923669247

def evaluate():
    val_data_file = "data/only_val.csv"
    
    sample_size = 150000
    val_sample_n = 859

    chunks_processed = 0
    targets = np.array([])
    for chunk in pd.read_csv(val_data_file, chunksize=sample_size):
        targets = np.append(targets, chunk.time_to_failure.tail(1).values)

        chunks_processed += 1

        if chunks_processed % 20 == 0:
            print("{0} chunks processed, {1} remaining".format(chunks_processed, val_sample_n - chunks_processed))

    loss = nn.L1Loss()
    inputs = np.zeros_like(targets) + 5.594792923669247
    loss = loss(input=torch.from_numpy(inputs), target=torch.from_numpy(targets))

    return loss.item()


# print(compute_mean())
print(evaluate())   # 3.208508934605948
    
