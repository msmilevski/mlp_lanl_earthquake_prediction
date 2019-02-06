import numpy as np # linear algebra
from numpy.lib.stride_tricks import as_strided
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

def create_x_dataframe():
    return pd.DataFrame(dtype=np.float64, columns=[
        "min", "max", "mean", "std", "q25", "q75", "iqr", "q01", "q99",
        "min_50", "max_50", "mean_50", "std_50", "q25_50", "q75_50", "iqr_50", "q01_50", "q99_50",
        "min_100", "max_100", "mean_100", "std_100", "q25_100", "q75_100", "iqr_100", "q01_100", "q99_100",
        "min_1000", "max_1000", "mean_1000", "std_1000", "q25_1000", "q75_1000", "iqr_1000", "q01_1000", "q99_1000"
    ])
        
def create_y_dataframe(times=None):
    if times is None:
        return pd.DataFrame(dtype=np.float64, columns=["time"])
    return pd.DataFrame(data={"time": times})
        
def calculate_statistical_features(data_chunk, window_size=None):
    if window_size == None:
        Q75 = data_chunk.acoustic_data.quantile(0.75)
        Q25 = data_chunk.acoustic_data.quantile(0.25)
        return {
            "min": [data_chunk.acoustic_data.min()],
            "max": [data_chunk.acoustic_data.max()],
            "mean": [data_chunk.acoustic_data.mean()],
            "std": [data_chunk.acoustic_data.std()],
            "q25": [Q25],
            "q75": [Q75],
            "q01": [data_chunk.acoustic_data.quantile(0.01)],
            "q99": [data_chunk.acoustic_data.quantile(0.99)],
            "iqr": [Q75 - Q25]
        }
    else:
        Q75 = data_chunk.acoustic_data.rolling(window=window_size).quantile(0.75).mean()
        Q25 = data_chunk.acoustic_data.rolling(window=window_size).quantile(0.25).mean()
        return {
            "min_" + str(window_size): [data_chunk.acoustic_data.rolling(window=window_size).min().mean()],
            "max_" + str(window_size): [data_chunk.acoustic_data.rolling(window=window_size).max().mean()],
            "mean_" + str(window_size): [data_chunk.acoustic_data.rolling(window=window_size).mean().mean()],
            "std_" + str(window_size): [data_chunk.acoustic_data.rolling(window=window_size).std().mean()],
            "q25_" + str(window_size): [Q25],
            "q75_" + str(window_size): [Q75],
            "q01_" + str(window_size): [data_chunk.acoustic_data.rolling(window=window_size).quantile(0.01).mean()],
            "q99_" + str(window_size): [data_chunk.acoustic_data.rolling(window=window_size).quantile(0.99).mean()],
            "iqr_" + str(window_size): [Q75 - Q25]
        }
        

def process_data():
    sample_size = int(150e3)
    window_sizes = [None, 50, 100, 1000]
    train_sample_n = 3334   # We have about 629 million rows in total, so dedicating 3334 * 150 000  = 500 100 000 for training
    train_row_n = sample_size * train_sample_n
        
    samples_read = 0
    x_train = create_x_dataframe()
    x_val = create_x_dataframe()
    y_train = create_y_dataframe()
    y_val = create_y_dataframe()
    
    for chunk in pd.read_csv("../input/train.csv", chunksize=sample_size):
        if chunk.shape[0] < sample_size:        # throwing out last remaining rows
            break
        features = {}
        for window_size in window_sizes:
            features.update(calculate_statistical_features(chunk, window_size))
            
        if samples_read < train_row_n:
            x_train = x_train.append(pd.DataFrame(data=features))
            y_train = y_train.append(create_y_dataframe(chunk.time_to_failure.tail(1).values))
        else:
            x_val = x_val.append(pd.DataFrame(data=features))
            y_val = y_val.append(create_y_dataframe(chunk.time_to_failure.tail(1).values))
        
        samples_read += sample_size
    
        print(x_train)
        print(y_train)
        print(x_val)
        print(y_val)
    


    return x_train, y_train, x_val, y_val

x_train, y_train, x_val, y_val = process_data()

