import numpy as np # linear algebra
from numpy.lib.stride_tricks import as_strided
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

def create_x_dataframe():
    return pd.DataFrame(dtype=np.float64, columns=[
        "min", "max"])
        
def create_y_dataframe(time=None):
    if time is None:
        return pd.DataFrame(dtype=np.float64, columns=["time"])
    return pd.DataFrame(data={"time": [time]})
        
def calculate_statistical_features(data_chunk):
    return {
        "min": [data_chunk.acoustic_data.rolling(window=window_size).min().mean()],
        "max": [data_chunk.acoustic_data.rolling(window=window_size).max().mean()]
    }

def process_data():
    sample_size = int(150e3)
    window_size = 50
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
        
        features = calculate_statistical_features(chunk)
        if samples_read < train_row_n:
            x_train = x_train.append(pd.DataFrame(data=features))
            y_train = y_train.append(create_y_dataframe(chunk.time_to_failure.mean()))
        else:
            x_val = x_val.append(pd.DataFrame(data=features))
            y_val = y_val.append(create_y_dataframe(chunk.time_to_failure.mean()))
        
        samples_read += sample_size
    
    print(x_train)
    print(y_train)
    print(x_val)
    print(y_val)
    
    return x_train, y_train, x_val, y_val

x_train, y_train, x_val, y_val = process_data()