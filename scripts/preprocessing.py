import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

def feature_name_for_window(feature, window):
    if window == None:
        return feature
    return "{0}_{1}".format(feature, window)

def feature_names_for_windows(feature, windows):
    return list(map(lambda window: feature_name_for_window(feature, window), windows))

def flatten(l):
    return [item for sublist in l for item in sublist]

def get_columns(windowed_features, window_sizes, other_features):
    return flatten(
        list(
            map(
                lambda feature: feature_names_for_windows(feature, window_sizes), windowed_features)
            )
        ) + other_features

def create_x_dataframe(columns):
    return pd.DataFrame(dtype=np.float64, columns=columns)
        
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
            "var": [data_chunk.acoustic_data.var()],
            "q25": [Q25],
            "q75": [Q75],
            "q01": [data_chunk.acoustic_data.quantile(0.01)],
            "q99": [data_chunk.acoustic_data.quantile(0.99)],
            "iqr": [Q75 - Q25]
        }
    else:
        windows = data_chunk.acoustic_data.rolling(window=window_size)
        Q75 = windows.quantile(0.75).mean()
        Q25 = windows.quantile(0.25).mean()
        return {
            feature_name_for_window("min", window_size): [windows.min().mean()],
            feature_name_for_window("max", window_size): [windows.max().mean()],
            feature_name_for_window("mean", window_size): [windows.mean().mean()],
            feature_name_for_window("var", window_size): [windows.var().mean()],
            feature_name_for_window("q25", window_size): [Q25],
            feature_name_for_window("q27", window_size): [Q75],
            feature_name_for_window("q01", window_size): [windows.quantile(0.01).mean()],
            feature_name_for_window("q99", window_size): [windows.quantile(0.99).mean()],
            feature_name_for_window("iqr", window_size): [Q75 - Q25]
        }

def process_data():
    sample_size = int(150e3)
    windowed_features = ["min", "max", "mean", "var", "q25", "q75", "iqr", "q01", "q99"]
    window_sizes = [None, 50, 100, 1000]
    other_features = []
    columns = get_columns(windowed_features, window_sizes, other_features)
    train_sample_n = 3334   # We have about 629 million rows in total, so dedicating 3334 * 150 000  = 500 100 000 for training
    train_row_n = sample_size * train_sample_n
            
    x_train = create_x_dataframe(columns)
    x_val = create_x_dataframe(columns)
    y_train = create_y_dataframe()
    y_val = create_y_dataframe()

    train_data_file = "../data/train.csv"
    if not os.path.isfile(train_data_file):
        print("train.csv file not found in data folder")
        print("this file is not commited in the repo and needs to be added manually")
        return (None, None, None, None)
    
    chunks_processed = 0    
    for chunk in pd.read_csv(train_data_file, chunksize=sample_size):
        if chunk.shape[0] < sample_size:        # throwing out last remaining rows
            break
        features = {}
        for window_size in window_sizes:
            features.update(calculate_statistical_features(chunk, window_size))
            
        if chunks_processed * sample_size < train_row_n:
            x_train = x_train.append(pd.DataFrame(data=features))
            y_train = y_train.append(create_y_dataframe(chunk.time_to_failure.tail(1).values))
        else:
            x_val = x_val.append(pd.DataFrame(data=features))
            y_val = y_val.append(create_y_dataframe(chunk.time_to_failure.tail(1).values))
        
        chunks_processed += 1

        if chunks_processed % 20 == 0:
            print("{0} chunks processed, {1} remaining".format(chunks_processed, 4193 - chunks_processed))

    return x_train, y_train, x_val, y_val

x_train, y_train, x_val, y_val = process_data()
