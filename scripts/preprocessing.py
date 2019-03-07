import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import os
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments") )
from test_file import DataProvider

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

def write_to_file(file, df):
    with open(file, 'a') as f:
        df.to_csv(f, header=False, index=False)

def fourier_transform(signal, start, end):
    ft = np.fft.fft(signal)
    np.put(ft, range(start, end), 0.0)
    ft = np.real(np.fft.ifft(ft))
    return ft

def calculate_statistical_features(acoustic_data, window_size=None):
    if window_size == None:
        Q75 = acoustic_data.quantile(0.75)
        Q25 = acoustic_data.quantile(0.25)
        mean = acoustic_data.mean()
        var = acoustic_data.var()
        return {
            "min": acoustic_data.min(),
            "max": acoustic_data.max(),
            "mean": mean,
            "var": var,
            "var_norm": var / mean,
            "q25": Q25,
            "q75": Q75,
            "q01": acoustic_data.quantile(0.01),
            "q99": acoustic_data.quantile(0.99),
            "iqr": Q75 - Q25,
            "kurtosis": acoustic_data.kurtosis(),
            "skew": acoustic_data.skew()
        }
    else:
        if type(window_size) == int:
            windows = acoustic_data.rolling(window=window_size)
            windows_mean = windows.mean()

            Q75 = windows_mean.quantile(0.75)
            Q25 = windows_mean.quantile(0.25)
            mean = windows_mean.mean()
            var = windows_mean.var()
            return {
                feature_name_for_window("min", window_size): windows_mean.min(),
                feature_name_for_window("max", window_size): windows_mean.max(),
                feature_name_for_window("mean", window_size): mean,
                feature_name_for_window("var", window_size): var,
                feature_name_for_window("var_norm", window_size): var / mean,
                feature_name_for_window("q25", window_size): Q25,
                feature_name_for_window("q75", window_size): Q75,
                feature_name_for_window("q01", window_size): windows_mean.quantile(0.01),
                feature_name_for_window("q99", window_size): windows_mean.quantile(0.99),
                feature_name_for_window("iqr", window_size): Q75 - Q25,
                feature_name_for_window("kurtosis", window_size): windows_mean.kurt(),
                feature_name_for_window("skew", window_size): windows_mean.skew()
            }
        else:
            window = int(window_size[2:4])
            if window >= 50:
                signal = fourier_transform(acoustic_data.values, np.int32(acoustic_data.values.shape[0] * window / 100), np.int32(acoustic_data.values.shape[0]))
            else:
                signal = fourier_transform(acoustic_data.values, 0, np.int32(acoustic_data.values.shape[0] * window / 100))
            
            Q75 = np.quantile(signal, 0.75)
            Q25 = np.quantile(signal, 0.25)
            mean = signal.mean()
            var = signal.var()            
            x = pd.Series(signal.ravel())
            
            return{
            feature_name_for_window("min", window_size): [signal.min()],
            feature_name_for_window("max", window_size): [signal.max()],
            feature_name_for_window("mean", window_size): [mean],
            feature_name_for_window("var", window_size): [var],
            feature_name_for_window("var_norm", window_size): [var / mean],
            feature_name_for_window("q25", window_size): [Q25],
            feature_name_for_window("q75", window_size): [Q75],
            feature_name_for_window("q01", window_size): [np.quantile(signal, 0.01)],
            feature_name_for_window("q99", window_size): [np.quantile(signal, 0.99)],
            feature_name_for_window("iqr", window_size): [Q75 - Q25],
            feature_name_for_window("kurtosis", window_size): [x.kurtosis()],
            feature_name_for_window("skew", window_size): [x.skew()]
            }

def process_data():
    sample_size = int(150e3)
    windowed_features = ["min", "max", "mean", "var", "var_norm", "kurtosis", "skew",
        "q25", "q75", "iqr", "q01", "q99"]
    window_sizes = [None, 50, 100, 1000, "FT10", "FT90"]
    other_features = []
    columns = get_columns(windowed_features, window_sizes, other_features)
    train_sample_n = 3334   # We have about 629 million rows in total, so dedicating 3334 * 150 000  = 500 100 000 for training
    train_row_n = sample_size * train_sample_n * 10   # 10 -- is because we're overlapping by 90%
            
    x_train = create_x_dataframe(columns)
    x_val = create_x_dataframe(columns)
    y_train = create_y_dataframe()
    y_val = create_y_dataframe()

    train_data_file = "data/train.csv"
    save_x_train_file = "data/new_x_train.csv"
    save_x_val_file = "data/new_x_val.csv"
    save_y_train_file = "data/new_y_train.csv"
    save_y_val_file = "data/new_y_val.csv"
    
    chunks_processed = 0
    data_provider = DataProvider(data_filepath=train_data_file, num_chunks=2)
    for acoustic_data, time in data_provider.next(is_baseline=True):
        if acoustic_data.shape[0] < sample_size:        # throwing out last remaining rows
            break
        features = {}
        acoustic_data = pd.DataFrame(acoustic_data)
        for window_size in window_sizes:
            features.update(calculate_statistical_features(acoustic_data, window_size))
            
        
        x = pd.DataFrame(data=features)
        y = create_y_dataframe([time])
        if chunks_processed * sample_size < train_row_n:
            write_to_file(save_x_train_file, x)
            write_to_file(save_y_train_file, y)
            # x_train = x_train.append(x, sort=False, ignore_index=True)
            # y_train = y_train.append(y, sort=False, ignore_index=True)
        else:
            write_to_file(save_x_val_file, x)
            write_to_file(save_y_val_file, y)
            # x_val = x_val.append(x, sort=False, ignore_index=True)
            # y_val = y_val.append(y, sort=False, ignore_index=True)
        
        chunks_processed += 1

        if chunks_processed % 10 == 0:
            print("{0} chunks processed, {1} remaining".format(chunks_processed, 41930 - chunks_processed))

    # return x_train, y_train, x_val, y_val

x_train, y_train, x_val, y_val = process_data()
print(x_train)
