import pandas as pd
import numpy as np


class DataProvider(object):
    def __init__(self, data_filepath='../../data/train.csv', chunk_size=150000, num_chunks=6):
        '''
        :param data_filepath: string with the location of the training set
        :param chunk_size: size of the training instance
        :param num_chunks: number of continious chunks we want to take from the time series
        '''
        self.data_file = data_filepath
        self.chunk_size = chunk_size
        self.num_points = chunk_size * num_chunks
        self.window_size = int(chunk_size * 0.3)

    def next(self):
        # Initialize helper variables
        batch_sample_x = []
        batch_sample_y = []
        num_batch = 0

        for chunk in pd.read_csv(self.data_file, chunksize=self.num_points):
            # Gather data by column name
            sample_x = np.array(chunk['acoustic_data'])
            sample_y = np.array(chunk['time_to_failure'])

            for j in np.arange(0, self.num_points - self.chunk_size, self.window_size):
                # Append chunk to batch
                batch_sample_x.append(sample_x[j:j + self.chunk_size])
                batch_sample_y.append(sample_y[j + self.chunk_size])

            yield np.array(batch_sample_x), np.array(batch_sample_y)
            num_batch += 1

# Example of usage:
#   dp = DataProvider(chunk_size=10, num_chunks=2)
#   for k in dp.next():
#       print(k)
#       break

