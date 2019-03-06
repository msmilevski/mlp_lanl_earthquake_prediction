import pandas as pd
import numpy as np


class DataProvider(object):
    def __init__(self, data_filepath='data/train.csv', chunk_size=150000, num_chunks=6):
        self.data_file = data_filepath
        self.chunk_size = chunk_size
        self.num_points = chunk_size * num_chunks
        self.window_size = int(chunk_size * 0.1)

    def next(self):
        batch_sample_x = []
        batch_sample_y = []
        num_batch = 0
        iterator = pd.read_csv(self.data_file, chunksize=self.num_points)

        for chunk in iterator.get_chunk():
            sample_x = np.array(chunk['acoustic_data'])
            sample_y = np.array(chunk['time_to_failure'])
            for j in np.arange(num_batch * self.num_points, 2 * num_batch * self.num_points, self.chunk_size):
                batch_sample_x.append(sample_x[j, j + self.chunk_size])
                batch_sample_y.append(sample_y[j + self.chunk_size])

            yield np.array(batch_sample_x), np.array(batch_sample_y)
            num_batch += 1
