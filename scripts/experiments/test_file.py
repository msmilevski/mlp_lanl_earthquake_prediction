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

    def next(self, is_baseline=False):
        # Initialize helper variables
        batch_sample_x = []
        batch_sample_y = []
        last_chunk_x = None
        last_chunk_y = None
        num_batch = 0

        for chunk in pd.read_csv(self.data_file, chunksize=self.num_points):
            # Gather data by column name
            sample_x = np.array(chunk['acoustic_data'])
            sample_y = np.array(chunk['time_to_failure'])

            if num_batch > 0:
                last_chunk_x = sample_x[-self.chunk_size:]
                last_chunk_y = sample_y[-self.chunk_size:]

            if (last_chunk_x != None) and (last_chunk_y != None):
                sample_x = last_chunk_x + sample_x
                sample_y = last_chunk_y + sample_y

            assert len(sample_x) == len(sample_y)

            sample_size = len(sample_x)

            for j in np.arange(0, sample_size - self.chunk_size, self.window_size):
                # Append chunk to batch
                batch_sample_x.append(sample_x[j:j + self.chunk_size])
                batch_sample_y.append(sample_y[j + self.chunk_size])

                if is_baseline:
                    yield sample_x[j:j + self.chunk_size], sample_y[j + self.chunk_size]

            if not is_baseline:
                # Create indecies
                idx = np.arange(0, sample_size)
                # Shuffle them
                np.random.shuffle(idx)
                batch_sample_x = np.array(batch_sample_x)
                batch_sample_y = np.array(batch_sample_y)
                # Re-arange batch
                batch_sample_x = batch_sample_x[idx]
                batch_sample_y = batch_sample_y[idx]

                yield batch_sample_x, batch_sample_y

            num_batch += 1


# Example of usage:
# dp = DataProvider(chunk_size=10, num_chunks=2)
# i = 0
# for k in dp.next():
#     print(k)
#
#     if i == 2:
#         break
#     i += 1
