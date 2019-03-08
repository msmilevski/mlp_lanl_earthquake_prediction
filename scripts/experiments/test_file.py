import pandas as pd
import numpy as np


class DataProvider(object):
    def __init__(self, data_filepath='../../data/train.csv', batch_size=64, chunk_size=150000, num_chunks=6):
        '''
        :param data_filepath: string with the location of the training set
        :param chunk_size: size of the training instance
        :param num_chunks: number of continious chunks we want to take from the time series
        '''
        self.data_file = data_filepath
        self.chunk_size = chunk_size
        self.num_points = chunk_size * num_chunks
        self.window_size = int(chunk_size * 0.1)
        self.batch_size = batch_size


    def next(self, is_baseline=False):
        # Initialize helper variables
        batch_sample_x = []
        batch_sample_y = []
        last_chunk_x = np.array([])
        last_chunk_y = np.array([])
        num_reads = 0

        for chunk in pd.read_csv(self.data_file, chunksize=self.num_points):
            # Gather data from columns
            sample_x = np.array(chunk['acoustic_data'])
            sample_y = np.array(chunk['time_to_failure'])

            # Concatenate the points from the last batch with the new batch
            sample_x = np.concatenate((last_chunk_x, sample_x), axis=0)
            sample_y = np.concatenate((last_chunk_y, sample_y), axis=0)

            assert len(sample_x) == len(sample_y)

            sample_size = len(sample_x)

            for j in np.arange(0, sample_size - self.chunk_size, self.window_size):
                # Append chunk to batch
                batch_sample_x.append(sample_x[j:j + self.chunk_size])
                batch_sample_y.append(sample_y[j + self.chunk_size])

                if is_baseline:
                    yield sample_x[j:j + self.chunk_size], sample_y[j + self.chunk_size]

                if (not is_baseline) and (len(batch_sample_y) == self.batch_size):
                    # Create indecies
                    idx = np.arange(0, len(batch_sample_x))
                    # Shuffle them
                    np.random.shuffle(idx)
                    # Return re-aranged batch
                    yield np.array(batch_sample_x)[idx], np.array(batch_sample_y)[idx]
                    # Reset batch
                    batch_sample_x = []
                    batch_sample_y = []

            num_reads += 1
            # Save 90% of the data from this chunk
            if num_reads > 0:
                last_chunk_x = sample_x[-int(self.chunk_size * 0.9):]
                last_chunk_y = sample_y[-int(self.chunk_size * 0.9):]

        #print(len(batch_sample_x))




# Example of usage:
# import matplotlib.pyplot as plt
# dp = DataProvider(data_filepath='/afs/inf.ed.ac.uk/user/s18/s1885778/mlp_lanl_earthquake_prediction/data/mini_train_dataset.csv', chunk_size=100, num_chunks=3)
#
# for k in dp.next():
#     print(k[0].shape)

