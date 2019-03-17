import pandas as pd
import numpy as np
import math

class OverlappedDataProvider2(object):
    def __init__(self, file_path='data/provider_test_2.csv', batch_size=10, segment_size=150000, 
        overlap_fraction=0.9, data_splits=20):
        '''
        :param data_filepath: string with the location of the training set
        :param chunk_size: size of the training instance
        '''
        self.file_path = file_path
        self.segment_size = segment_size
        self.overlap_fraction = overlap_fraction
        self.slide_size = 1 - self.overlap_fraction
        self.window_size = math.ceil(self.slide_size * self.segment_size)
        self.batch_size = batch_size
        train_rows = int(5001e5)   # this is only used for progress bar so do not worry too much about it being hardcoded
        self.num_batches = int(train_rows / segment_size * (1 / self.slide_size) / batch_size)
        self.data_splits = data_splits

        print("loading data: {0}".format(self.file_path))
        self.loaded = np.loadtxt(self.file_path, delimiter=",", skiprows=1, 
            dtype=[('signal', np.int16), ('time', np.float)])
        print("finished loading data")

    def next(self, is_baseline=False):
        # Initialize helper variables
        last_chunk_x = np.array([])
        last_chunk_y = np.array([])
        num_reads = 0

        dtypes = {
            'acoustic_data': 'Int16',
            'time_to_failure': 'Float64'
        }

        
        total_num_points = self.loaded['signal'].shape[0]
        chunk_size = total_num_points // self.data_splits
        chunk_starts = np.arange(0, total_num_points, chunk_size)
        np.random.shuffle(chunk_starts)

        leftover_x = np.array([], dtype=np.int16)
        leftover_y = np.array([])
        for start in chunk_starts:            
            end = start + chunk_size
            end = min(end, total_num_points)   # off by 1?
            sample_x = self.loaded['signal'][start:end]
            sample_y = self.loaded['time'][start:end]
            sample_size = len(sample_x)
            
            batch_sample_x = leftover_x if len(leftover_x) > 0 else np.array([])
            batch_sample_y = leftover_y if len(leftover_x) > 0 else np.array([])
            for j in np.arange(0, sample_size - self.segment_size, self.window_size):
                # Append chunk to batch
                segment_x = sample_x[j:j + self.segment_size]
                segment_y = sample_y[(j-1) + self.segment_size]
                if len(batch_sample_x) == 0:
                    batch_sample_x = np.array([segment_x])
                    batch_sample_y = np.array([segment_y])
                else:
                    batch_sample_x = np.vstack([batch_sample_x, np.array([segment_x])])
                    batch_sample_y = np.vstack([batch_sample_y, np.array([segment_y])])

            # Create indecies
            idx = np.arange(0, len(batch_sample_x))
            # Shuffle them
            np.random.shuffle(idx)

            previous = 0
            for i in np.arange(self.batch_size, len(idx), self.batch_size):
                batch_ids = idx[previous:i]
                yield batch_sample_x[batch_ids], batch_sample_y[batch_ids]
                previous = i

            leftover_ids = idx[previous:]
            leftover_x = batch_sample_x[leftover_ids]
            leftover_y = batch_sample_y[leftover_ids]

    def __iter__(self):
        for next_batch in self.next():
            yield next_batch