from data_provider import DataProvider
import os
import numpy as np

class SeqDataProvider(DataProvider):
    # returns the data of shape:  (num_segments, segment_size / element_size, feature_size)

    def __init__(self, data_path, which_set='train', segment_size=150000, element_size=1000, feature_size=58, overlap=False,
        batch_size=1, rng=None):
        assert which_set in ['train', 'val'], (
            'Expected which_set to be either train or val '
            'Got {0}'.format(which_set)
        )
        
        
        file_path_x = os.path.join(data_path, "x_{0}.csv".format(which_set))
        assert os.path.isfile(file_path_x), (
            'Data file does not exist at expected path: ' + file_path_x
        )
        
        file_path_y = os.path.join(data_path, "y_{0}.csv".format(which_set))
        assert os.path.isfile(file_path_y), (
            'target file does not exist at expected path: ' + file_path_y
        )
        
        signal = np.loadtxt(file_path_x, delimiter=",", skiprows=1)        
        time = np.loadtxt(file_path_y, delimiter=",", skiprows=1)
        
        if overlap == False:
            data_length = signal.shape[0] // (segment_size // element_size) * (segment_size // element_size)
            signal = signal[:data_length, :]
            time = time[:data_length]
        
            n_signal = signal.shape[0]        
            inputs = signal.reshape(n_signal // (segment_size // element_size), segment_size // element_size, feature_size)
            targets = time[np.arange((segment_size // element_size) - 1, time.shape[0], (segment_size // element_size))]
        
        else:
            overlap_percentage = 10
            data_length = signal.shape[0] // (segment_size // element_size // overlap_percentage) * (segment_size // element_size // overlap_percentage)
            signal = signal[:data_length, :]
            time = time[:data_length]
        
            n_signal = int((signal.shape[0] / (segment_size // element_size // overlap_percentage))  - overlap_percentage)
            inputs = np.zeros((n_signal, segment_size // element_size, feature_size))
            targets = np.zeros(n_signal)
            for i in range(n_signal):
                inputs[i] = signal[(segment_size // element_size // overlap_percentage)*i:(segment_size // element_size)+((segment_size // element_size // overlap_percentage)*i)].reshape(segment_size // element_size, feature_size)
                targets[i] = time[(segment_size // element_size)+((segment_size // element_size // overlap_percentage)*i) - 1 ]
        
        print("inputs shape: {0}, targets shape: {1}".format(inputs.shape, targets.shape))
        assert inputs.shape[0] == targets.shape[0]
        
        # pass the loaded data to the parent class __init__
        super(SeqDataProvider, self).__init__(
            inputs, targets, batch_size, rng=rng)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(SeqDataProvider, self).next()
        return inputs_batch, targets_batch
