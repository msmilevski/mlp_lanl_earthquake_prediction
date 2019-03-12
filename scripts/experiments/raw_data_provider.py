from data_provider import DataProvider
import os
import numpy as np

class RawDataProvider(DataProvider):
    # returns the raw data of shape:  (num_segments, segment_size / element_size, element_size)

    def __init__(self, data_path, which_set='train', segment_size=150000, element_size=1000,
        batch_size=1, rng=None, downsampled=False, mini=False):
        assert which_set in ['train', 'val'], (
            'Expected which_set to be either train or val '
            'Got {0}'.format(which_set)
        )

        # path of this python script
        # file_path = os.path.join(data_path, "mini_{0}_dataset.csv".format(which_set))
        file_name_template = "only_{0}.csv"
        if downsampled:
            file_name_template = "downsampled4_{0}.csv" 
        elif mini:
            file_name_template = "only_{0}_mini.csv"

        file_path = os.path.join(data_path, file_name_template.format(which_set))
        assert os.path.isfile(file_path), (
            'Data file does not exist at expected path: ' + file_path
        )
        
        print("file_path: {0}".format(file_path))
        print("Loading data")

        loaded = np.loadtxt(file_path, delimiter=",", skiprows=1, dtype=[('signal', np.int16), ('time', np.float)])        

        suitable_data_length = loaded.shape[0] // segment_size * segment_size
        amplitudes = loaded['signal'][:suitable_data_length, None]
        times = loaded['time'][:suitable_data_length]

        n_amplitudes = amplitudes.shape[0]        
        inputs = amplitudes.reshape(n_amplitudes // segment_size, segment_size // element_size, element_size)

        targets = times[np.arange(segment_size - 1, times.shape[0], segment_size)]

        print("inputs shape: {0}, targets shape: {1}".format(inputs.shape, targets.shape))
        assert inputs.shape[0] == targets.shape[0]

        
        # pass the loaded data to the parent class __init__
        super(RawDataProvider, self).__init__(
            inputs, targets, batch_size, rng=rng)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(RawDataProvider, self).next()
        return inputs_batch, targets_batch