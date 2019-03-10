import numpy as np
import torch
from experiment_builder import ExperimentBuilder
from arg_extractor import get_args
from Conv import ConvolutionalNetwork
from test_file import DataProvider

args = get_args()

rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)

train_data = DataProvider(data_filepath=args.data_path + 'train.csv', batch_size=args.batch_size,
                          chunk_size=args.segment_size)
valid_data = DataProvider(data_filepath=args.data_path + 'valid.csv', batch_size=args.batch_size,
                          chunk_size=args.segment_size)

model = ConvolutionalNetwork(input_shape=(args.batch_size, 1, args.segment_size), kernel_size=15, num_filters=32,
                             num_layers=args.num_layers)

experiment = ExperimentBuilder(network_model=model,
                               experiment_name=args.experiment_name,
                               num_epochs=args.num_epochs,
                               weight_decay_coefficient=args.weight_decay_coefficient,
                               use_gpu=args.use_gpu,
                               train_data=train_data,
                               val_data=valid_data,
                               gpu_id=args.gpu_id,
                               learning_rate=args.learning_rate)

experiment_metrics = experiment.run_experiment()
