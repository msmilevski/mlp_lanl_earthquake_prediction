from mini_data_provider import MiniDataProvider
from SeqDataProvider import SeqDataProvider
from raw_data_provider import RawDataProvider
from overlapped_data_provider import OverlappedDataProvider
from LSTM import LSTM
import numpy as np
import torch
from experiment_builder import ExperimentBuilder
from arg_extractor import get_args
import os


args = get_args() 

rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed

data_path = os.path.join(args.data_path, "only_train.csv")
train_data = OverlappedDataProvider(data_filepath=data_path, chunk_size=args.segment_size, batch_size=args.batch_size)
batch_size = train_data.batch_size
val_data = RawDataProvider(which_set='val', data_path=args.data_path, segment_size=args.segment_size, 
  element_size=args.element_size, rng=rng, batch_size=batch_size)


model = LSTM(input_size = args.element_size, hidden_size = 100, output_size=1, 
	sequence_len=args.segment_size // args.element_size, dropout=args.dropout, batch_size=batch_size)

experiment = ExperimentBuilder(network_model=model,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    use_gpu=args.use_gpu,
                                    train_data=train_data, 
                                    val_data=val_data,
                                    gpu_id=args.gpu_id,
                                    learning_rate=args.learning_rate)

experiment_metrics = experiment.run_experiment()  # run experiment and return experiment metrics
