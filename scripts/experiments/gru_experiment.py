from mini_data_provider import MiniDataProvider
from raw_data_provider import RawDataProvider
from GRU import GRU
import numpy as np
import torch
from experiment_builder import ExperimentBuilder

seed = 1239931
rng = np.random.RandomState(seed=seed)  # set the seeds for the experiment
torch.manual_seed(seed=seed) # sets pytorch's seed

segment_size = 150000
element_size = 1000

train_data = RawDataProvider('train', segment_size=segment_size, element_size=element_size, rng=rng)
val_data = RawDataProvider('val', segment_size=segment_size, element_size=element_size, rng=rng)

model = GRU(input_size = element_size, hidden_size = 100, output_size=1)

experiment = ExperimentBuilder(network_model=model,
                                    experiment_name="raw_data",
                                    num_epochs=30,
                                    weight_decay_coefficient=1e-05,
                                    use_gpu=False,
                                    train_data=train_data, 
                                    val_data=val_data
                                    )
experiment_metrics = experiment.run_experiment()  # run experiment and return experiment metrics
