from mini_data_provider import MiniDataProvider
from SeqDataProvider import SeqDataProvider
from raw_data_provider import RawDataProvider
from LSTM import LSTM
import numpy as np
import torch
from experiment_builder import ExperimentBuilder
from arg_extractor import get_args

args = get_args() 

rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed


# train_data = SeqDataProvider(which_set='train', data_path=args.data_path, segment_size=args.segment_size,
#   element_size=args.element_size, feature_size=58, overlap=True, rng=rng)
# val_data = SeqDataProvider(which_set='val', data_path=args.data_path, segment_size=args.segment_size, 
#   element_size=args.element_size, feature_size=58, overlap=True, rng=rng)
train_data = MiniDataProvider(which_set='train', data_path=args.data_path, segment_size=args.segment_size,
  element_size=args.element_size, rng=rng)
val_data = MiniDataProvider(which_set='val', data_path=args.data_path, segment_size=args.segment_size,
  element_size=args.element_size, rng=rng)

model = LSTM(input_size = args.element_size, hidden_size = 100, output_size=1, dropout=args.dropout, num_layers=args.num_layers)

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

experiment.load_model("weight_save_test/saved_models", "train_model", 3)

# model.load_state_dict(torch.load("weight_save_test/saved_models/train_model_3")['network'])
# print(model.state_dict())

print("\nyayy it worked")

# print("Done")
