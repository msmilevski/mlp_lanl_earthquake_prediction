import pandas as pd
import torch
import torch.nn as nn
from LSTM import LSTM
import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import test_data_provider

class TestEvaluator(nn.Module):
	def __init__(self, model, test_data, weights_file):
		super(TestEvaluator, self).__init__()

		self.model = model
		self.weights_file = weights_file

		state = torch.load(f=weights_file)
		self.load_state_dict(state_dict=state['network'])

		self.test_data = test_data

	def evaluate(self):
		predictions = np.array([])
		evaluated = 0
		for x in self.test_data:
			x = torch.Tensor(x.values).float()
			out = self.model.forward(x)  # forward the data in the model
			predictions = np.append(predictions, out.item())
			
			evaluated +=1
			print("{0}/2624 test points evaluated".format(evaluated))

		
		submission = pd.read_csv('data/sample_submission.csv')

		submission['time_to_failure'] = predictions

		submission.to_csv("submission_test.csv", index=False)


model = LSTM(input_size = 1000, hidden_size = 100, output_size=1, num_layers=2,
	sequence_len=150, dropout=0, batch_size=1)

evaluator = TestEvaluator(model=model, test_data=test_data_provider.iterate_test_data(), 
	weights_file="weight_test/saved_models/train_model_3")
evaluator.evaluate()

			
		