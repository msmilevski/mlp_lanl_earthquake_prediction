import torch
import torch.nn as nn
from torch.autograd import Variable

class GRU(nn.Module):
	def __init__(self, input_size, sequence_len, hidden_size, output_size, batch_size=1, 
		num_layers=1, dropout=0):
		super(GRU, self).__init__()

		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.sequence_len = sequence_len

		self.num_layers = num_layers
		self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
		self.linear = nn.Linear(hidden_size, output_size)

	def forward(self, input):
		hidden = self.initHidden(input.device)
		gru_out, hidden = self.gru(input.view(self.sequence_len, self.batch_size, self.input_size), hidden)		

		rearranged = gru_out[-1].view(self.batch_size, self.hidden_size)

		out = self.linear(rearranged)
		return out.view(-1)

	def initHidden(self, device):
		return Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size)).to(device)

	def reset_parameters(self):
		"""
		Re-initializes the networks parameters
		"""
		self.gru.reset_parameters()
		self.linear.reset_parameters()