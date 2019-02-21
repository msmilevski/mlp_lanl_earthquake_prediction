import torch
import torch.nn as nn
from torch.autograd import Variable

class GRU(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(GRU, self).__init__()

		self.hidden_size = hidden_size
		self.input_size = input_size
		# self.element_size = element_size

		self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
		self.linear = nn.Linear(hidden_size, output_size)

	def forward(self, input):
		hidden = self.initHidden()
		_, hn = self.gru(input, hidden)
		## from (1, N, hidden) to (N, hidden)
		rearranged = hn.view(hn.size()[1], hn.size(2))
		out1 = self.linear(rearranged)
		return out1

	def initHidden(self):
		return Variable(torch.randn(1, 1, self.hidden_size))

	def reset_parameters(self):
		"""
		Re-initializes the networks parameters
		"""
		self.gru.reset_parameters()
		self.linear.reset_parameters()