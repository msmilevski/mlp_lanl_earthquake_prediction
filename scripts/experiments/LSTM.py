import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class LSTM(nn.Module):

    #input_size: number of features for each block
    #hidden_size: number of hidden states
    #sequence_len: sequence length 150 = (150000 / 1000)
    #batch_size: is controlled by the data provider so leave it 1
    def __init__(self, input_size=58, hidden_size=100, sequence_len=150, batch_size=1, output_size=1, dropout=0, num_layers=2):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.sequence_len = sequence_len
        self.dropout = dropout

        #LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout)

        #Output layer
        self.linear = nn.Linear(self.hidden_size, output_size)

    def init_hidden(self, device):
        # Hidden states init
        return (Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)),
                Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size)).to(device))

    def forward(self, input):
        hidden = self.init_hidden(input.device)
        #lstm_out: [input_size, batch_size, hidden_dim]
        #self.hidden: (a, b), both of size [num_layers, batch_size, hidden_dim]
        lstm_out, self.hidden = self.lstm(input.view(self.sequence_len, self.batch_size, self.input_size), hidden)

        #output of the last element to the output layer.
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, self.hidden_size))
        return y_pred.view(-1)

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        self.lstm.reset_parameters()
        self.linear.reset_parameters()
