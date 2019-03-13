import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_shape):
        """
        Initializes a convolutional network module object.
        :param input_shape: The shape of the inputs going in to the network.
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every conv layer, except dim reduction stages, where those are automatically infered.
        """
        super(ConvolutionalNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        x = torch.zeros((self.input_shape))
        out = x
        print("Input: " + str(out.shape))
        self.layer_dict['conv_{}'.format(0)] = nn.Conv1d(in_channels=out.shape[1],
                                                         out_channels=16,
                                                         kernel_size=64,
                                                         stride=2,
                                                         padding=32)

        out = self.layer_dict['conv_{}'.format(0)](out)

        out = F.relu(out)
        self.layer_dict['max-pool_{}'.format(0)] = nn.MaxPool1d(kernel_size=8,
                                                                stride=8,
                                                                padding=0)
        out = self.layer_dict['max-pool_{}'.format(0)](out)

        self.layer_dict['conv_{}'.format(1)] = nn.Conv1d(in_channels=out.shape[1],
                                                         out_channels=64,
                                                         kernel_size=32,
                                                         stride=2,
                                                         padding=16)

        out = self.layer_dict['conv_{}'.format(1)](out)

        out = F.relu(out)
        self.layer_dict['max-pool_{}'.format(1)] = nn.MaxPool1d(kernel_size=8,
                                                                stride=8,
                                                                padding=0)
        out = self.layer_dict['max-pool_{}'.format(1)](out)

        self.layer_dict['conv_{}'.format(2)] = nn.Conv1d(in_channels=out.shape[1],
                                                         out_channels=128,
                                                         kernel_size=16,
                                                         stride=2,
                                                         padding=8)

        out = self.layer_dict['conv_{}'.format(2)](out)

        out = F.relu(out)
        self.layer_dict['max-pool_{}'.format(2)] = nn.MaxPool1d(kernel_size=8,
                                                                stride=8,
                                                                padding=0)
        out = self.layer_dict['max-pool_{}'.format(2)](out)

        out = F.relu(out)
        self.layer_dict['conv_{}'.format(3)] = nn.Conv1d(in_channels=out.shape[1],
                                                         out_channels=256,
                                                         kernel_size=8,
                                                         stride=2,
                                                         padding=4)

        out = self.layer_dict['conv_{}'.format(3)](out)

        out = F.relu(out)
        self.layer_dict['conv_{}'.format(4)] = nn.Conv1d(in_channels=out.shape[1],
                                                         out_channels=1401,
                                                         kernel_size=16,
                                                         stride=12,
                                                         padding=4)

        out = self.layer_dict['conv_{}'.format(4)](out)

        out = F.relu(out)
        out = torch.reshape(out, (out.shape[0], out.shape[1]))
        self.linear_layer = nn.Linear(in_features=out.shape[1],
                                      out_features=1,
                                      bias=True)
        out = self.linear_layer(out)

        return out

    def forward(self, x):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        print("Input: " + str(out.shape))
        out = self.layer_dict['conv_{}'.format(0)](out)
        out = F.relu(out)
        out = self.layer_dict['max-pool_{}'.format(0)](out)
        out = self.layer_dict['conv_{}'.format(1)](out)
        out = F.relu(out)
        out = self.layer_dict['max-pool_{}'.format(1)](out)
        out = self.layer_dict['conv_{}'.format(2)](out)
        out = F.relu(out)
        out = self.layer_dict['max-pool_{}'.format(2)](out)
        out = F.relu(out)
        out = self.layer_dict['conv_{}'.format(3)](out)
        out = F.relu(out)
        out = self.layer_dict['conv_{}'.format(4)](out)
        out = F.relu(out)
        out = torch.reshape(out, (out.shape[0], out.shape[1]))
        out = self.linear_layer(out)

        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

        self.linear_layer.reset_parameters()