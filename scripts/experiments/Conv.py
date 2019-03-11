import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_shape, kernel_size, num_filters, num_layers, use_bias=False):
        """
        Initializes a convolutional network module object.
        :param input_shape: The shape of the inputs going in to the network.
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every conv layer, except dim reduction stages, where those are automatically infered.
        :param num_layers: Number of conv layers (excluding dim reduction stages)
        :param use_bias: Whether our convolutions will use a bias.
        """
        super(ConvolutionalNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.num_layers = num_layers
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        x = torch.zeros((self.input_shape))
        out = x
        print("Input" + str(out.shape))

        self.layer_dict['conv_{}'.format(0)] = nn.Conv1d(in_channels=out.shape[1],
                                                         kernel_size=self.kernel_size,
                                                         out_channels=self.num_filters,
                                                         stride=50,
                                                         bias=self.use_bias)
        out = self.layer_dict['conv_{}'.format(0)](out)
        print("Conv" + str(0) + ":" + str(out.shape))
        out = F.relu(out)
        self.layer_dict['conv_{}'.format(1)] = nn.Conv1d(in_channels=out.shape[1],
                                                         kernel_size=self.kernel_size,
                                                         out_channels=self.num_filters,
                                                         stride=5,
                                                         bias=self.use_bias,
                                                         dilation=2)
        out = self.layer_dict['conv_{}'.format(1)](out)
        print("Conv" + str(1) + ":" + str(out.shape))
        out = F.relu(out)
        self.layer_dict['conv_{}'.format(2)] = nn.Conv1d(in_channels=out.shape[1],
                                                         kernel_size=self.kernel_size,
                                                         out_channels=self.num_filters,
                                                         stride=3,
                                                         bias=self.use_bias,
                                                         dilation=4)
        out = self.layer_dict['conv_{}'.format(2)](out)
        print("Conv" + str(2) + ":" + str(out.shape))
        out = F.relu(out)

        out = torch.reshape(out, (out.shape[0], out.shape[1] * out.shape[2]))
        self.linear_layer = nn.Linear(in_features=out.shape[1],
                                      out_features=1,
                                      bias=self.use_bias)
        print("End of conv: " + str(out.shape))
        out = self.linear_layer(out)
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        for i in range(self.num_layers):  # for number of layers
            out = self.layer_dict['conv_{}'.format(i)](out)  # pass through conv layer indexed at i
            out = F.relu(out)  # pass conv outputs through ReLU

        out = torch.reshape(out, (out.shape[0], out.shape[1] * out.shape[2]))
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


