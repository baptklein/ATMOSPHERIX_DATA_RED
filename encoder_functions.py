#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:00:48 2022

@author: florian
"""

import numpy as np # linear algebra
import random as rad
import os
import gc 
import seaborn 
from matplotlib import pyplot as plt
from pylab import *

import torch
from torch import tensor
from torch import nn

from tqdm.notebook import tqdm as tqdm
import pickle


class AE(nn.Module):
#     Create neural network
# An autoencoder is a type of artificial neural network used to learn efficient codings of unlabeled data. 
#The encoding is validated and refined by attempting to regenerate the input from the encoding.
# The autoencoder learns a representation (encoding) for a set of data, typically for dimensionality reduction
#, by training the network to ignore insignificant data (“noise”).
# Here the latent representation will be of dimension 5
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer_1 = nn.Linear(kwargs['length'],1022)

        self.encoder_hidden_layer_2 = nn.Linear(1022,511)

        self.encoder_hidden_layer_3 = nn.Linear(511,256)

        self.encoder_hidden_layer_4 = nn.Linear(256,128)

        self.encoder_hidden_layer_5 = nn.Linear(128,32)

        self.encoder_hidden_layer_6 = nn.Linear(32,16)

        self.encoder_output_layer = nn.Linear(16,5)

        self.decoder_hidden_layer_1 = nn.Linear(5,16)

        self.decoder_hidden_layer_2 = nn.Linear(16,32)

        self.decoder_hidden_layer_3 = nn.Linear(32,128)

        self.decoder_hidden_layer_4 = nn.Linear(128, 256)

        self.decoder_hidden_layer_5 = nn.Linear(256, 511)

        self.decoder_hidden_layer_6 = nn.Linear(511, 1022)

        self.decoder_output_layer = nn.Linear(1022,kwargs['length'])
        
        self.activation = nn.PReLU()

    def forward(self, features):
        activation = self.encoder_hidden_layer_1(features)
        activation = self.activation(activation)

        activation = self.encoder_hidden_layer_2(activation)
        activation = self.activation(activation)
        
        activation = self.encoder_hidden_layer_3(activation)
        activation = self.activation(activation)
        
        activation = self.encoder_hidden_layer_4(activation)
        activation = self.activation(activation)
        
        activation = self.encoder_hidden_layer_5(activation)
        activation = self.activation(activation)
        
        activation = self.encoder_hidden_layer_6(activation)
        activation = self.activation(activation)

        code = self.encoder_output_layer(activation)
        code = self.activation(code)
        
        activation = self.decoder_hidden_layer_1(code)
        activation = self.activation(activation)
        
        activation = self.decoder_hidden_layer_2(activation)
        activation = self.activation(activation)
        
        activation = self.decoder_hidden_layer_3(activation)
        activation = self.activation(activation)
        
        activation = self.decoder_hidden_layer_4(activation)
        activation = self.activation(activation)
        
        activation = self.decoder_hidden_layer_5(activation)
        activation = self.activation(activation)
        
        activation = self.decoder_hidden_layer_6(activation)
        activation = self.activation(activation)
        
        activation = self.decoder_output_layer(activation)
        reconstructed = self.activation(activation)

        return reconstructed




def get_training_and_test_sets(flux):
    # Randomly select indexes of training and test sets
    # The model will be trained on the training data only
    indexes = range(len(flux))
    train_indexes = rad.sample(indexes, int(0.8*len(indexes)))
    boolean_indexes = []
    for idx in indexes:
        if idx in train_indexes:
            boolean_indexes.append(True)
        else:
            boolean_indexes.append(False)
    boolean_indexes=np.array(boolean_indexes)
    indexes=np.array(indexes)
    
    train_x = tensor(np.float32(flux[boolean_indexes]))
    train_y = train_x
    test_x = tensor(np.float32(flux[~boolean_indexes]))
    test_y = test_x
    
    return train_x, train_y, test_x, test_y

def train_model(train_x, train_y, test_x, test_y, device,epochs = 5000, lr = 2e-6, debug=False):

    # jargon: epochs := number of iteration in the training loop

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(length=len(train_x[0])).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate lr 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # mean-squared error loss
    criterion = nn.MSELoss()

    gc.collect()
    torch.cuda.empty_cache()

    train_losses = []
    test_losses = []
    if debug:
        for epoch in tqdm(range(epochs), leave=False):
            output = model(train_x)
            train_loss = criterion(output, train_y)
            train_losses.append(train_loss)
            test_losses.append(criterion(model(test_x), test_y).detach().cpu().numpy())
            train_loss.backward()
            optimizer.step()
    else:
        for epoch in range(epochs):
            output = model(train_x)
            train_loss = criterion(output, train_y)
            train_losses.append(train_loss)
            test_losses.append(criterion(model(test_x), test_y).detach().cpu().numpy())
            train_loss.backward()
            optimizer.step()
        
    return model, train_losses, test_losses

