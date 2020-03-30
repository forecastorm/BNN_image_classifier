from __future__ import print_function
import sys
import os
import time
from argparse import ArgumentParser
import numpy as np
np.random.seed(1234)  # for reproducibility
import theano
import theano.tensor as T
import lasagne
import cPickle as pickle
import gzip
import quantized_net
import lfc
from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial
from collections import OrderedDict
from load_data import *
# LFC support 1 bit or 2 bit activations
# LFC support 1 bit weights 

if __name__ == "__main__":
    
    # Parse some command line options
    parser = ArgumentParser(
        description="Train the LFC network on the train_validation dataset")
    parser.add_argument('-ab', '--activation-bits', type=int, default=1, choices=[1, 2],
        help="Quantized the activations to the specified number of bits, default: %(default)s")
    parser.add_argument('-wb', '--weight-bits', type=int, default=1, choices=[1],
        help="Quantized the weights to the specified number of bits, default: %(default)s")
    args = parser.parse_args()
    learning_parameters = OrderedDict()
    # Quantization parameters
    learning_parameters.activation_bits = args.activation_bits
    learning_parameters.weight_bits = args.weight_bits
    print("activation_bits = "+str(learning_parameters.activation_bits))
    print("weight_bits = "+str(learning_parameters.weight_bits))

    # hyper parameters-------------------------------
    batch_size = 100
    learning_parameters.alpha = .1
    learning_parameters.epsilon = 1e-4
    num_epochs = 1000
    learning_parameters.dropout_in = .2 # 0. means no dropout
    learning_parameters.dropout_hidden = .5
    # "Glorot" means we are using the coefficients from Glorot's paper
    learning_parameters.W_LR_scale = "Glorot" 
    # learning rate decay
    LR_start = .003
    LR_fin = 0.0000003
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    shuffle_parts = 1
    save_path = "train_validation-%dw-%da.npz" % (learning_parameters.weight_bits, learning_parameters.activation_bits)
    # ------------------------------------------------

    # will use train_validation in training process
    img_path = "../train_validation/"
    img_files = []
    # there was a ghost file getting added to this
    for file in os.listdir(img_path):
        if file.endswith(".jpg"):
            img_files.append(file)

    # load all images from folder, split them later
    imgs = [load_image(img_path + file) for file in img_files]
    labels = [extract_label(file) for file in img_files]


    print(imgs.shape)