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
    batch_size = 27
    learning_parameters.alpha = .1
    learning_parameters.epsilon = 1e-4
    num_epochs = 100
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
    # ----------------------------------------------------

    img_path = "../train_validation/"
    img_files = []
    # there was a ghost file getting added to this
    for file in os.listdir(img_path):
        if file.endswith(".jpg"):
            img_files.append(file)

    # load all images from folder, split them later
    imgs = [load_image(img_path + file) for file in img_files]
    labels = [extract_label(file) for file in img_files]


    # imgs shape by [158,1,28,28]
    imgs = 2* np.expand_dims(imgs,axis=1) -1

    # shuffle the imgs
    np.random.shuffle(imgs)

    # make (0, 255) to (-1, 1)
    #imgs = np.where(imgs == 0, -1, 1).astype(theano.config.floatX)
    # split set as 
    # train: 0.7; 
    # valid: 0.15
    # test:  0.15
    
    # shape by [110,1,28,28]
    train_images = imgs[0:50]
    # shape by [24,1,28,28]
    valid_images = imgs[50:100]
    # shape by [24,1,28,28]
    test_images = imgs[100:]


    # Binarise the inputs.
    train_images = np.where(train_images < 0, -1, 1).astype(theano.config.floatX)
    valid_images = np.where(valid_images < 0, -1, 1).astype(theano.config.floatX)
    test_images = np.where(test_images < 0, -1, 1).astype(theano.config.floatX)

    train_labels = np.hstack(labels[0:50])
    valid_labels = np.hstack(labels[50:100])
    test_labels = np.hstack(labels[100:])

    # Onehot the targets
    train_labels = np.float32(np.eye(2)[train_labels])    
    valid_labels = np.float32(np.eye(2)[valid_labels])
    test_labels = np.float32(np.eye(2)[test_labels])
    
    # -------------------------------------------------------

    print('Building the MLP...') 

    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    mlp = lfc.genLfc(input, 2, learning_parameters)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    # W updates
    W = lasagne.layers.get_all_params(mlp, quantized=True)
    W_grads = quantized_net.compute_grads(loss,mlp)
    updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
    updates = quantized_net.clipping_scaling(updates,mlp)
    
    # other parameters updates
    params = lasagne.layers.get_all_params(mlp, trainable=True, quantized=False)
    updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    quantized_net.train(
            train_fn,val_fn,
            mlp,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_images,train_labels,
            valid_images,valid_labels,
            test_images,test_labels,
            save_path,
            shuffle_parts)