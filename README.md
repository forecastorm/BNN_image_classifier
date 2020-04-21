# BNN_image_classifier

This repo takes examples from [here](https://github.com/Xilinx/BNN-PYNQ), to get more details visit original repo. This repo will focus on the LFC model discussed in the original repository



#  Training Quick Start 

## 1. Set up docker environment
a cpu version of docker image is already built, one can pull the image from docker hub
```
docker pull forecastorm/bnn-pynq:cv2
```

and run the image with mounted volume

```
docker run -i -t -v ~/BNN_image_classifier:/root/BNN-PYNQ -v ~/BNN_image_classifier/train_validation:/root/.pylearn2 forecastorm/bnn-pynq:cv2 /bin/bash
```

where `~/BNN_image_classifier` referes to where you cloned this repository and `~/BNN_image_classifier/train_validation` refers to the path to our training dataset 

## 2. Running the Script

LFC support for 1-bit weights and 1 or 2 bit activations. 

```
$ python training.py -ab <activation_bits>  
```

Note that LFC requires special formating:

1. Import training data and store in a Nx1x28x28 numpy array, with each value either -1 to 1.
2. Put each output label into [one hot encoded](https://en.wikipedia.org/wiki/One-hot) format.

which was done in 'load_data.py' within this project

```
$ python training.py -ab <activation_bits>  
```

```
$ python gen-weights-W1A1.py 
```

This will generate folder 'binparam-lfc-pynq'

## 3. Inference on pynq supported board

1. Install the official PYNQ/BNN repository with pip3 install, this will call 'setup.py' which is neccessary to set up the environment; we will run notebooks inside the official 'bnn' just using our customized parameters with their supported overlays

2. Copy the 'binparam-lfc-pynq' folder to where your 'bnn/params' is located at on your board, referece [this excellent tutorial](https://www.hackster.io/adam-taylor/training-implementing-a-bnn-using-pynq-1210b9)