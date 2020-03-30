# BNN_image_classifier

This repo takes examples from [here](https://github.com/Xilinx/BNN-PYNQ), to get more details visit original repo

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

