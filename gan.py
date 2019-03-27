
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

# ----------------------------------------------

entrada = Input(shape = (10,) )
rede = Dense(10, activation='relu')(entrada)




