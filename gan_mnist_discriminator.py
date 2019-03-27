
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

# ----------------------------------------------

entrada = Input(shape = (28, 28, 1) )
rede = Conv2D(64, kernel_size=(7,7), activation='relu', padding='same', strides=1)(entrada)
rede = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', strides=1)(rede)
rede = Flatten()(rede)
rede = Dense(10, activation='softmax')(rede)

model = Model(inputs=entrada, outputs=rede)
model.compile(optimizer=SGD(lr=0.012, momentum=0.005, decay=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train =   x_train.reshape( x_train.shape[0], x_train.shape[1], x_train.shape[2], 1 )
x_test =    x_test.reshape( x_test.shape[0], x_test.shape[1], x_test.shape[2], 1 )
x_train =   x_train.astype('float32') / 255
x_test =    x_test.astype('float32') / 255
y_train =   to_categorical(y_train, 10)
y_test =    to_categorical(y_test, 10)

model.fit( x_train, y_train, batch_size=32, validation_data=(x_test, y_test), epochs= 8 )
# ---
model.save('redes/mnist_simples_1.h5')

