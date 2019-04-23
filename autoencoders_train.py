
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from keras.initializers import Constant

# ----------------------------------------------

entradaEnc = Input(shape = (28, 28, 1) ) # 784 total multiplicado
redeEnc = Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', strides=1)(entradaEnc)
redeEnc = MaxPooling2D(pool_size=(2, 2))(redeEnc)
redeEnc = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', strides=1)(redeEnc)
redeEnc = MaxPooling2D(pool_size=(2, 2))(redeEnc)
redeEnc = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', strides=1)(redeEnc)
redeEnc = MaxPooling2D(pool_size=(2, 2))(redeEnc)
redeEnc = Flatten()(redeEnc)
redeEnc = Dense(256, activation='relu')(redeEnc)

encoder = Model(inputs=entradaEnc, outputs=redeEnc)

entradaDec = Input(shape = [256] )
redeDec = Reshape( (16, 16, 1) )(entradaDec)
redeDec = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same' )(redeDec)
redeDec = UpSampling2D(size=(2, 2))(redeDec)
redeDec = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same' )(redeDec)
redeDec = Conv2D(16, kernel_size=(3,3), activation='relu', padding='valid' )(redeDec)
redeDec = Conv2D(1, kernel_size=(3,3), activation='tanh', padding='valid' )(redeDec)

decoder = Model(inputs=entradaDec , outputs= redeDec)

# binary_crossentropy
# mean_squared_error
autoencoder = Model( encoder.input , decoder(encoder.output) )
autoencoder.compile(optimizer='adadelta', loss='logcosh', metrics=['accuracy'])
print( autoencoder.summary() )
print( decoder.summary() )

# ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train =   x_train.reshape( x_train.shape[0], x_train.shape[1], x_train.shape[2], 1 )
x_test =    x_test.reshape( x_test.shape[0], x_test.shape[1], x_test.shape[2], 1 )
x_train =   x_train.astype('float32') / 255
x_test =    x_test.astype('float32') / 255
#y_train =   to_categorical(y_train, 10)
#y_test =    to_categorical(y_test, 10)

autoencoder.fit( x_train, x_train, batch_size=32, validation_data=(x_test, x_test), epochs= 5,
  callbacks=[ EarlyStopping(monitor='val_loss', min_delta=0.001) ] 
)
# ---
encoder.save('redes/mnist_encoder.h5')
decoder.save('redes/mnist_decoder.h5')
autoencoder.save('redes/mnist_autoencoder.h5')

