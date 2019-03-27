
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Reshape
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

# ----------------------------------------------

# entradaEnc = Input(shape = (28, 28, 1) ) # 784 total multiplicado
# redeEnc = Conv2D(64, kernel_size=(7,7), activation='relu', padding='same', strides=1)(entradaEnc)
# redeEnc = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', strides=1)(redeEnc)
# redeEnc = Flatten()(redeEnc)
# redeEnc = Dense(10, activation='softmax')(redeEnc)

# modelEnc = Model(inputs=entradaEnc, outputs=redeEnc)
# modelEnc.compile(optimizer=SGD(lr=0.012, momentum=0.005, decay=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# ---

# 28 * 28 * 32 = 25088
# entradaDec = Input(shape = [10] )
# redeDec = Dense( 10,  activation='softmax')(entradaDec)
# redeDec = Dense( 250 * 250 * 4,  activation='tanh')(entradaDec)
# redeDec = Reshape( (250, 250, 4) )(redeDec)
# redeDec = Conv2D(4, kernel_size=(3,3), activation='softplus',    padding='valid', strides=4, dilation_rate=1 )(redeDec)
# # 62 , 62 + 2
# redeDec = Conv2D(2, kernel_size=(3,3), activation='relu', padding='valid', strides=2, dilation_rate=1 )(redeDec)
# # 30 , 30 + 2
# redeDec = Conv2D(1, kernel_size=(3,3), activation='relu', padding='valid', strides=1, dilation_rate=1 )(redeDec)
# 28 , 28

entradaDec = Input(shape = [10] )
redeDec = Dense( 10,  activation='softmax')(entradaDec)
redeDec = Dense( 36 * 36 * 32,  activation='relu')(entradaDec)
redeDec = Reshape( (36, 36, 32) )(redeDec)
redeDec = Conv2D(32, kernel_size=(5,5), activation='relu', padding='valid' )(redeDec)
redeDec = Conv2D(16, kernel_size=(3,3), activation='softplus', padding='valid' )(redeDec)
redeDec = Conv2D(1, kernel_size=(3,3), activation='tanh', padding='valid' )(redeDec)

# modelDec = Model(inputs=entradaDec, outputs=redeDec)
# modelDec.compile(optimizer=SGD(lr=0.012, momentum=0.005, decay=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

decoder = Model(inputs=entradaDec , outputs= redeDec)
decoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
print( decoder.summary() )

# ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train =   x_train.reshape( x_train.shape[0], x_train.shape[1], x_train.shape[2], 1 )
x_test =    x_test.reshape( x_test.shape[0], x_test.shape[1], x_test.shape[2], 1 )
x_train =   x_train.astype('float32') / 255
x_test =    x_test.astype('float32') / 255
y_train =   to_categorical(y_train, 10)
y_test =    to_categorical(y_test, 10)

decoder.fit( y_train, x_train, batch_size=32, validation_data=(y_test, x_test), epochs= 8 )
# ---
decoder.save('redes/mnist_decoder_1.h5')

