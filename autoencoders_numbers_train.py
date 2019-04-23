
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.models import load_model

# ----------------------------------------------

entrada = Input( shape=[10] )
rede = Dense(64, activation='tanh')(entrada)
rede = Dense(512, activation='relu')(rede)
rede = Dense(256, activation='relu')(rede)

model = Model(inputs=entrada , outputs=rede)
model.compile(optimizer=SGD(lr=0.012, momentum=0.005, decay=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train =   x_train.reshape( x_train.shape[0], x_train.shape[1], x_train.shape[2], 1 )
x_test =    x_test.reshape( x_test.shape[0], x_test.shape[1], x_test.shape[2], 1 )
x_train =   x_train.astype('float32') / 255
x_test =    x_test.astype('float32') / 255
y_train =   to_categorical(y_train, 10)
y_test =    to_categorical(y_test, 10)

encoder = load_model('redes/mnist_encoder.h5')
x_train_num = encoder.predict( x_train )
x_test_num = encoder.predict( x_test )

model.fit( y_train, x_train_num, batch_size=32, validation_data=(y_test, x_test_num), epochs= 20,
  #callbacks=[ EarlyStopping(monitor='val_loss', min_delta=0.00001) ] 
)
# ---
model.save('redes/mnist_autoencoder_numero.h5')


