
#
# https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a
#
# https://www.kaggle.com/poonaml/deep-neural-network-keras-way
#
# Modelo de funcionamnto de ResNet50 (rede residual)
# https://datascience.stackexchange.com/questions/33022/how-to-interpert-resnet50-layer-types
# 
# Explicação do funcionamento de redes residuais
# https://towardsdatascience.com/understanding-residual-networks-9add4b664b03
# 

import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Add, AveragePooling2D
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

# ----------------------------------------------------
# ----   Função para criar módulo residual
# Deve criar 2 camadas convolucionais na rede, seguida de uma soma
# dos valores de entrada com a resposta da 2ª camada conv.
# 
# @dados: camada de entrada para o módulo residual
# @filtros: quantidade de filtros em que cada camada conv neste módulo residual
# @strides: o 'pulo' aplicado no funcionamento da primeira camada conv deste módulo
def Residual(dados, filtros = 32, strides = 1):
  soma = dados
  if strides != 1 :
    soma = Conv2D(filtros, kernel_size=(1,1), activation='relu', padding='same', strides=strides)(dados)
  rede = Conv2D(filtros, kernel_size=(3,3), activation='relu', padding='same', strides=strides)(dados)
  rede = Conv2D(filtros, kernel_size=(3,3), activation='relu', padding='same')(rede)
  rede = Add()([ rede, soma ])
  return rede

# ----------------------------------------------------
# ----   Criar a rede
entrada = Input(shape = (28, 28, 1) )
rede = Conv2D(64, kernel_size=(7,7), activation='relu', padding='same', strides=1)(entrada)
#rede = MaxPooling2D(pool_size=(2,2))(rede)
rede = Residual(rede, 64)
rede = Residual(rede, 64)
rede = Residual(rede, 64)
rede = Residual(rede, 128, 2)
rede = Residual(rede, 128)
rede = Residual(rede, 128)
rede = Residual(rede, 128)
rede = Residual(rede, 512, 2)
rede = Residual(rede, 512)
rede = Residual(rede, 512)
rede = Residual(rede, 512)
rede = Residual(rede, 512)
rede = Residual(rede, 512)

rede = AveragePooling2D(pool_size=(4,4))(rede)
rede = Flatten()(rede)
#rede = Dense(512, activation='relu')(rede)
#rede = Dropout(0.1)(rede)
#rede = Dense(64, activation='elu')(rede)
#rede = Dropout(0.1)(rede)
rede = Dense(10, activation='softmax')(rede)

# criar o modelo final
model = Model(inputs=entrada, outputs=rede)
model.compile(optimizer=SGD(lr=0.012, momentum=0.005, decay=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# ----------------------------------------------------
# ----   Carregar os dados e treinar
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train =   x_train.reshape( x_train.shape[0], x_train.shape[1], x_train.shape[2], 1 )
x_test =    x_test.reshape( x_test.shape[0], x_test.shape[1], x_test.shape[2], 1 )
x_train =   x_train.astype('float32') / 255
x_test =    x_test.astype('float32') / 255
y_train =   to_categorical(y_train, 10)
y_test =    to_categorical(y_test, 10)
print('Shape entrada: ' + str(x_train.shape) )

# treinar
model.fit( x_train, y_train, batch_size=32, validation_data=(x_test, y_test), epochs= 16 )

# ----------------------------------------------------
# ----   Guardar a inteligência
model.save('redes/mnist.h5')

