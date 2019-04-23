
# 
# Modelos de GAN:
#   https://github.com/eriklindernoren/Keras-GAN
# 

import numpy as np
from PIL import Image
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Activation, Dropout, Flatten, Dense, LeakyReLU, Reshape, Conv2DTranspose, BatchNormalization, UpSampling2D
from keras.optimizers import Adam, RMSprop
from keras.datasets import mnist

# ----------------------------------------------
def criarGerador():
  entrada = Input(shape = (100,) )
  rede = Dense ( 7 * 7 * 128, activation='relu' )(entrada)
  rede = Reshape( (7, 7, 128) )(rede)

  rede = UpSampling2D()(rede)
  rede = Conv2D( 128, kernel_size=3, padding='same', strides=1 )(rede)
  rede = BatchNormalization(momentum=0.8)(rede)
  rede = Activation('relu')(rede)

  rede = UpSampling2D()(rede)
  rede = Conv2D( 64, kernel_size=3, padding='same', strides=1 )(rede)
  rede = BatchNormalization(momentum=0.8)(rede)
  rede = Activation('relu')(rede)

  rede = Conv2D( 1, kernel_size=3, padding='same', activation='tanh')(rede)

  model = Model(inputs=entrada, outputs=rede, name='gerador')
  print( model.summary() )
  return model

def criarDiscriminador():
  entrada = Input(shape = (28, 28, 1) )
  rede = Conv2D(32, kernel_size=(3,3), padding='same', strides=2)(entrada)
  rede = LeakyReLU(alpha = 0.2)(rede)

  rede = Conv2D(64, kernel_size=(3,3), padding='same', strides=2)(rede)
  rede = BatchNormalization(momentum=0.8)(rede)
  rede = LeakyReLU(alpha = 0.2)(rede)
  rede = Dropout(0.25)(rede)

  rede = Conv2D(128, kernel_size=(3,3), padding='same', strides=2)(rede)
  rede = BatchNormalization(momentum=0.8)(rede)
  rede = LeakyReLU(alpha = 0.2)(rede)
  rede = Dropout(0.25)(rede)

  rede = Conv2D(256, kernel_size=(3,3), padding='same', strides=1)(rede)
  rede = BatchNormalization(momentum=0.8)(rede)
  rede = LeakyReLU(alpha = 0.2)(rede)
  rede = Dropout(0.25)(rede)

  rede = Flatten()(rede)
  rede = Dense(1, activation='sigmoid')(rede) # não pode ter 'sigmoid' no modelo de Wasserstein

  model = Model(inputs=entrada, outputs=rede, name='discriminador')
  print( model.summary() )
  return model

def criarGAN():
  # otimizador = RMSprop(lr=0.00005) # Wasserstein
  otimizador = Adam(0.0002, 0.85, decay = 0.00001)
  entrada = Input(shape = (100,) )
  gerador = criarGerador()
  discriminador = criarDiscriminador()
  discriminador.compile(loss='binary_crossentropy', optimizer= otimizador, metrics=['accuracy'])
  discriminador.trainable = False # para não treinar o discriminador, ele apenas irá regular o gerador

  rede = gerador( entrada )
  rede = discriminador( rede )

  model = Model(inputs=entrada, outputs=rede)
  model.compile(loss='binary_crossentropy', optimizer= otimizador, metrics=['accuracy'])
  # print( model.summary() )
  return model, gerador, discriminador

def treinar(gan, gerador, discriminador, epochs = 1500, batch_size = 32):
  (x, y), (_, _) = mnist.load_data()
  print( x.shape )
  x = x[ y == 4 ]
  print( x.shape )
  x = x / 127.5 - 1 # para acelerar o treinamento
  x = np.expand_dims( x, axis=3)

  # respostas para o treinamento
  real = np.ones( (batch_size, 1) )
  fake = np.zeros( (batch_size, 1) )

  noise = np.random.normal( 0, 1, (batch_size, 100) )

  for epoch in range( epochs ):

    # treinamento de Wasserstein
    #for i in range(5):
    # pegar imagens para o treinamento
    idx = np.random.randint(0, x.shape[0], batch_size)
    imgs = x[ idx ]

    # criar imagens falsas
    
    fakeImgs = gerador.predict( noise )

    # treinar discriminador
    erro_real = discriminador.train_on_batch( imgs, real )
    erro_fake = discriminador.train_on_batch( fakeImgs, fake )
    erro_final = np.add( erro_real , erro_fake ) * 0.5

      # correção de pesos
      # for layer in discriminador.layers:
      #   weights = layer.get_weights()
      #   weights = [ np.clip(w, -0.01, 0.01) for w in weights ]
      #   layer.set_weights( weights )
      #   pass

    erro_gan = gan.train_on_batch( noise, real )
    print("Epoch %d   ( Disc. loss: %.3f , acc.: %.3f )   ( GAN loss: %.3f , acc.: %.3f )" % 
      ( epoch, erro_final[0], erro_final[1], erro_gan[0], erro_gan[1] ) )
    
    if epoch % 100 == 0:
      fazerImagens( gerador )
    pass

  pass

# testar rede
def fazerImagens(gerador):
  noise = np.random.normal( 0, 1, [3, 100] )
  res = gerador.predict( noise )
  # devolver para valor de cores
  res = np.array( res ) * 255
  res = res.astype('int8')

  for i in range( res.shape[0] ):
    item = res[ i ]
    img = Image.new('L', (28,28) )
    img.putdata( item.reshape( 28 * 28 ) )
    img.save('temp_gan_%d.jpg' % ( i ) )
    pass
# --------------------------

gan, gerador, discriminador = criarGAN()

treinar( gan, gerador, discriminador )

fazerImagens( gerador )
