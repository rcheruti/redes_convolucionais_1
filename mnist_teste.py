
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image
from keras.models import load_model

# carregar modelo do keras HDF5
model = load_model('redes/mnist.h5')
print( model.summary() )

caminho = 'imgs'
for f in listdir( caminho ):
  img = Image.open( join( caminho, f ) ).convert('L')
  dados = np.array( list( img.getdata() ) ).reshape(1, 28, 28, 1).astype('float32') / 255
  res = model.predict( dados )
  print( 'Resposta: ' + str( np.argmax(res[0]) )+ ' , Image: ' + str(f) )
  pass

