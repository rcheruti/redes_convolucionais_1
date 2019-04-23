
import sys
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

numero = sys.argv[ 1 ] if len(sys.argv) > 1 else 0

# --------------------------
# carregar modelo do keras HDF5
model = load_model('redes/mnist_autoencoder_numero.h5')
print( model.summary() )
modelImg = load_model('redes/mnist_decoder.h5')

print('Número: ' + str( numero ) )
resArr = model.predict( to_categorical( [numero], 10 ) )
res = modelImg.predict( resArr )


# devolver para valor de cores
res = np.array( res ) * 255
res = res.astype('int8')

img = Image.new('L', (28,28) )
img.putdata( res.reshape( 28 * 28 ) )
img.save('temp.jpg')

