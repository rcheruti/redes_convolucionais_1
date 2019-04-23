
import sys
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

# numero = sys.argv[ 1 ] if len(sys.argv) > 1 else 0

# --------------------------
# carregar modelo do keras HDF5
model = load_model('redes/mnist_autoencoder.h5')
print( model.summary() )

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train =   x_train.reshape( x_train.shape[0], x_train.shape[1], x_train.shape[2], 1 )
x_test =    x_test.reshape( x_test.shape[0], x_test.shape[1], x_test.shape[2], 1 )
x_train =   x_train.astype('float32') / 255
x_test =    x_test.astype('float32') / 255
#y_train =   to_categorical(y_train, 10)
#y_test =    to_categorical(y_test, 10)

index = 28
print('Número: ' + str( y_test[ index ] ) )
res = model.predict( np.array( [ x_test[ index ] ] ) )

# devolver para valor de cores
res = np.array( res ) * 255
res = res.astype('int8')

img = Image.new('L', (28,28) )
img.putdata( res.reshape( 28 * 28 ) )
img.save('temp.jpg')

