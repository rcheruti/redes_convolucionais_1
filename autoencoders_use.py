
import numpy as np
from PIL import Image
from keras.models import load_model

# --------------------------
# carregar modelo do keras HDF5
model = load_model('redes/mnist_decoder_1.h5')
print( model.summary() )

res = model.predict( np.array([ 0,0,0,0,0  ,  0,0,0,0,1 ]).reshape( 1, 10 ) ) # criar numero 6

# devolver para valor de cores
res = np.array( res ) * 255
res = res.astype('int8')

img = Image.new('L', (28,28) )
img.putdata( res.reshape( 28 * 28 ) )
img.save('temp.jpg')

