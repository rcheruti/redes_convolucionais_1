
import numpy as np
from PIL import Image
from keras.applications.inception_v3 import InceptionV3, decode_predictions

img = Image.open( 'carro.png' ).convert('RGB')
dados = np.array( list( img.getdata() ) ).reshape(1, 299, 299, 3).astype('float32') / 255.0

print('-------------------------')
print('Carregando InceptionV3')
inception = InceptionV3()
res = inception.predict( dados )

print('-------------------------')
print( 'Shape da resposta: ' + str( res.shape ) )
idx = np.argmax( res[0] )
print( 'Maior index: ' + str( idx ) )
print( 'Resposta: ' + str( decode_predictions( res, top = 3 ) ) )

