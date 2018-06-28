from keras.layers import Lambda, Input, Dense
from keras.layers import Flatten, Conv1D, GRU, RepeatVector, TimeDistributed
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    eps = K.random_normal(shape=(batch,dim))
    return z_mean + K.exp(0.5 * z_log_var) * eps

X = np.load('./250k.npz')['arr']
X_train, X_test = train_test_split(X, random_state=0)
input_shape = (120, 35, )
intermediate_dim = 435
batch_size = 256
latent_dim = 292
epochs = 30


# encoder
inputs = Input(shape=input_shape, name='encoder_input')
x = Conv1D(9, 9, activation='relu')(inputs)
x = Conv1D(9, 9, activation='relu')(x)
x = Conv1D(10, 11, activation='relu')(x)
x = Flatten()(x)
x = Dense(intermediate_dim, activation='relu')(x)

z_mean = Dense(latent_dim, activation='linear', name='z_mean')(x)
z_log_var = Dense(latent_dim, activation='linear', name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim, ), name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# decoder
latent_inputs = Input(shape=(latent_dim, ), name='z_sampling')
x = RepeatVector(120)(latent_inputs)
x = GRU(501, return_sequences=True)(x)
x = GRU(501, return_sequences=True)(x)
x = GRU(501, return_sequences=True)(x)
outputs = TimeDistributed(Dense(35, activation='softmax'))(x)
decoder = Model(latent_inputs, outputs)

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs)

def vae_loss(x, x_decoded_mean):
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = binary_crossentropy(x, x_decoded_mean)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
    return xent_loss + kl_loss

vae.compile(optimizer='adam', loss=vae_loss, metrics=['acc'])
vae.fit(X_train, X_train, shuffle=True, epochs=epochs, batch_size=batch_size,
        validation_data=(X_test, X_test))
vae.save('./vae.h5')
