import keras
from keras.layers import Input, Dense, Flatten, Conv1D, Lambda
from keras.layers import GRU, RepeatVector, TimeDistributed, Layer
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model


input_vec = Input(shape=(120, 35))
x = Conv1D(9, 9, activation='relu')(input_vec)
x = Conv1D(9, 9, activation='relu')(x)
x = Conv1D(10, 11, activation='relu')(x)

x = Flatten()(x)
x = Dense(435, activation='relu')(x)

z_mean = Dense(292)(x)
z_log_var = Dense(292)(x)

def sampling(args):
    z_mean, z_log_var = args
    eps = K.random_normal(shape=(K.shape(z_mean)[0], 292), mean=0, stddev=0.01)
    return z_mean + K.exp(z_log_var) * eps

z = Lambda(sampling)([z_mean, z_log_var])

decoder_input = Input((292, ))
x = Dense(292, activation='relu')(decoder_input)
x = RepeatVector(120)(x)
x = GRU(501, return_sequences=True)(x)
x = GRU(501, return_sequences=True)(x)
x = GRU(501, return_sequences=True)(x)
x = TimeDistributed(Dense(35, activation='softmax'))(x)

decoder = Model(decoder_input, x)
z_decoded = decoder(z)

class CustomVariationalLayer(Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xen_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
                1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        return K.mean(xen_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

y = CustomVariationalLayer()([input_vec, z_decoded])
vae = Model(input_vec, y)
vae.compile(optimizer='rmsprop', loss=None)
print(vae.summary())
plot_model(vae, to_file='model.png', show_shapes=True)


from featurizer import OneHotFeaturizer
ohf = OneHotFeaturizer()
with open('./5k.smi') as f:
    smiles = [smi.rstrip() for smi in f]
X = ohf.featurize(smiles)

vae.fit(X, y=None, shuffle=True, epochs=10, batch_size=32)
