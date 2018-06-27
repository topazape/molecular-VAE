from keras.layers import Input, Dense, Flatten, Conv1D, Lambda
from keras.layers import GRU, RepeatVector, TimeDistributed
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model


inputs = Input(shape=(120, 35))
h = Conv1D(9, 9, activation='relu')(inputs)
h = Conv1D(9, 9, activation='relu')(h)
h = Conv1D(10, 11, activation='relu')(h)

h = Flatten()(h)
h = Dense(435, activation='relu')(h)

z_mean = Dense(292)(h)
z_log_var = Dense(292)(h)

def sampling(args):
    z_mean, z_log_var = args
    eps = K.random_normal(shape=(K.shape(z_mean)[0], 292), mean=0, stddev=0.01)
    return z_mean + K.exp(z_log_var) * eps

z = Lambda(sampling)([z_mean, z_log_var])

decoder_input = Input((292, ))
h = Dense(292, activation='relu')(decoder_input)
h = RepeatVector(120)(h)
h = GRU(501, return_sequences=True)(h)
h = GRU(501, return_sequences=True)(h)
h = GRU(501, return_sequences=True)(h)
h = TimeDistributed(Dense(35, activation='softmax'))(h)

decoder = Model(decoder_input, h)
z_decoded = decoder(z)

model = Model(inputs, y)

from featurizer import OneHotFeaturizer
ohf = OneHotFeaturizer()
with open('./5k.smi') as f:
    smiles = [smi.rstrip() for smi in f]
X = ohf.featurize(smiles)
model.fit(X, X)
