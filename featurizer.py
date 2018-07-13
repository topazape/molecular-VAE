import numpy as np

CHARSET = [' ', '#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7',
        '8', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\', ']',
        'c', 'l', 'n', 'o', 'r', 's']

class OneHotFeaturizer(object):
    def __init__(self, charset=CHARSET, padlength=120):
        self.charset = CHARSET
        self.pad_length = padlength

    def featurize(self, smiles):
        return np.array([self.one_hot_encode(smi) for smi in smiles])

    def one_hot_array(self, i):
        return [int(x) for x in [ix == i for ix in range(len(self.charset))]]

    def one_hot_index(self, c):
        return self.charset.index(c)

    def pad_smi(self, smi):
        return smi.ljust(self.pad_length)

    def one_hot_encode(self, smi):
        return np.array([
            self.one_hot_array(self.one_hot_index(x)) for x in self.pad_smi(smi)
            ])

    def one_hot_decode(self, z):
        z1 = []
        for i in range(len(z)):
            s = ''
            for j in range(len(z[i])):
                oh = np.argmax(z[i][j])
                s += self.charset[oh]
            z1.append([s.strip()])
        return z1

    def decode_smiles_from_index(self, vec):
        return ''.join(map(lambda x: CHARSET[x], vec)).strip()
