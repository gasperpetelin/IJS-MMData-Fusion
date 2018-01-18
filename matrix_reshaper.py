import numpy as np

class Reshape:
    def __init__(self, m, n, k):
        self.m = m
        self.n = n
        self.k = k

    def mat2vec(self, G, Ss):
        return np.concatenate((G.flatten(), Ss.flatten()), axis=0)

    def vec2mat(self, x):
        x = np.array(x)
        G = x[0:self.n * self.k].reshape((self.n, self.k))
        Ss = x[(self.n * self.k):].reshape((self.m, self.k, self.k))
        return G, Ss

