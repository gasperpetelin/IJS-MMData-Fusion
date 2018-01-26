import tensorflow as tf

class AngleRegulatrization:
    def __init__(self, _lambda):
        self._lambda = _lambda

    def add_regularization(self, R, G, S):
        # Dot products of pairs
        C = tf.matmul(G, tf.transpose(G))
        L = tf.sqrt(tf.diag_part(C))

        # Lengths
        LTL = tf.multiply(tf.transpose(L), L)

        # Angles between columns
        D = tf.divide(C, LTL)

        # Mean on angles
        return self._lambda * tf.reduce_mean(tf.matrix_band_part(D, 0, -1))

class NoneRegulatrization:
    def add_regularization(self, R, G, S):
        return 0