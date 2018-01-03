import numpy as np
import tensorflow as tf
from tensorflow.python.training import optimizer
from random import randint


class GDProblem:
    def __init__(self, m, n, k):
        self.m, self.n, self.k = m, n, k
        self.R_data = np.empty((m, n, n))
        for i in range(m):
            self.R_data[i, :, :] = np.loadtxt('data/R' + str(i + 1) + '.csv', delimiter=',')

        self.sess = tf.Session()
        self.tfR = tf.constant(self.R_data)
        self.sess.run(self.tfR)

    def new_weights(self, G, Ss, steps=5):
        tfG = tf.Variable(G)
        tfS = tf.Variable(Ss)

        GS = tf.tensordot(tf.abs(tfG), tf.abs(tfS), axes=[[1], [1]])
        Gt = tf.transpose(tfG)
        GSGt = tf.tensordot(GS, tf.abs(Gt), axes=[[2], [0]])
        R_GSGt = tf.subtract(self.tfR, tf.transpose(GSGt, perm=[1, 0, 2]))

        cost = tf.divide(tf.reduce_sum(tf.pow(R_GSGt, 2)), tf.reduce_sum(tf.pow(self.tfR, 2)))

        #global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(0.2, global_step, 1000, 0.87, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.007)
        train_step = optimizer.minimize(cost)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        for i in range(steps):
            _, c = self.sess.run([train_step, cost])

        newG = self.sess.run([tfG])[0]
        newSs = self.sess.run([tfS])[0]

        return c, newG, newSs


    def cost_grad(self, G, Ss):
        tfG = tf.Variable(G)
        tfS = tf.Variable(Ss)

        # tfR = tf.constant(self.R_data)


        GS = tf.tensordot(tfG, tfS, axes=[[1], [1]])
        Gt = tf.transpose(tfG)
        GSGt = tf.tensordot(GS, Gt, axes=[[2], [0]])
        R_GSGt = tf.subtract(self.tfR, tf.transpose(GSGt, perm=[1, 0, 2]))

        result = tf.divide(tf.reduce_sum(tf.pow(R_GSGt, 2)), tf.reduce_sum(tf.pow(self.tfR, 2)))

        init_op = tf.global_variables_initializer()

        self.sess.run(init_op)
        cost = self.sess.run(result)

        grad = tf.gradients(result, [tfG, tfS])

        grad_G = grad[0].eval(session=self.sess)
        grad_S = grad[1].eval(session=self.sess)

        return cost, grad_G, grad_S

    def close(self):
        self.sess.close()
