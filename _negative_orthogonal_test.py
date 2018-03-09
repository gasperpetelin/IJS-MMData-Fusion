from regularization import NoneRegulatrization
from util import read_matrices
import numpy as np
import tensorflow as tf

class GradientDescentOptimizer:
    def __init__(self, R_data, lr=0.01, beta_1=0.9, beta_2=0.999, eps=1e-8, regularization=NoneRegulatrization()):
        # Save the parameters of Adam gradient descent
        self.lr, self.beta_1, self.beta_2, self.eps = lr, beta_1, beta_2, eps
        self.R_data = R_data

        # Sum of square of Frobenius norm of matrices R_i
        self.fnR = np.sum(np.square(self.R_data))

        # Definition of tensorflow computational graph
        # Tensor definitions
        self.R = tf.constant(self.R_data, name='R', dtype=tf.float64)
        self.G = tf.Variable(initial_value=[], validate_shape=False, name='G', dtype=tf.float64)
        self.S = tf.Variable(initial_value=[], validate_shape=False, name='S', dtype=tf.float64)
        self.Gm = tf.Variable(initial_value=[], validate_shape=False, name='Gm', dtype=tf.float64)
        self.Gv = tf.Variable(initial_value=[], validate_shape=False, name='Gv', dtype=tf.float64)
        self.Sm = tf.Variable(initial_value=[], validate_shape=False, name='Sm', dtype=tf.float64)
        self.Sv = tf.Variable(initial_value=[], validate_shape=False, name='Sv', dtype=tf.float64)
        self.t = tf.Variable(initial_value=0.0, name='t', dtype=tf.float64)
        self.Gp = tf.placeholder(tf.float64)
        self.Sp = tf.placeholder(tf.float64)

        # Initialization of tensors
        self.assign_G = tf.assign(self.G, self.Gp, validate_shape=False)
        self.assign_S = tf.assign(self.S, self.Sp, validate_shape=False)
        self.assign_Gm = tf.assign(self.Gm, tf.zeros_like(self.Gp), validate_shape=False)
        self.assign_Gv = tf.assign(self.Gv, tf.zeros_like(self.Gp), validate_shape=False)
        self.assign_Sm = tf.assign(self.Sm, tf.zeros_like(self.Sp), validate_shape=False)
        self.assign_Sv = tf.assign(self.Sv, tf.zeros_like(self.Sp), validate_shape=False)
        self.assign_t = tf.assign(self.t, 0.0)
        self.new_cost = tf.group(self.assign_G, self.assign_S)
        self.new_descent = tf.group(self.assign_G, self.assign_S,
                                    self.assign_Gm, self.assign_Gv, \
                                    self.assign_Sm, self.assign_Sv, self.assign_t)
        # Cost calculation
        self.G_abs = tf.abs(self.G)
        self.S_abs = tf.abs(self.S)

        self.GS = tf.tensordot(self.G_abs, self.S_abs, axes=[[1], [1]])
        self.GSGt = tf.tensordot(self.GS, tf.abs(self.G), axes=[[2], [1]])
        self.R_GSGt = tf.subtract(self.R, tf.transpose(self.GSGt, perm=[1, 0, 2]))
        self.cost = tf.divide(tf.reduce_sum(tf.pow(self.R_GSGt, 2)), self.fnR)

        self.regularized_cost = self.cost + regularization.add_regularization(self.R, self.G_abs, self.S_abs)

        # Step of gradient descent using Adam
        self.g_G, self.g_S = tf.gradients(self.regularized_cost, [self.G, self.S])
        self.newt = tf.assign(self.t, self.t + 1)
        with tf.control_dependencies([self.newt]):
            self.alpha_t = self.lr * tf.sqrt(1. - self.beta_2 ** self.t) / (1. - self.beta_1 ** self.t)
            self.newGm = tf.assign(self.Gm, self.beta_1 * self.Gm + (1. - self.beta_1) * self.g_G)
            self.newSm = tf.assign(self.Sm, self.beta_1 * self.Sm + (1. - self.beta_1) * self.g_S)
            self.newGv = tf.assign(self.Gv, self.beta_2 * self.Gv + (1. - self.beta_2) * self.g_G ** 2)
            self.newSv = tf.assign(self.Sv, self.beta_2 * self.Sv + (1. - self.beta_2) * self.g_S ** 2)
            with tf.control_dependencies([self.newGm, self.newSm, self.newGv, self.newSv]):
                self.newG = tf.assign(self.G, self.G - self.alpha_t * self.Gm / (tf.sqrt(self.Gv) + self.eps))
                self.newS = tf.assign(self.S, self.S - self.alpha_t * self.Sm / (tf.sqrt(self.Sv) + self.eps))
        self.new_step = tf.group(self.newGm, self.newSm, self.newGv, self.newSv, self.newG, self.newS, self.newt)
        # Start tensorflow session
        self.sess = tf.Session()

    # Function that does gradient descent
    def adam(self, npG, npS, steps):
        self.sess.run(self.new_descent, feed_dict={self.Gp: npG, self.Sp: npS})

        #Only find best G and S
        best_cost = 10**8
        best_G = None
        best_S = None

        for i in range(steps):
            self.sess.run(self.new_step)
            cur_cost = self.sess.run(self.cost)

            if cur_cost<best_cost:
                best_cost = cur_cost
                best_G, best_S = self.sess.run([self.G_abs, self.S_abs])

            print(cur_cost, best_cost)
        c = self.sess.run(self.cost)
        return c, best_G, best_S
        #npG_new, npS_new = self.sess.run([self.G, self.S])
        #return c, np.abs(npG_new), np.abs(npS_new)


    # Function that calculates the cost
    def calculate_cost(self, npG, npS):
        self.sess.run(self.new_cost, feed_dict={self.Gp: npG, self.Sp: npS})
        c = self.sess.run(self.cost)
        return c

    # Method that closes tensorflow session
    def close(self):
        self.sess.close()


class GradientDescentMaskOptimizer:
    def __init__(self, R_data, lr=0.01, beta_1=0.9, beta_2=0.999, eps=1e-8, regularization=NoneRegulatrization()):
        # Save the parameters of Adam gradient descent
        self.lr, self.beta_1, self.beta_2, self.eps = lr, beta_1, beta_2, eps
        self.R_data = R_data

        # Sum of square of Frobenius norm of matrices R_i
        self.fnR = np.sum(np.square(self.R_data))

        # Definition of tensorflow computational graph
        # Tensor definitions
        self.R = tf.constant(self.R_data, name='R', dtype=tf.float64)
        self.G = tf.Variable(initial_value=[], validate_shape=False, name='G', dtype=tf.float64)
        self.S = tf.Variable(initial_value=[], validate_shape=False, name='S', dtype=tf.float64)
        self.Gm = tf.Variable(initial_value=[], validate_shape=False, name='Gm', dtype=tf.float64)
        self.Gv = tf.Variable(initial_value=[], validate_shape=False, name='Gv', dtype=tf.float64)
        self.Sm = tf.Variable(initial_value=[], validate_shape=False, name='Sm', dtype=tf.float64)
        self.Sv = tf.Variable(initial_value=[], validate_shape=False, name='Sv', dtype=tf.float64)
        self.t = tf.Variable(initial_value=0.0, name='t', dtype=tf.float64)
        self.Gp = tf.placeholder(tf.float64)
        self.Gp_orthogonality = tf.placeholder(tf.float64, name="G_orthogonal")
        self.Sp = tf.placeholder(tf.float64)

        # Initialization of tensors
        self.assign_G = tf.assign(self.G, self.Gp, validate_shape=False)
        self.assign_S = tf.assign(self.S, self.Sp, validate_shape=False)
        #self.assign_G_mask = tf.assign(self.G_orthogonality_mask, self.Gp_orthogonality, validate_shape=False)
        self.assign_Gm = tf.assign(self.Gm, tf.zeros_like(self.Gp), validate_shape=False)
        self.assign_Gv = tf.assign(self.Gv, tf.zeros_like(self.Gp), validate_shape=False)
        self.assign_Sm = tf.assign(self.Sm, tf.zeros_like(self.Sp), validate_shape=False)
        self.assign_Sv = tf.assign(self.Sv, tf.zeros_like(self.Sp), validate_shape=False)
        self.assign_t = tf.assign(self.t, 0.0)
        self.new_cost = tf.group(self.assign_G, self.assign_S)#, self.Gp_orthogonality)
        self.new_descent = tf.group(self.assign_G, self.assign_S,#, #self.Gp_orthogonality,
                                    self.assign_Gm, self.assign_Gv, \
                                    self.assign_Sm, self.assign_Sv, self.assign_t)
        # Cost calculation
        self.G_orthogonal = tf.multiply(self.G, self.Gp_orthogonality)

        self.G_abs = tf.abs(self.G)
        self.S_abs = tf.abs(self.S)

        self.GS = tf.tensordot(self.G_abs, self.S_abs, axes=[[1], [1]])
        self.GSGt = tf.tensordot(self.GS, tf.abs(self.G), axes=[[2], [1]])
        self.R_GSGt = tf.subtract(self.R, tf.transpose(self.GSGt, perm=[1, 0, 2]))
        self.cost = tf.divide(tf.reduce_sum(tf.pow(self.R_GSGt, 2)), self.fnR)

        self.regularized_cost = self.cost + regularization.add_regularization(self.R, self.G_abs, self.S_abs)

        # Step of gradient descent using Adam
        self.g_G, self.g_S = tf.gradients(self.regularized_cost, [self.G, self.S])
        self.newt = tf.assign(self.t, self.t + 1)
        with tf.control_dependencies([self.newt]):
            self.alpha_t = self.lr * tf.sqrt(1. - self.beta_2 ** self.t) / (1. - self.beta_1 ** self.t)
            self.newGm = tf.assign(self.Gm, self.beta_1 * self.Gm + (1. - self.beta_1) * self.g_G)
            self.newSm = tf.assign(self.Sm, self.beta_1 * self.Sm + (1. - self.beta_1) * self.g_S)
            self.newGv = tf.assign(self.Gv, self.beta_2 * self.Gv + (1. - self.beta_2) * self.g_G ** 2)
            self.newSv = tf.assign(self.Sv, self.beta_2 * self.Sv + (1. - self.beta_2) * self.g_S ** 2)
            with tf.control_dependencies([self.newGm, self.newSm, self.newGv, self.newSv]):
                #self.newG = tf.assign(self.G, self.G - self.alpha_t * self.Gm / (tf.sqrt(self.Gv) + self.eps))
                self.newG = tf.assign(self.G, self.G -  tf.multiply(self.Gp_orthogonality, self.alpha_t * self.Gm / (tf.sqrt(self.Gv) + self.eps)))
                self.newS = tf.assign(self.S, self.S - self.alpha_t * self.Sm / (tf.sqrt(self.Sv) + self.eps))
        self.new_step = tf.group(self.newGm, self.newSm, self.newGv, self.newSv, self.newG, self.newS, self.newt)
        # Start tensorflow session
        self.sess = tf.Session()

    # Function that does gradient descent
    def adam(self, npG, npG_mask, npS, steps):
        self.sess.run(self.new_descent, feed_dict={self.Gp: npG, self.Sp: npS})
        for i in range(steps):
            self.sess.run(self.new_step, feed_dict={self.Gp_orthogonality: npG_mask})
            cur_cost = self.sess.run(self.cost)
            print(cur_cost)
        c = self.sess.run(self.cost, feed_dict={self.Gp_orthogonality: npG_mask})
        npG_new, npS_new = self.sess.run([self.G, self.S])

        npG_new_new = np.copy(npG)
        npG_new_new[npG_mask>0] = npG_new[npG_mask>0]
        m = np.abs(npG_new_new-npG)

        return c, np.abs(npG_new_new), np.abs(npS_new)

    # Function that calculates the cost
    def calculate_cost(self, npG, npG_mask, npS):
        self.sess.run(self.new_cost, feed_dict={self.Gp: npG, self.Sp: npS})
        #n = self.sess.run(self.G_orthogonal, feed_dict={self.Gp_orthogonality: npG_mask})
        c = self.sess.run(self.cost, feed_dict={self.Gp_orthogonality: npG_mask})

        self.sess.run(self.new_cost, feed_dict={self.Gp: npG*npG_mask, self.Sp: npS})
        #n1 = self.sess.run(self.G, feed_dict={self.Gp_orthogonality: npG_mask})
        c1 = self.sess.run(self.cost, feed_dict={self.Gp_orthogonality: npG_mask})
        return c

    # Method that closes tensorflow session
    def close(self):
        self.sess.close()


directory = 'data/test_data_orthogonal_n=100_k=20N=5_density=10.0/'
matrices = ['R1.csv','R2.csv','R3.csv','R4.csv','R5.csv']
n = 100

Ri = read_matrices(directory, matrices, n)
p = GradientDescentOptimizer(Ri, lr=0.07)


k = 20

c, npG_new, npS_new = p.adam(np.random.rand(n,k), np.random.rand(5, k,k), 10000)
p.close()



def norm(G, S):
    g_column_sum = G.sum(axis=0, keepdims=True)

    newG = (G.T/g_column_sum.T).T
    Sn = np.zeros(S.shape)
    for i, s in enumerate(S):
        Sn[i,:,:] = ((s*g_column_sum).T * g_column_sum).T

    return newG, Sn

Gn, Sn = norm(npG_new, npS_new)

#print(Gn.sum(axis=0))

cost_calc = GradientDescentOptimizer(Ri)
print(cost_calc.calculate_cost(Gn, Sn))



import matplotlib.pyplot as plt
plt.imshow(npG_new)
plt.show()





#arg_max = np.argmax(npG_new, 1)
#mask = np.zeros(npG_new.shape)
#for i in range(npG_new.shape[0]):
#    mask[i, arg_max[i]] = 1

def binarize_max(M):
    arg_max = np.argmax(M, 1)
    mask = np.zeros(M.shape)
    for i in range(M.shape[0]):
       mask[i, arg_max[i]] = 1
    return mask

G = read_matrices(directory, ['G.csv'], n)[0]
G_original_mask = binarize_max(G[:, 0:20])

G_generated_mask = binarize_max(npG_new)

#import matplotlib.pyplot as plt
#plt.subplot(121)
#plt.imshow(G_generated_mask)
#plt.subplot(122)
#plt.imshow(G_original_mask)
#plt.show()


order = []
for column in G_original_mask.T:

    best_cost = 0
    best_column = None

    for i, column_gen in enumerate(G_generated_mask.T):
        cost = np.sum(column*column_gen)/np.sum(column+column_gen)
        if cost > best_cost:
            best_cost = cost
            best_column = i
    order.append(best_column)

print(order)

#[npG_new[i,:] for i in order]

#lucy_try = np.column_stack(order)
lucy_try = np.column_stack([npG_new[:,i] for i in order])
#print(lucy_try.shape)

import matplotlib.pyplot as plt
plt.subplot(121)
plt.imshow(lucy_try)
plt.subplot(122)
plt.imshow(G[:, 0:20])
plt.show()



print(lucy_try)


def orthogonalize(M):
    arg_max = np.argmax(M, 1)
    new_M = np.zeros(M.shape)
    for i in range(M.shape[0]):
       new_M[i, arg_max[i]] = M[i, arg_max[i]]
    return new_M

lucy_try_1 = orthogonalize(npG_new)


p = GradientDescentOptimizer(Ri, lr=0.07)
print(p.calculate_cost(lucy_try_1, npS_new))
p.close()



po = GradientDescentMaskOptimizer(Ri, lr=0.02)
msk = binarize_max(lucy_try_1)

c_f, npG_new_f, npS_new_f = po.adam(lucy_try_1, msk, npS_new, 500)

print(npG_new_f)

