import tensorflow as tf
import numpy as np

from regularization import NoneRegulatrization


class GradientDescentOptimizer:
    def __init__(self, R_data, lr=0.01, beta_1=0.9, beta_2=0.999, eps=1e-8, regularization=NoneRegulatrization()):
        # Save the parameters of Adam gradient descent
        self.lr, self.beta_1, self.beta_2, self.eps = lr, beta_1, beta_2, eps
        self.R_data = R_data

        # Sum of square of Frobenius norm of matrices R_i
        self.fnR=np.sum(np.square(self.R_data))
        
        # Definition of tensorflow computational graph
        # Tensor definitions
        self.R=tf.constant(self.R_data,name='R',dtype=tf.float64)
        self.G=tf.Variable(initial_value=[],validate_shape=False,name='G',dtype=tf.float64)
        self.S=tf.Variable(initial_value=[],validate_shape=False,name='S',dtype=tf.float64)
        self.Gm=tf.Variable(initial_value=[],validate_shape=False,name='Gm',dtype=tf.float64)
        self.Gv=tf.Variable(initial_value=[],validate_shape=False,name='Gv',dtype=tf.float64)
        self.Sm=tf.Variable(initial_value=[],validate_shape=False,name='Sm',dtype=tf.float64)
        self.Sv=tf.Variable(initial_value=[],validate_shape=False,name='Sv',dtype=tf.float64)
        self.t=tf.Variable(initial_value=0.0,name='t',dtype=tf.float64)
        self.Gp=tf.placeholder(tf.float64)
        self.Sp=tf.placeholder(tf.float64)

        # Initialization of tensors
        self.assign_G=tf.assign(self.G,self.Gp,validate_shape=False)
        self.assign_S=tf.assign(self.S,self.Sp,validate_shape=False)
        self.assign_Gm=tf.assign(self.Gm,tf.zeros_like(self.Gp),validate_shape=False)
        self.assign_Gv=tf.assign(self.Gv,tf.zeros_like(self.Gp),validate_shape=False)
        self.assign_Sm=tf.assign(self.Sm,tf.zeros_like(self.Sp),validate_shape=False)
        self.assign_Sv=tf.assign(self.Sv,tf.zeros_like(self.Sp),validate_shape=False)
        self.assign_t=tf.assign(self.t,0.0)
        self.new_cost=tf.group(self.assign_G,self.assign_S)
        self.new_descent=tf.group(self.assign_G,self.assign_S,
                                  self.assign_Gm,self.assign_Gv, \
                                  self.assign_Sm,self.assign_Sv,self.assign_t)
        # Cost calculation
        self.G_abs = tf.abs(self.G)
        self.S_abs = tf.abs(self.S)

        self.GS=tf.tensordot(self.G_abs, self.S_abs, axes=[[1],[1]])
        self.GSGt=tf.tensordot(self.GS,tf.abs(self.G),axes=[[2],[1]])
        self.R_GSGt=tf.subtract(self.R,tf.transpose(self.GSGt,perm=[1,0,2]))
        self.cost=tf.divide(tf.reduce_sum(tf.pow(self.R_GSGt,2)),self.fnR)

        self.regularized_cost = self.cost + regularization.add_regularization(self.R, self.G_abs, self.S_abs)

        # Step of gradient descent using Adam
        self.g_G,self.g_S=tf.gradients(self.regularized_cost,[self.G,self.S])
        self.newt=tf.assign(self.t,self.t+1)
        with tf.control_dependencies([self.newt]):
            self.alpha_t=self.lr*tf.sqrt(1.-self.beta_2**self.t)/(1.-self.beta_1**self.t)
            self.newGm=tf.assign(self.Gm,self.beta_1*self.Gm+(1.-self.beta_1)*self.g_G)
            self.newSm=tf.assign(self.Sm,self.beta_1*self.Sm+(1.-self.beta_1)*self.g_S)
            self.newGv=tf.assign(self.Gv,self.beta_2*self.Gv+(1.-self.beta_2)*self.g_G**2)
            self.newSv=tf.assign(self.Sv,self.beta_2*self.Sv+(1.-self.beta_2)*self.g_S**2)
            with tf.control_dependencies([self.newGm,self.newSm,self.newGv,self.newSv]):
                self.newG=tf.assign(self.G,self.G-self.alpha_t*self.Gm/(tf.sqrt(self.Gv)+self.eps))
                self.newS=tf.assign(self.S,self.S-self.alpha_t*self.Sm/(tf.sqrt(self.Sv)+self.eps))
        self.new_step=tf.group(self.newGm,self.newSm,self.newGv,self.newSv,self.newG,self.newS,self.newt)
        # Start tensorflow session
        self.sess=tf.Session()

    # Function that does gradient descent
    def adam(self, npG, npS, steps):
        self.sess.run(self.new_descent,feed_dict={self.Gp: npG, self.Sp: npS})
        for i in range(steps):
            self.sess.run(self.new_step)
            #c = self.sess.run(self.cost)
        c=self.sess.run(self.cost)
        npG_new,npS_new=self.sess.run([self.G,self.S])
        return c,np.abs(npG_new),np.abs(npS_new)

    # Function that calculates the cost
    def calculate_cost(self,npG,npS):
        self.sess.run(self.new_cost,feed_dict={self.Gp: npG, self.Sp: npS})
        c=self.sess.run(self.cost)
        return c

    # Method that closes tensorflow session
    def close(self):
        self.sess.close()
