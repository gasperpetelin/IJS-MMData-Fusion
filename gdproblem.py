import numpy as np
import tensorflow as tf


#Tu se zna pojaviti težava, kjer se computational graph preveč
# napolni iz starimi informacijami in se program sesuje.
# Ta varianta dela hitreje kot, če bi v vsakem koraku izdelali nov session, vendar se občasno sesuje


class GDProblem:
    def __init__(self, m, n, k):
        self.m, self.n, self.k = m, n, k
        self.R_data = np.empty((m, n, n))
        for i in range(m):
            self.R_data[i, :, :] = np.loadtxt('data/R' + str(i + 1) + '.csv', delimiter=',')

        self.sess = tf.Session()
        self.tfR = tf.constant(self.R_data)
        self.tfR_norm = tf.reduce_sum(tf.pow(self.tfR, 2))
        self.sess.run(self.tfR)
        self.sess.run(self.tfR_norm)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        #print(n, m ,k)



        self.tfG1 = tf.Variable(np.zeros((n, k)))
        self.tfSS1 = tf.Variable(np.zeros((m, k, k)))
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.sess.run(self.tfG1)
        self.sess.run(self.tfSS1)

    def new_weights(self, G, Ss, steps=5):
        #assign_op_tfG = self.tfG1.assign(G)
        #assign_op_tfSs = self.tfSS1.assign(Ss)

        #self.sess.run(assign_op_tfG)
        #self.sess.run(assign_op_tfSs)


        tfG = tf.Variable(G)
        tfS = tf.Variable(Ss)

        GS = tf.tensordot(tf.abs(tfG), tf.abs(tfS), axes=[[1], [1]])
        Gt = tf.transpose(tfG)
        GSGt = tf.tensordot(GS, tf.abs(Gt), axes=[[2], [0]])
        R_GSGt = tf.subtract(self.tfR, tf.transpose(GSGt, perm=[1, 0, 2]))

        cost = tf.divide(tf.reduce_sum(tf.pow(R_GSGt, 2)), self.tfR_norm)


        train_step = self.optimizer.minimize(cost)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        c = self.sess.run([cost])
        #print("new cost0:", c)

        for i in range(steps):
            _, c = self.sess.run([train_step, cost])



        newG = self.sess.run([tfG])[0]
        newSs = self.sess.run([tfS])[0]

        #print("new cost:", c)
        #print("new cost2:", self.cost_grad(newG, newSs)[0])


        c = self.sess.run([cost])
        #tf.reset_default_graph()
        return c, newG, newSs

    def cost_grad(self, G, Ss):
        tfG = tf.Variable(G)
        tfS = tf.Variable(Ss)

        GS = tf.tensordot(tf.abs(tfG), tf.abs(tfS), axes=[[1], [1]])
        Gt = tf.transpose(tfG)
        GSGt = tf.tensordot(GS, Gt, axes=[[2], [0]])
        R_GSGt = tf.subtract(self.tfR, tf.transpose(GSGt, perm=[1, 0, 2]))

        result = tf.divide(tf.reduce_sum(tf.pow(R_GSGt, 2)), self.tfR_norm)

        init_op = tf.global_variables_initializer()

        self.sess.run(init_op)
        cost = self.sess.run(result)

        grad = tf.gradients(result, [tfG, tfS])

        grad_G = grad[0].eval(session=self.sess)
        grad_S = grad[1].eval(session=self.sess)

        return cost, grad_G, grad_S

    def close(self):
        self.sess.close()
