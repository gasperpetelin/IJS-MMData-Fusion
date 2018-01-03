from __future__ import print_function
import pygmo as pg
import numpy as np
import timeit
# from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from gdproblem import GDProblem
from platypus import NSGAII, Problem, Real, UM, AttributeDominance, fitness_key

from gradient_mutation import TestMut, TestMut2, IBEA, TournamentSelectorMod
from stohastic_universal_sampling import SUS

m = 5
n = 100
k = 10

p = GDProblem(m, n, k)
G = np.random.rand(n, k)
S = np.random.rand(m, k, k)

def schaffer(x):
    x = np.array(x)
    G = x[0:n * k].reshape((n, k))
    Ss = x[(n * k):].reshape((m, k, k))
    cost, gradG, gradS = p.cost_grad(G, Ss)
    print("eval_print", cost)
    #p.new_weights(G, Ss)
    return [cost]

problem = Problem(n * k + k * k * m, 1)
problem.types[:] = Real(0, 1)
problem.function = schaffer



#sel = TournamentSelectorMod(2, AttributeDominance(fitness_key, False))
sel = SUS(2, AttributeDominance(fitness_key, False))

algorithm = IBEA(problem, selector=sel,  variator=TestMut(n, k, m, p, probability=0.3), population_size = 20,)
algorithm.run(1000)




#def schaffer(x):
#    x = np.array(x)
#
#    G = x[0:n * k].reshape((n, k))
#    Ss = x[(n * k):].reshape((m, k, k))
#
#    cost, gradG, gradS = p.cost_grad(G, Ss)
#    return [cost]
#
#
#problem = Problem(n * k + k * k * m, 1)
#problem.types[:] = Real(-10000, 10000)
#problem.function = schaffer
#
#algorithm = NSGAII(problem, variator=TestMut(n, k, m, p))
#algorithm.run(1000)

#print("res:", len(algorithm.result))

# fmin_l_bfgs_b()


#start = timeit.default_timer()
#for _ in range(30):
#    cost, gradG, gradS = p.cost_grad(G, S)
#    flt_g = gradG.flatten()
#    flt_s = gradS.flatten()
#    con = np.concatenate((flt_g, flt_s), axis=0)
#
#    G1 = con[0:n * k].reshape((n, k))
#    G2 = con[(n * k):].reshape((m, k, k))
#    lr = 0.003
#    G = G - lr * G1
#    S = S - lr * G2
#
#stop = timeit.default_timer()
#print("time", stop - start)


# G = np.random.rand(n, k)
# S = np.random.rand(m, k, k)
# x1 = G.flatten()
# x2 = S.flatten()
# x0 = np.concatenate((x1, x2), axis=0)
# print("fmin_l_bfgs_b")
# def value(x):
#    G = x[0:n*k].reshape((n,k))
#    S = x[(n*k):].reshape((m, k, k))
#
#    cost, gradG, gradS = p.cost_grad(G, S)
#    flt_g = gradG.flatten()
#    flt_s = gradS.flatten()
#    con = np.concatenate((flt_g, flt_s), axis=0)
#    return cost, con

# w = fmin_l_bfgs_b(value, x0=x0, fprime=None, maxfun=1)[0]
# for _ in range(10):
#    w = fmin_l_bfgs_b(value, x0=w, fprime=None, maxfun=1, maxls=1)[0]
class py_rosenbrock:
    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
        retval = np.zeros((1,))
        for i in range(len(x) - 1):
            retval[0] += 100. * (x[i + 1] - x[i] ** 2) ** 2 + (1. - x[i]) ** 2
        return retval

    def get_bounds(self):
        return (np.full((self.dim,), -5.), np.full((self.dim,), 10.))

    def cross(self, x1, x2):
        fgj = 34


prob_python = pg.problem(py_rosenbrock(2000))
prob_cpp = pg.problem(pg.rosenbrock(2000))
dummy_x = np.full((2000,), 1.)
import time

start_time = time.time();
[prob_python.fitness(dummy_x) for i in range(1000)];
# print(time.time() - start_time)
start_time = time.time();
[prob_cpp.fitness(dummy_x) for i in range(1000)];
# print(time.time() - start_time)

# 1 - Instantiate a pygmo problem constructing it from a UDP
# (user defined problem).
# prob = pg.problem(pg.schwefel(30))
#
## 2 - Instantiate a pagmo algorithm
# algo = pg.algorithm(pg.sade(gen=100))
#
## 3 - Instantiate an archipelago with 16 islands having each 20 individuals
# archi = pg.archipelago(16, algo=algo, prob=prob, pop_size=20)
#
## 4 - Run the evolution in parallel on the 16 separate islands 10 times.
# archi.evolve(10)
#
## 5 - Wait for the evolutions to be finished
# archi.wait()
#
## 6 - Print the fitness of the best solution in each island
# res = [isl.get_population().champion_f for isl in archi]
# print(res)
