from platypus import Problem
from functools import partial

from abc import ABCMeta


class Tensor(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Tensor, self).__init__()

    def rand(self):
        raise NotImplementedError("method not implemented")

    def encode(self, value):
        return value

    def decode(self, value):
        return value

class Tri_Factorization(Problem):
    def __init__(self, gd_optimizer):

        # The objective function
        def obj_fun(cost_evaluator, x):
            G = x[0]
            S = x[1]
            k = S.shape[1]
            cost = cost_evaluator.calculate_cost(G, S)
            return [cost, k]

        # Partial application of argument cost_evaluator
        obj_fun_partial = partial(obj_fun, gd_optimizer)
        super().__init__(2,2,function=obj_fun_partial)

        #Set type of variables
        self.types[:] = Tensor()
        self.gdOptimizer = gd_optimizer

class Tri_Factorization_obj_1(Problem):
    def __init__(self, gd_optimizer):

        # The objective function
        def obj_fun(cost_evaluator, x):
            G = x[0]
            S = x[1]
            k = S.shape[1]
            cost = cost_evaluator.calculate_cost(G, S)
            return [cost]

        # Partial application of argument cost_evaluator
        obj_fun_partial = partial(obj_fun, gd_optimizer)
        super().__init__(2,1,function=obj_fun_partial)

        #Set type of variables
        self.types[:] = Tensor()
        self.gdOptimizer = gd_optimizer


