from __future__ import print_function

import numpy as np
from platypus import Problem, Real, AttributeDominance, fitness_key, DifferentialEvolution, TournamentSelector, PM, \
    UniformMutation

from gdproblem import GDProblem
from gradient_mutation import TestMut, IBEA

m = 5
n = 100
k = 10

p = GDProblem(m, n, k)


def schaffer(x):
    x = np.array(x)
    G = x[0:n * k].reshape((n, k))
    Ss = x[(n * k):].reshape((m, k, k))
    cost, gradG, gradS = p.cost_grad(G, Ss)
    print("eval_print", cost)
    return [cost]


problem = Problem(n * k + k * k * m, 1)
problem.types[:] = Real(0, 1)
problem.function = schaffer

sel = TournamentSelector(2, AttributeDominance(fitness_key, False))
# sel = SUS(2, AttributeDominance(fitness_key, False))

# algorithm = IBEA(problem, selector=sel,  variator=TestMut(n, k, m, p, probability=1), population_size = 7,)
cross = DifferentialEvolution(1, 0.08)
local_search = TestMut(n, k, m, p, probability=1)
mutation = UniformMutation(0.0004, 0.03)

algorithm = IBEA(problem, mutator=mutation, local_search=local_search, selector=sel, variator=cross, population_size=5)
algorithm.run(1000)
