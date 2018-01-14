from __future__ import print_function

import numpy as np
from platypus import Problem, Real, AttributeDominance, fitness_key, DifferentialEvolution, TournamentSelector, PM, \
    UniformMutation

from gdproblem import GDProblem
from gradient_mutation import AdamLocalSearch, IBEA



m = 5
n = 100
k = 10

p = GDProblem(m, n, k)


def eval(x):
    x = np.array(x)
    G = x[0:n * k].reshape((n, k))
    Ss = x[(n * k):].reshape((m, k, k))
    cost, gradG, gradS = p.cost_grad(G, Ss)
    print("eval_print", cost)
    return [cost]


problem = Problem(n * k + k * k * m, 1)
problem.types[:] = Real(0, 1)
problem.function = eval

sel = TournamentSelector(2, AttributeDominance(fitness_key, False))
# sel = SUS(2, AttributeDominance(fitness_key, False))
# algorithm = IBEA(problem, selector=sel,  variator=TestMut(n, k, m, p, probability=1), population_size = 7,)

# SPREMINJANJE PARAMETROV ZA MUTACIJE, KRIZANJA IN LOCAL SEARCH
cross = DifferentialEvolution(1, 0.08) #crossover_rate=1, step_size=0.08


local_search = AdamLocalSearch(n, k, m, p, probability=1, steps = 200) #probability=1, STEVILO KORAKOV ADAMA = 200
mutation = UniformMutation(0.0004, 0.03) # probability=0.0004, perturbation=0.03

algorithm = IBEA(problem, mutator=mutation,
                 local_search=local_search,
                 selector=sel,
                 variator=cross,
                 population_size=5) #Velikost populacije
algorithm.run(1000) #Stevilo generacij
