from __future__ import print_function

import numpy as np
from platypus import Problem, Real, AttributeDominance, fitness_key, DifferentialEvolution, TournamentSelector, PM, \
    UniformMutation

from MatrixReShaper import Reshape
from data_reader import read_R
from gdproblem import GDProblem
from gradient_mutation import AdamLocalSearch, IBEA, CustomVariator

m = 5
n = 100
k = 10
resharper = Reshape(m,n,k)

run_str='n100k10'
directory='data/test_data_n=100_k=10_N=5_density=6.00/'
matrices=['R1_100_1.1800.csv','R2_100_2.0000.csv','R3_100_3.7000.csv','R4_100_2.5000.csv','R5_100_1.9000.csv']

R_data = read_R(directory, matrices, n)

#matrices=[directory+mat for mat in matrices]
p = GDProblem(R_data)

def RSE(x):
    G, Ss = resharper.vec2mat(x)
    cost = p.calculate_cost(G, Ss)
    return [cost]

problem = Problem(n * k + k * k * m, 1)
problem.types[:] = Real(0, 0.1)
problem.function = RSE

sel = TournamentSelector(2, AttributeDominance(fitness_key, False))

cross = CustomVariator(m, n, k, 1, 0.08)

local_search = AdamLocalSearch(p, resharper, probability=1, steps = 200)
mutation = UniformMutation(0.0004, 0.03)

number_of_runs=10
population_size=15
number_of_generations=1000

costs=np.empty((number_of_runs,population_size))
for run in range(number_of_runs):
    algorithm = IBEA(problem=problem, mutator=mutation,
                 local_search=local_search,
                 selector=sel,
                 variator=cross,
                 population_size=population_size)
    algorithm.run(number_of_generations)
    for i,sol in enumerate(algorithm.population):
        costs[run,i]=sol.objectives[0]

np.savetxt(run_str+'.csv',costs,delimiter=',')

