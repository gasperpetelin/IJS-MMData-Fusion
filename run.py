from __future__ import print_function

import random
import time

import numpy as np
from platypus import AttributeDominance, fitness_key, TournamentSelector, ParetoDominance, copy

from evolutionary import RandomKMatrixGenerator, AdamLocalSearch, DeleteColumn, JoinMatrices, customIBEA, \
    OrthogonalityEnforcer, ConstantKMatrixGenerator, NullMutation, NullCrossover, AdamLocalSearch_one_obj, \
    GradientDescentJoin, JoinMatrices2
from gradientdescentoptimizer import GradientDescentOptimizer
# Optimization specifications
from problems import Tri_Factorization, Tri_Factorization_obj_1
from regularization import AngleRegulatrization
from util import read_matrices, write_value, write_population_objectives
from itertools import product

def _gradientdescent(G, D):
    k = G.shape[1]

    Am = []
    valid_Am = []
    for m in range(0, k):
        try:
            vals = np.argwhere(G[:, m] > 0)
        except:
            print(G[:, m])
        Am.append(vals)
        if len(vals) != 0:
            valid_Am.append(m)

    S = np.zeros((k, k))

    # Cartesian product between nonempty Am's
    for x, y in product(valid_Am, repeat=2):
        Ax_Ay_cross_product = list(product(Am[x], Am[y]))
        denumerator = sum([D[i, j] * G[i, x] * G[j, y] for i, j in Ax_Ay_cross_product])
        denominator = sum([G[i, x] ** 2 * G[j, y] ** 2 for i, j in Ax_Ay_cross_product])
        S[x, y] = denumerator / denominator
    return S



m=5
max_k=10
population_size=15
initial_scale=0.2
local_search_steps=100
crossover_extent=1.0
mutation_probability=1.0
# The number of function evaluations
expected_num_steps=np.array([5000,5,6])
nfes=population_size*expected_num_steps

# The number of runs
number_of_runs=1

# Chose the optimization specifics
#crossover=JoinMatrices2(extent=crossover_extent)
#crossover = GradientDescentJoin()
mutation=DeleteColumn(probability=mutation_probability)
#mutation = NullMutation()
selector=TournamentSelector(2,AttributeDominance(fitness_key,False)) # crossover selector
constraint = OrthogonalityEnforcer()
comparator=ParetoDominance() # used to find worst candidates



# List the problems to solve
ns=[100,500,800]
directories=['data/test_data_n=100_k=10_N=5_density=6.00/','data/test_data_n=500_k=10_N=5_density=6.00/','data/test_data_n=800_k=10_N=5_density=6.00/']
out_file_names=['multi_n100k10d6','multi_n500k10d6','multi_n800k10d6']
matrices_list=[['R1_100_1.1800.csv','R2_100_2.0000.csv','R3_100_3.7000.csv','R4_100_2.5000.csv','R5_100_1.9000.csv'], \
          ['R1_500_1.8408.csv','R2_500_2.0232.csv','R3_500_1.8416.csv','R4_500_1.7220.csv','R5_500_2.1224.csv'], \
          ['R1_800_2.0164.csv','R2_800_1.8628.csv','R3_800_1.5189.csv','R4_800_1.9077.csv','R5_800_1.6077.csv']]
# Loop through all problems
for n,directory,out_file_name,matrices,nfe in zip(ns,directories,out_file_names,matrices_list,nfes):

    Ri = read_matrices(directory, matrices, n)
    # Generate tensorflow computational graphs for RSE evaluation and gradient descent
    p = GradientDescentOptimizer(Ri)#, regularization=AngleRegulatrization(1))

    problem = Tri_Factorization(p)

    generator=RandomKMatrixGenerator(m, n, 10, scale=initial_scale)

    #crossover = GradientDescentJoin(Ri)
    local_search=AdamLocalSearch(p, steps=local_search_steps)
    crossover = JoinMatrices2(Ri, extent=crossover_extent)
    #local_search = AdamLocalSearch_one_obj(p, steps=local_search_steps)
    
    cumulative_time=0.0
    cumulative_nfe=0.0
    for run in range(number_of_runs):
        # Suffix that tells which run this is
        run_str='r'+str(run)
        print('starting case '+out_file_name+run_str)
        # Set a random seed for repeatability
        random_seed=run
        np.random.seed(random_seed)
        random.seed(random_seed+123)
        # Run the algorithm
        start_time=time.time()
        algorithm=customIBEA(problem=problem,
                             generator=generator,
                             mutator=mutation,
                             local_search=local_search,
                             selector=selector,
                             crossover=crossover,
                             fitness_comparator=comparator,
                             constraint_enforcer=constraint,
                             population_size=population_size)

        algorithm.run(int(nfe))

        elapsed_time=time.time()-start_time
        cumulative_time=cumulative_time+elapsed_time
        cumulative_nfe+=algorithm.nfe
        print('time needed='+str(elapsed_time))

        population = algorithm.result
        new_p = constraint.evolve(population)

        new_new_p = []
        for sub in new_p:
            G = sub.variables[0]
            S = sub.variables[1]
            p_copy = copy.deepcopy(sub)

            S_new = np.zeros((Ri.shape[0], G.shape[1], G.shape[1]))
            for i in range(Ri.shape[0]):
                Rii = Ri[i, :, :]
                S_new[i] = _gradientdescent(G, Rii)

            p_copy.variables = [G, S_new]
            new_new_p.append(p_copy)

        algorithm.evaluate_all(new_new_p)


    
        # Save values of objectives and solutions
        write_population_objectives(algorithm.result, out_file_name+run_str)
        write_population_objectives(new_new_p, out_file_name+"_ortoghonal_" + run_str)



    # Close tensorflow session
    p.close()

    # Save mean time needed to perform one run and mean nfe
    write_value(cumulative_time/number_of_runs, out_file_name+'.time')
    write_value(cumulative_nfe/number_of_runs, out_file_name + '.nfe')

