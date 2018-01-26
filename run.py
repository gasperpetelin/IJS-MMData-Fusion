from __future__ import print_function

import pickle
import random
import time

import numpy as np
from platypus import AttributeDominance, fitness_key, TournamentSelector, ParetoDominance

from evolutionary import MatrixGenerator, AdamLocalSearch, DeleteColumn, JoinMatrices, customIBEA, \
    OrthogonalityEnforcer
from gradientdescentoptimizer import GradientDescentOptimizer

# Optimization specifications
from problems import Tri_Factorization
from regularization import AngleRegulatrization
from util import read_matrices, write_value

m=5
max_k=10
population_size=20
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
crossover=JoinMatrices(extent=crossover_extent)
mutation=DeleteColumn(probability=mutation_probability)
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
    p = GradientDescentOptimizer(Ri, regularization=AngleRegulatrization(0.0001))

    problem = Tri_Factorization(p)

    generator=MatrixGenerator(m,n,max_k, scale=initial_scale)
    local_search=AdamLocalSearch(p, steps=local_search_steps)
    
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
    
        # Save values of objectives and solutions
        front=np.empty((population_size,2))
        Gs=[]
        Ss=[]
        for i,sol in enumerate(algorithm.result):
            front[i,0]=sol.objectives[0]
            front[i,1]=sol.objectives[1]
            Gs.append(np.abs(sol.variables[0]))
            Ss.append(np.abs(sol.variables[1]))
        np.savetxt(out_file_name+run_str+'.csv',front,delimiter=',')
        # Save the population
        matrices_file=open(out_file_name+run_str+'.pickle','wb')
        pickle.dump([Gs,Ss,front],matrices_file)
        matrices_file.close()

    # Close tensorflow session
    p.close()

    # Save mean time needed to perform one run and mean nfe


    write_value(cumulative_time/number_of_runs, out_file_name+'.time')
    write_value(cumulative_nfe / number_of_runs, out_file_name + '.nfe')

    #cumulative_time/=number_of_runs
    #with open(out_file_name+'.time','w') as time_file:
    #    print(cumulative_time,file=time_file)
    #cumulative_nfe/=number_of_runs
    #with open(out_file_name+'.nfe','w') as nfe_file:
    #    print(cumulative_nfe,file=nfe_file)
