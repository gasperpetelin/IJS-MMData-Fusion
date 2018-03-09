from platypus import ParetoDominance, TournamentSelector, AttributeDominance, fitness_key, copy

from evolutionary import DeleteColumn, OrthogonalityEnforcer, RandomKMatrixGenerator, AdamLocalSearch, JoinMatrices2, \
    customIBEA, RandomKMatrixGenerator_masked, JoinMatrices2_masked, NullMutation, AdamLocalSearch_masked, \
    MaskedMutation
from gradientdescentoptimizer import GradientDescentOptimizer, GradientDescentMaskOptimizer
from problems import Tri_Factorization, Tri_Factorization_Masked
from util import read_matrices, write_population_objectives
import numpy as np

mutation=MaskedMutation(mask_mutation_probability=0.03)#DeleteColumn(probability=1)
selector=TournamentSelector(2,AttributeDominance(fitness_key,False)) # crossover selector
constraint = OrthogonalityEnforcer()
comparator=ParetoDominance()


#directories=['data/test_data_n=100_k=10_N=5_density=6.00/','data/test_data_n=500_k=10_N=5_density=6.00/','data/test_data_n=800_k=10_N=5_density=6.00/']
#out_file_names=['multi_n100k10d6','multi_n500k10d6','multi_n800k10d6']
#matrices_list=[['R1_100_1.1800.csv','R2_100_2.0000.csv','R3_100_3.7000.csv','R4_100_2.5000.csv','R5_100_1.9000.csv'], \
#          ['R1_500_1.8408.csv','R2_500_2.0232.csv','R3_500_1.8416.csv','R4_500_1.7220.csv','R5_500_2.1224.csv'], \
#          ['R1_800_2.0164.csv','R2_800_1.8628.csv','R3_800_1.5189.csv','R4_800_1.9077.csv','R5_800_1.6077.csv']]

directory = 'data/test_data_orthogonal_n=100_k=20N=5_density=0.1/'
matrices = ['R1.csv','R2.csv','R3.csv','R4.csv','R5.csv']
n = 100
m=5

Ri = read_matrices(directory, matrices, n)
# Generate tensorflow computational graphs for RSE evaluation and gradient descent
p = GradientDescentMaskOptimizer(Ri)

problem = Tri_Factorization_Masked(p, number_of_variables=3, number_of_objectives=2)

generator=RandomKMatrixGenerator_masked(m, n, 80, scale=0.1)

#crossover = GradientDescentJoin(Ri)
local_search=AdamLocalSearch_masked(p, steps=100)#TODO
crossover = JoinMatrices2_masked(Ri, extent=1)

algorithm = customIBEA(problem=problem,
                       generator=generator,
                       mutator=mutation,
                       local_search=local_search,
                       selector=selector,
                       crossover=crossover,
                       fitness_comparator=comparator,
                       constraint_enforcer=constraint,
                       population_size=30)


algorithm.run(150000)


out_file_name = 'multi_n100k10d6'

write_population_objectives(algorithm.result, out_file_name + "_mask_")