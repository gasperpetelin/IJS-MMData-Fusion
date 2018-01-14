import random

import numpy as np
from platypus import ParetoDominance


#def select(population_fit, sample_size=3):
#    normalized_cum = np.cumsum(population_fit/np.sum(population_fit))
#    pointers = np.arange(0,1,1/sample_size)
#    random_offset = np.random.rand(1)/sample_size
#    cum_pointer = 0
#    selected = []
#    for i in pointers+random_offset:
#        while normalized_cum[cum_pointer]<i:
#            cum_pointer+=1
#        selected.append(cum_pointer)
#    return selected
#
#a = np.array([1.5,3,1,1,1.5,1,3])
#print(select(a, sample_size=len(a)))


#class SUS(object):
#    def __init__(self, tournament_size=2, dominance=ParetoDominance()):
#        super(SUS, self).__init__()
#        self.tournament_size = tournament_size
#        self.dominance = dominance
#
#    def select(self, n, population):
#        fitness_values = np.array([1/(x.objectives[0]+0.0001) for x in population])
#        indexes = select(fitness_values, sample_size=n)
#        v = [population[x] for x in indexes]
#        return v
#
#    def select_one(self, population):
#        winner = random.choice(population)
#
#        for _ in range(self.tournament_size - 1):
#            candidate = random.choice(population)
#            flag = self.dominance.compare(winner, candidate)
#            if flag > 0:
#                winner = candidate
#        return winner