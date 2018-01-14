import numpy as np
from platypus import Mutation, copy, Real, random, default_variator, RandomGenerator, \
    HypervolumeFitnessEvaluator, AttributeDominance, AbstractGeneticAlgorithm, fitness_key, TournamentSelector


class AdamLocalSearch(Mutation):
    def __init__(self, n, k, m, p, probability=1, steps = 200):
        super(AdamLocalSearch, self).__init__()
        self.probability = probability
        self.n=n
        self.k=k
        self.m=m
        self.p=p
        self.steps = steps

    def mutate(self, parent):
        child = copy.deepcopy(parent)
        problem = child.problem
        probability = self.probability

        if isinstance(probability, int):
            probability /= float(len([t for t in problem.types if isinstance(t, Real)]))

        if random.uniform(0.0, 1.0) <= self.probability:
            #print("is_mutated")
            x = np.array(child.variables)

            G = x[0:self.n * self.k].reshape((self.n, self.k))
            Ss = x[(self.n * self.k):].reshape((self.m, self.k, self.k))

            c, newG, newS = self.p.new_weights(G, Ss, steps=self.steps)
            con = np.concatenate((newG.flatten(), newS.flatten()), axis=0)
            child.variables = con.tolist()


            #cost, gradG, gradS = self.p.cost_grad(newG, newS)

            #print(c, cost, child.variables[:5])

            #child.objectives=[cost]
            child.evaluated = False

            #child.objectives=[cost]

        return child

    def um_mutation(self, x, lb, ub):
        return random.uniform(lb, ub)

class IBEA(AbstractGeneticAlgorithm):
    def __init__(self, problem,
                 local_search,
                 mutator,
                 population_size=100,
                 generator=RandomGenerator(),
                 fitness_evaluator=HypervolumeFitnessEvaluator(),
                 fitness_comparator=AttributeDominance(fitness_key, False),
                 variator=None,
                 selector=None,
                 **kwargs):
        super(IBEA, self).__init__(problem, population_size, generator, **kwargs)
        self.fitness_evaluator = fitness_evaluator
        self.fitness_comparator = fitness_comparator
        self.selector = selector
        self.variator = variator
        self.mutation_every_n_steps = 3
        self._cur_step = 0
        self.mutator = mutator

        self.local_search = local_search

    def initialize(self):
        super(IBEA, self).initialize()
        self.fitness_evaluator.evaluate(self.population)

        if self.variator is None:
            self.variator = default_variator(self.problem)

        if self.selector is None:
            self.selector = TournamentSelector(2, self.fitness_comparator)

    def iterate(self):
        offspring = []

        print("crossover")
        while len(offspring) < self.population_size:
            parents = self.selector.select(self.variator.arity, self.population)
            offspring.extend(self.variator.evolve(parents))

        print("mutation")
        offspring = [self.mutator.mutate(x) for x in offspring]









        self.evaluate_all(offspring)



        self.population.extend(offspring)
        #self.evaluate_all(self.population)
        self.fitness_evaluator.evaluate(self.population)
        while len(self.population) > self.population_size:
            self.fitness_evaluator.remove(self.population, self._find_worst())

        if self._cur_step%self.mutation_every_n_steps==0:
            print("local search whole population")
            self.population = [self.local_search.mutate(x) for x in self.population]
        self._cur_step+=1

        self.evaluate_all(self.population)



        while len(self.population) > self.population_size:
            self.fitness_evaluator.remove(self.population, self._find_worst())

    def _find_worst(self):
        index = 0

        for i in range(1, len(self.population)):
            if self.fitness_comparator.compare(self.population[index], self.population[i]) < 0:
                index = i

        return index