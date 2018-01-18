import numpy as np
from platypus import Mutation, copy, Real, random, default_variator, RandomGenerator, \
    HypervolumeFitnessEvaluator, AttributeDominance, AbstractGeneticAlgorithm, fitness_key, TournamentSelector, Variator


class CustomVariator(Variator):
    def __init__(self, m, n, k, crossover_rate=0.1, step_size=0.5):
        super(CustomVariator, self).__init__(2)
        self.crossover_rate = crossover_rate
        self.step_size = step_size
        self.m, self.n, self.k = m, n, k


    def evolve(self, parents):
        result = copy.deepcopy(parents[0])
        G1 = np.array(parents[0].variables)[0:self.n * self.k].reshape((self.n, self.k))
        Ss1 = np.array(parents[0].variables)[(self.n * self.k):].reshape((self.m, self.k, self.k))

        G2 = np.array(parents[1].variables)[0:self.n * self.k].reshape((self.n, self.k))
        Ss2 = np.array(parents[1].variables)[(self.n * self.k):].reshape((self.m, self.k, self.k))

        newG = np.zeros(G1.shape)

        from_G1 = 0

        for i in range(G1.shape[0]):
            bit = bool(random.getrandbits(1))
            from_G1+=int(bit)
            newG[i,:] = (G1[i, :] if bit else G2[i,:])

        if from_G1 < G1.shape[0]/2:
            result.variables = list(np.concatenate((newG.flatten(), Ss1.flatten()), axis=0))
        else:
            result.variables = list(np.concatenate((newG.flatten(), Ss2.flatten()), axis=0))
        return [result]



class AdamLocalSearch(Mutation):
    def __init__(self, p, reshaper, probability=1, steps=100):
        super(AdamLocalSearch, self).__init__()
        self.probability = probability
        self.p = p
        self.steps = steps
        self.reshaper = reshaper

    def mutate(self, parent):
        child = copy.deepcopy(parent)
        problem = child.problem
        probability = self.probability

        if isinstance(probability, int):
            probability /= float(len([t for t in problem.types if isinstance(t, Real)]))

        if random.uniform(0.0, 1.0) <= self.probability:
            G, Ss = self.reshaper.vec2mat(child.variables)

            c, newG, newS = self.p.adam(G, Ss, self.steps)
            con = self.reshaper.mat2vec(newG, newS)
            child.variables = con.tolist()

            child.objectives = [c]
            child.evaluated = True

        return child


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
        self.fitness_evaluator.evaluate(self.population)
        while len(self.population) > self.population_size:
            # self.fitness_evaluator.remove(self.population, self._find_worst())
            ii = self._find_worst()
            print('---' + str(self.population[ii].objectives))
            self.fitness_evaluator.remove(self.population, ii)

        for cand in self.population:
            print('RSE: ' + str(cand.objectives))

        if self._cur_step % self.mutation_every_n_steps == 0:
            print("local search whole population")
            self.population = [self.local_search.mutate(x) for x in self.population]
            self.fitness_evaluator.evaluate(self.population)
            for cand in self.population:
                print('RSE: ' + str(cand.objectives))
        self._cur_step += 1

    def _find_worst(self):
        index = 0

        for i in range(1, len(self.population)):
            if self.fitness_comparator.compare(self.population[index], self.population[i]) < 0:
                index = i

        return index

