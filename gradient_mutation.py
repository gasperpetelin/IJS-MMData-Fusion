from platypus import Mutation, copy, Real, random, Selector, ParetoDominance, default_variator, RandomGenerator, \
    HypervolumeFitnessEvaluator, AttributeDominance, AbstractGeneticAlgorithm, fitness_key, TournamentSelector, \
    abstractmethod, Algorithm, ABCMeta, LOGGER, logging, datetime, time, MaxEvaluations, TerminationCondition
import numpy as np
from platypus.config import PlatypusConfig
from platypus.core import _EvaluateJob


class TestMut2(Mutation):
    """Uniform mutation."""

    def __init__(self, probability=1):
        super(TestMut2, self).__init__()
        self.probability = probability

    def mutate(self, parent):
        child = copy.deepcopy(parent)
        problem = child.problem
        probability = self.probability

        if isinstance(probability, int):
            probability /= float(len([t for t in problem.types if isinstance(t, Real)]))


        for i in range(len(child.variables)):
            if isinstance(problem.types[i], Real):
                if random.uniform(0.0, 1.0) <= self.probability:
                    child.variables[i] = self.um_mutation(float(child.variables[i]),
                                                          problem.types[i].min_value,
                                                          problem.types[i].max_value)
                    child.evaluated = False
        #print("mut")
        return child

    def um_mutation(self, x, lb, ub):
        return random.uniform(lb, ub)

class TestMut(Mutation):
    """Uniform mutation."""

    def __init__(self, n, k, m, p, probability=1):
        super(TestMut, self).__init__()
        self.probability = probability
        self.n=n
        self.k=k
        self.m=m
        self.p=p

    def mutate(self, parent):
        child = copy.deepcopy(parent)
        problem = child.problem
        probability = self.probability

        if isinstance(probability, int):
            probability /= float(len([t for t in problem.types if isinstance(t, Real)]))

        if random.uniform(0.0, 1.0) <= self.probability:
            print("is_mutated")
            x = np.array(child.variables)

            G = x[0:self.n * self.k].reshape((self.n, self.k))
            Ss = x[(self.n * self.k):].reshape((self.m, self.k, self.k))

            cost, gradG, gradS = self.p.cost_grad(G, Ss)

            #lr = 0.003
            #newG = G - lr * gradG
            #newS = Ss - lr * gradS

            c, newG, newS = self.p.new_weights(G, Ss, steps=5)

            con = np.concatenate((newG.flatten(), newS.flatten()), axis=0)

            n = con.tolist()
            child.variables = n
            #child.objectives=[cost]
            child.evaluated = False

            #child.objectives=[cost]
            #neki = 23

        return child

    def um_mutation(self, x, lb, ub):
        return random.uniform(lb, ub)


class TournamentSelectorMod(Selector):
    def __init__(self, tournament_size=2, dominance=ParetoDominance()):
        super(TournamentSelectorMod, self).__init__()
        self.tournament_size = tournament_size
        self.dominance = dominance

    def select_one(self, population):
        winner = random.choice(population)

        for _ in range(self.tournament_size - 1):
            candidate = random.choice(population)
            flag = self.dominance.compare(winner, candidate)

            if flag > 0:
                winner = candidate

        return winner

class RouletteStohasticUniversalSampling(Selector):
    pass


#class Algorithm(object):
#    __metaclass__ = ABCMeta
#
#    def __init__(self,
#                 problem,
#                 evaluator=None,
#                 log_frequency=None, config=None):
#        super(Algorithm, self).__init__()
#        self.problem = problem
#        self.evaluator = evaluator
#        self.log_frequency = log_frequency
#        self.nfe = 0
#
#        if self.evaluator is None:
#            self.evaluator = PlatypusConfig.default_evaluator
#
#        if self.log_frequency is None:
#            self.log_frequency = PlatypusConfig.default_log_frequency
#
#    @abstractmethod
#    def step(self):
#        raise NotImplementedError("method not implemented")
#
#    def evaluate_all(self, solutions):
#        unevaluated = [s for s in solutions if not s.evaluated]
#
#        jobs = [_EvaluateJob(s) for s in unevaluated]
#        results = self.evaluator.evaluate_all(jobs)
#
#        # if needed, update the original solution with the results
#        for i, result in enumerate(results):
#            if unevaluated[i] != result.solution:
#                unevaluated[i].variables[:] = result.solution.variables[:]
#                unevaluated[i].objectives[:] = result.solution.objectives[:]
#                unevaluated[i].constraints[:] = result.solution.constraints[:]
#                unevaluated[i].constraint_violation = result.solution.constraint_violation
#                unevaluated[i].feasible = result.solution.feasible
#                unevaluated[i].evaluated = result.solution.evaluated
#
#        self.nfe += len(unevaluated)
#
#    def run(self, condition):
#        if isinstance(condition, int):
#            condition = MaxEvaluations(condition)
#
#        if isinstance(condition, TerminationCondition):
#            condition.initialize(self)
#
#        last_log = self.nfe
#        start_time = time.time()
#
#        LOGGER.log(logging.INFO, "%s starting", type(self).__name__)
#
#        while not condition(self):
#            self.step()
#
#            if self.log_frequency is not None and self.nfe >= last_log + self.log_frequency:
#                LOGGER.log(logging.INFO,
#                           "%s running; NFE Complete: %d, Elapsed Time: %s",
#                           type(self).__name__,
#                           self.nfe,
#                           datetime.timedelta(seconds=time.time() - start_time))
#
#        LOGGER.log(logging.INFO,
#                   "%s finished; Total NFE: %d, Elapsed Time: %s",
#                   type(self).__name__,
#                   self.nfe,
#                   datetime.timedelta(seconds=time.time() - start_time))

#class AbstractGeneticAlgorithm(Algorithm):
#    __metaclass__ = ABCMeta
#
#    def __init__(self, problem,
#                 population_size=100,
#                 generator=RandomGenerator(),
#                 **kwargs):
#        super(AbstractGeneticAlgorithm, self).__init__(problem, **kwargs)
#        self.population_size = population_size
#        self.generator = generator
#        self.result = []
#
#    def step(self):
#        if self.nfe == 0:
#            self.initialize()
#            self.result = self.population
#        else:
#            self.iterate()
#            self.result = self.population
#
#    def initialize(self):
#        self.population = [self.generator.generate(self.problem) for _ in range(self.population_size)]
#        self.evaluate_all(self.population)
#
#    @abstractmethod
#    def iterate(self):
#        raise NotImplementedError("method not implemented")

class IBEA(AbstractGeneticAlgorithm):
    def __init__(self, problem,
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

    def initialize(self):
        super(IBEA, self).initialize()
        self.fitness_evaluator.evaluate(self.population)

        if self.variator is None:
            self.variator = default_variator(self.problem)

        if self.selector is None:
            self.selector = TournamentSelector(2, self.fitness_comparator)

    def iterate(self):
        offspring = []



        #while len(offspring) < self.population_size:
        #    parents = self.selector.select(self.variator.arity, self.population)
        #    offspring.extend(self.variator.evolve(parents))

        for parent in self.population:
            offspring.extend(self.variator.evolve([parent]))





        #parents = self.selector.select(self.population_size, self.population)
        #for p in parents:
        #    offspring.extend(self.variator.evolve([p]))

        self.evaluate_all(offspring)

        self.population.extend(offspring)
        self.fitness_evaluator.evaluate(self.population)


        #TODO selects only best solutions
        #self.population = self.selector.select(self.population_size, self.population)

        self.population.sort(key=lambda x: x.objectives[0])

        self.population = self.population[:self.population_size]

        #while len(self.population) > self.population_size:
        #    self.fitness_evaluator.remove(self.population, self._find_worst())

        #all_obj = [x.objectives[0] for x in self.population]
        #print(all_obj)

    def _find_worst(self):
        index = 0

        for i in range(1, len(self.population)):
            if self.fitness_comparator.compare(self.population[index], self.population[i]) < 0:
                index = i

        return index