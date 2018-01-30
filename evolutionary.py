import numpy as np
from platypus import Mutation, copy, HypervolumeFitnessEvaluator, AttributeDominance, \
        AbstractGeneticAlgorithm, fitness_key, Selector, Type, Generator, Variator, \
        Solution

from itertools import product


class RandomKMatrixGenerator(Generator):
    def __init__(self,m,n,max_k,scale=1):
        super(RandomKMatrixGenerator, self).__init__()
        self.m=m
        self.n=n
        self.max_k=max_k
        self.scale=scale

    def generate(self, problem):
        solution = Solution(problem)
        i=np.random.randint(self.max_k)+1
        solution.variables[0]=self.scale*np.random.rand(self.n,i)
        solution.variables[1]=self.scale*np.random.rand(self.m,i,i)
        return solution

class ConstantKMatrixGenerator(Generator):
    def __init__(self,m,n,k,scale=1):
        super(ConstantKMatrixGenerator, self).__init__()
        self.m=m
        self.n=n
        self.k=k
        self.scale=scale

    def generate(self, problem):
        solution = Solution(problem)
        solution.variables[0]=self.scale*np.random.rand(self.n, self.k)
        solution.variables[1]=self.scale*np.random.rand(self.m, self.k, self.k)
        return solution

class NoneEnforcer(Mutation):
    def mutate(self, parent):
        return parent

class OrthogonalityEnforcer(Mutation):
    def _make_orthogonal(self,P):
        # Find indexes of max elements in each row
        arg_max = np.argmax(P, 1)

        # Zero matrix
        Z = np.zeros(P.shape)
        for i in range(P.shape[0]):
            # Write max elements in Z
            Z[i, arg_max[i]] = P[i, arg_max[i]]
        return Z

    def mutate(self, parent):
        orthogonal_parent = copy.deepcopy(parent)
        G = np.abs(orthogonal_parent.variables[0])
        orthogonal_G = self._make_orthogonal(G)
        orthogonal_parent.variables[0] = orthogonal_G
        orthogonal_parent.evaluated = False
        return orthogonal_parent


class AdamLocalSearch_one_obj(Mutation):
    def __init__(self, p, probability=1, steps=100):
        super(AdamLocalSearch_one_obj, self).__init__()
        self.probability = probability
        self.p=p
        self.steps=steps

    def mutate(self,parent):
        child = copy.deepcopy(parent)
        if np.random.rand() <= self.probability:
            G=child.variables[0]
            S=child.variables[1]
            c,newG,newS=self.p.adam(G,S,self.steps)
            child.variables=[newG,newS]
            child.objectives=[c]
            child.evaluated=True
        return child

class AdamLocalSearch(Mutation):
    def __init__(self, p, probability=1, steps=100):
        super(AdamLocalSearch, self).__init__()
        self.probability = probability
        self.p=p
        self.steps=steps

    def mutate(self,parent):
        child = copy.deepcopy(parent)
        if np.random.rand() <= self.probability:
            G=child.variables[0]
            S=child.variables[1]
            k=S.shape[1]
            c,newG,newS=self.p.adam(G ,S ,self.steps)
            child.variables=[newG,newS]
            child.objectives=[c,k]
            child.evaluated=True
        return child

class NullMutation(Mutation):
    def __init__(self):
        super(NullMutation,self).__init__()

    def mutate(self, parent):
        return None

class DeleteColumn(Mutation):
    def __init__(self,probability=1, fill_empty_rows=False):
        super(DeleteColumn,self).__init__()
        self.probability=probability
        self.fill_empty_rows = fill_empty_rows

    def mutate(self,parent):
        child = copy.deepcopy(parent)
        if np.random.rand() <= self.probability:
            k=child.objectives[1]
            if k > 1:
                G=child.variables[0]
                S=child.variables[1]
                i=np.random.randint(k)
                newG=np.delete(G,i,1)

                if self.fill_empty_rows:
                    x_sum = np.sum(newG, 1)
                    for index, v in enumerate(x_sum):
                        if v==0:
                            newG[index, np.random.randint(newG.shape[1])] =\
                                0.001 * np.random.rand()


                newS=np.delete(S,i,1)
                newS=np.delete(newS,i,2)
                child.variables=[newG,newS]
                child.evaluated=False
                return child
        return None

class NullCrossover(Variator):
    def __init__(self, extent=1):
        super(NullCrossover,self).__init__(2)
        self.extent=extent

    def evolve(self, parents):
        return [parents[0]]

class GradientDescentJoin(Variator):
    def __init__(self, R, matrix_similarity = 0.5, extent=1):
        super(GradientDescentJoin, self).__init__(2)
        #matrix_similarity: 0 - equal to parent 1, 1 - equal to parent 2
        self.R = R
        self.matrix_similarity = matrix_similarity
        self.extent = extent

    def _gradientdescent(self, G, D):
        k = G.shape[1]

        Am = []
        valid_Am = []
        for m in range(0, k):
            vals = np.argwhere(G[:, m] > 0)
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

    def _merge(self, G1, G2, matrix_similarity=0.5):
        max_k_shape = np.max([G1.shape[1], G2.shape[1]])
        G = np.zeros((G1.shape[0], max_k_shape))

        for i in range(G1.shape[0]):
            if np.random.rand() < matrix_similarity:
                G[i, :len(G1[i, :])] = G1[i, :]
            else:
                G[i, :len(G2[i, :])] = G2[i, :]
        return G

    def evolve(self, parents):
        G1 = parents[0].variables[0]
        G2 = parents[1].variables[0]

        G = self._merge(G1, G2)

        S = np.zeros((self.R.shape[0], G.shape[1], G.shape[1]))
        for i in range(self.R.shape[0]):
            R = self.R[i,:,:]
            S[i] = self._gradientdescent(G, R)
            print("S",i)

        result = copy.deepcopy(parents[0])
        result.variables=[G,S]
        result.evaluated=False

        return [result]

def _gradientdescent(G, D):
    k = G.shape[1]

    Am = []
    valid_Am = []
    for m in range(0, k):

        vals = np.argwhere(G[:, m] > 0)
        Am.append(vals)
        if len(vals)!=0:
            valid_Am.append(m)

    S = np.zeros((k, k))

    # Cartesian product between nonempty Am's
    for x, y in product(valid_Am, repeat=2):
        Ax_Ay_cross_product = list(product(Am[x], Am[y]))
        denumerator = sum([D[i,j] * G[i,x] * G[j,y] for i,j in Ax_Ay_cross_product])
        denominator = sum([G[i,x]**2 * G[j,y]**2 for i,j in Ax_Ay_cross_product])
        S[x, y] = denumerator/denominator
    return S


class JoinMatrices2(Variator):
    def __init__(self, R, extent=1):
        super(JoinMatrices2,self).__init__(2)
        self.R = R
        self.extent=extent

    def evolve(self, parents):
        G1=parents[0].variables[0]
        G2=parents[1].variables[0]
        G = np.concatenate((G1, G2), axis=1)

        arg_max = np.argmax(G, 1)
        Z = np.zeros(G.shape)
        for i in range(G.shape[0]):
            Z[i, arg_max[i]] = G[i, arg_max[i]]

        result=copy.deepcopy(parents[0])

        S_new = np.zeros((self.R.shape[0], Z.shape[1], Z.shape[1]))
        for i in range(self.R.shape[0]):
            Rii = self.R[i, :, :]
            S_new[i] = _gradientdescent(Z, Rii)

        result.variables=[Z, S_new]
        result.evaluated=False
        return [result]

class JoinMatrices(Variator):
    def __init__(self,extent=1,scale=1e-8):
        super(JoinMatrices,self).__init__(2)
        self.extent=extent
        self.scale=scale
        self.factorG=0.5**0.25
        self.factorS=0.5**0.5

    def evolve(self,parents):
        result=copy.deepcopy(parents[0])
        # Join G matrices side by side
        G1=parents[0].variables[0]
        G2=parents[1].variables[0]
        newG=self.factorG*np.concatenate((G1,G2),axis=1)
        # Join S tensors as a direct sum
        S1=parents[0].variables[1]
        S2=parents[1].variables[1]
        k1=parents[0].objectives[1]
        k2=parents[1].objectives[1]
        m=S1.shape[0]
        newS=np.empty((m,k1+k2,k1+k2))
        newS[:,:k1,:k1]=self.factorS*S1
        newS[:,k1:,k1:]=self.factorS*S2
        almost_zero=self.scale*np.random.rand(m,k1,k2)
        newS[:,:k1,k1:]=almost_zero
        newS[:,k1:,:k1]=np.transpose(almost_zero,(0,2,1))
        # Save to result
        result.variables=[newG,newS]
        result.evaluated=False
        return [result]

class customIBEA(AbstractGeneticAlgorithm):
    def __init__(self, problem,
                 local_search, fitness_comparator, generator, selector, crossover, mutator, constraint_enforcer,
                 population_size=100, fitness_evaluator=HypervolumeFitnessEvaluator(),
                 **kwargs):

        super(customIBEA, self).__init__(problem, population_size, generator, **kwargs)
        self.fitness_evaluator=fitness_evaluator
        self.fitness_comparator=fitness_comparator
        self.crossover=crossover
        self.mutator=mutator
        self.local_search=local_search
        self.selector=selector
        self.iteration_count=1

        self.constraint_enforcer = constraint_enforcer

    def iterate(self):




        # Calculate fitness needed for crossover
        self.fitness_evaluator.evaluate(self.population)
        # Corssover
        crossover_offspring=[]


        # Force ortogonal TODO
        self.population = self.constraint_enforcer.evolve(self.population)

        while len(crossover_offspring) < self.crossover.extent*self.population_size:
            parents=self.selector.select(self.crossover.arity, self.population)
            crossover_offspring.extend(self.crossover.evolve(parents))
        # Mutation
        mutation_offspring=self.mutator.evolve(self.population)
        mutation_offspring=[x for x in mutation_offspring if x is not None]



        # Local search on all individuals
        self.population.extend(crossover_offspring)
        self.population.extend(mutation_offspring)



        self.population=self.local_search.evolve(self.population)

        self.population = self.constraint_enforcer.evolve(self.population)
        self.evaluate_all(self.population)
        self.fitness_evaluator.evaluate(self.population)

        self.nfe+=(self.local_search.steps+1)*len(self.population)
        # Remove worst individuals
        while len(self.population) > self.population_size:
            del self.population[self._find_worst()]
        print('iteration '+str(self.iteration_count)+' completed with nfe='+str(self.nfe))
        self.iteration_count+=1


    def _find_worst(self):
        pop_size=len(self.population)
        for init_index in np.random.permutation(pop_size):
            index=init_index
            for i in range(pop_size):
                if i == index:
                    continue
                if self.fitness_comparator.compare(self.population[index], self.population[i]) < 0:
                    index=i
            if index != init_index:
                return index
        return index
