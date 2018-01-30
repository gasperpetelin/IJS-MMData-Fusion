import numpy as np
import pickle

def read_matrices(directory, matrices, n):
    matrices = [directory + mat for mat in matrices]
    R_data = np.zeros((len(matrices), n, n))
    for i, mat in enumerate(matrices):
        sparse_R = np.loadtxt(mat, delimiter=',', skiprows=1)
        for j in range(sparse_R.shape[0]):
            R_data[i, int(sparse_R[j, 0]) - 1, int(sparse_R[j, 1]) - 1] = sparse_R[j, 2]
    return R_data

def write_value(value, file_name):
    with open(file_name,'w') as time_file:
        print(value,file=time_file)

def write_population_objectives(population, file_name):
    objectives_number = len(population[0].objectives)
    front = np.empty((len(population), objectives_number))
    Gs = []
    Ss = []
    for i, sol in enumerate(population):
        for o in range(objectives_number):
            front[i, o] = sol.objectives[o]
        #front[i, 1] = sol.objectives[1]
        Gs.append(np.abs(sol.variables[0]))
        Ss.append(np.abs(sol.variables[1]))
    np.savetxt(file_name + '.csv', front, delimiter=',')
    # Save the population
    matrices_file = open(file_name + '.pickle', 'wb')
    pickle.dump([Gs, Ss, front], matrices_file)
    matrices_file.close()




