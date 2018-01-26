import numpy as np

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


