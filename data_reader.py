import numpy as np

def read_R(directory, matrices_names, n):
    matrices_names = [directory + mat for mat in matrices_names]

    R_data = np.zeros((len(matrices_names), n, n))
    for i, mat in enumerate(matrices_names):
        sparse_R = np.loadtxt(mat, delimiter=',', skiprows=1)
        for j in range(sparse_R.shape[0]):
            R_data[i, int(sparse_R[j, 0]) - 1, int(sparse_R[j, 1]) - 1] = sparse_R[j, 2]
    return R_data
