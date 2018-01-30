import numpy as np

def merge(G1, G2, matrix_similarity = 0.5):
    max_k_shape = np.max([G1.shape[1], G2.shape[1]])
    G = np.zeros((G1.shape[0], max_k_shape))

    for i in range(G1.shape[0]):
        if np.random.rand()<matrix_similarity:
            G[i,:len(G1[i, :])] = G1[i, :]
        else:
            G[i,:len(G2[i, :])] = G2[i, :]
    return G






G = merge(np.random.rand(100, 5), np.random.rand(100, 8))

print(np.random.rand())