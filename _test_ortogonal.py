import numpy as np
from platypus import Mutation, copy

def make_ortogonal(P):
    arg_max = np.argmax(P, 1)
    Z = np.zeros(P.shape)
    for i in range(P.shape[0]):
        Z[i, arg_max[i]] = P[i, arg_max[i]]
    return Z

r = np.random.rand(14,4)
r[0, 0] = 3
r[0, 2] = 3

print(r)
print(make_ortogonal(r))
