import numpy as np
from itertools import product

def t(G, D):
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


def t1(G, D):
    k = G.shape[1]

    Am = []
    valid_Am = []
    for m in range(0, k):
        vals = np.argwhere(G[:, m] > 0)
        Am.append(vals)
        if len(vals)!=0:
            valid_Am.append(m)

    S = np.zeros((k, k))
    #for x, y in product(valid_Am, repeat=2):
    for x in range(k):
        for y in range(k):
            #print(x,y)
            if len(Am[x])!=0 and len(Am[y])!=0:
                denum = 0
                denom = 0
                for i in Am[x]:
                    for j in Am[y]:
                        denum += D[i,j] * G[i,x] * G[j,y]
                        denom += G[i,x]**2 * G[j,y]**2
                S[x, y] = denum/denom
    return S




#R = np.random.rand(5,5)
#S = np.random.rand(2,2)
#G = np.array([[0,1],[1,0],[1,0],[0,1], [1,0]])
#G = np.array([[0,1],[0,1],[0,1],[0,1], [0,1]])
#print("error", np.sum((R - G.dot(S).dot(G.T))**2))
##
#itr = t(G, R)
##
#print("error", np.sum((R - G.dot(itr).dot(G.T))**2))

G = np.array([[0,1,0],[0,1,0],[0,1,0],[0,0,1]])
R = np.random.rand(4,4)
S = np.random.rand(3,3)
print("error", np.sum((R - G.dot(S).dot(G.T))**2))
itr = t(G, R)
print("error", np.sum((R - G.dot(itr).dot(G.T))**2))
itr = t1(G, R)
print("error", np.sum((R - G.dot(itr).dot(G.T))**2))
print(itr)
#dir = "data/test_data_n=100_k=10_N=5_density=6.00/"
#matrices = ['R1_100_1.1800.csv']
#n = 100
#
#R_data = np.zeros((len(matrices), n, n))
#for i, mat in enumerate(matrices):
#    sparse_R = np.loadtxt(dir+mat, delimiter=',', skiprows=1)
#    for j in range(sparse_R.shape[0]):
#        R_data[i, int(sparse_R[j, 0]) - 1, int(sparse_R[j, 1]) - 1] = sparse_R[j, 2]
#
#
#
#print(R_data.shape)

