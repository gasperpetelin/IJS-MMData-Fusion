import numpy as np
import os
import itertools

def generate_matrix(n=100,
                    k=20,
                    N=5,
                    s_density = 0.1):

    G = np.zeros((n,k))
    for i in range(n):
        G[i, np.random.randint(k)] = np.random.rand()

    S_list = []

    for i in range(N):
        Si = np.random.rand(k,k)
        Si_mask = np.random.rand(k,k)>s_density
        Si[Si_mask] = 0
        S_list.append(Si)

    R_list = []

    for Si in S_list:
        R_list.append(G.dot(Si).dot(G.T))

    return G, S_list, R_list


def write_matrix(M, name):
    if not os.path.exists(directory + folder):
        os.makedirs(directory + folder)

    with open(name, 'w') as the_file:
        the_file.write(str(M.shape[0])+","+str(M.shape[1])+",0\n")

        nonzero_x, nonzero_y = np.nonzero(M)
        for x,y in zip(nonzero_x, nonzero_y):
            the_file.write(str(x+1) + "," + str(y+1) + ","+str(M[x,y])+"\n")




ns = [100, 200, 500, 800]
ks = [10,20,30,40,50]
Ns = [5]
s_densitys = [0.04, 0.06, 0.08, 0.1]


directory = "data/"

for n,k,N,s_density in list(itertools.product(ns,ks, Ns, s_densitys)):

    print(n, k, N, s_density)
    folder = "test_data_orthogonal_n="+str(n)+"_k="\
             +str(k)+"N="+str(N)+"_density="+str(s_density*100)+"/"

    G, S, R = generate_matrix(n,k,N,s_density)



    write_matrix(G, directory+folder+'G.csv')
    for i, s in enumerate(S):
        write_matrix(s, directory+folder+'S'+str(i+1)+'.csv')
    for i, r in enumerate(R):
        write_matrix(r, directory+folder+'R'+str(i+1)+'.csv')