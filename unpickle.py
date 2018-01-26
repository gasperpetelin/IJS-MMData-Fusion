import numpy as np
import pickle

name='multi_n500k10d6r1'
# Load the pickled file
with open(name+'.pickle','rb') as pickled_file:
    [Gs,Ss,front]=pickle.load(pickled_file)

# Choose an index of a solution in population
index=12
print(front[index])
# Save matrices of this solution to csv files
np.savetxt('G.csv',Gs[index],delimiter=',')
for i in range(Ss[index].shape[0]):
    name='S'+str(i+1)+'.csv'
    np.savetxt(name,Ss[index][i],delimiter=',')
