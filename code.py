# # # stepwise plan 
"""
1. Make a small example with all feasible solutions
2. make a plot of the possible path was in two dimensions
    a. plot first all innovations 

"""

import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import numpy as np

""" step 1 """

# initiate values 
stages = 3

i = [1,2,3]
c = [1,1,3]
a = [2,1,1]

C_agi = 4.0
A_aligned = 3.5 

df = pd.DataFrame({'c':c, 'a':a}, index = i)


C = 0; A=0
for i in [1,2,3]:
 C += df.at[i,'c']
 A += df.at[i,'a']
 print(C,A)

# functions to compute payoff 
def compute_payoff(i_order = [1,2,3], df_payoffs = df):
    
    C_list = []
    A_list = []
    
    C = 0; A = 0; 
    alligned = True
    
    for i in i_order:
     C += df.at[i,'c']
     A += df.at[i,'a']
     
     C_list.append(C)
     A_list.append(A)
     
     if (C >= C_agi and A <= A_aligned):
         alligned = False
    
    return C,A, C_list, A_list, alligned 


""" step 2 """

# create the vectors 

def plot_innovations(i,c, a, C_agi, A_aligned, sort = False):
    
    start_c =  np.zeros(len(c))
    start_a =  np.zeros(len(a))
    
    if sort:
        start_c = [0] + c
        start_a = [0] + a
        
        start_c.pop()
        start_a.pop()

    plt.figure()
    plt.ylabel('Alignment (a)')
    plt.xlabel('Capabilities (c)')
    ax = plt.gca()

    for nr in range(len(i)):
        ax.annotate(f'i{i[nr-1]}', [c[nr-1],a[nr-1]], fontsize=14)

    ax.quiver(start_c, start_a, np.subtract(c,start_c), np.subtract(a,start_a),angles='xy', scale_units='xy',color=np.random.rand(len(c),3),scale=1)

    ax.set_xlim([0, C_agi + 2])
    ax.set_ylim([0, A_aligned + 2])

    ax.axhline(y=A_aligned, color='g')
    ax.axvline(x=C_agi, color='r')

    # plt.grid()
    plt.draw()
    plt.show()
    



# # determine all solutions 

# plot innovations 
plot_innovations(i,c,a,C_agi, A_aligned, sort = False)

# payoff for one path 
g1, g2, test_c, test_a, aligned = compute_payoff([1,2,3], df)

# plot this path 
plot_innovations(i, test_c, test_a, C_agi, A_aligned, True)

# plot all paths of innovations 
all_paths = list(it.permutations([1,2,3]))

for p in range(len(all_paths)):
    g1, g2, test_c, test_a, aligned = compute_payoff(all_paths[p], df)
    plot_innovations(i, test_c, test_a, C_agi, A_aligned, True)

