import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import random
import itertools as it

def gen_list_val(length,scale):
    return np.around((random.uniform(low=0.1, high=1.0,size=(length))*scale)).tolist()


def lin_equ(l1, l2):
    m = (l2[1] - l1[1]) / (l2[0] - l1[0])
    c = (l2[1] - (m * l2[0]))
    return m, c

def compute_risk(seq_c, seq_a, C_agi, A_aligned):
    """
    risk is a function of distance between capabilities threshold and the 
    intersection of alignment with ai progress curve. 
    """
    index_cross = next(i for i,v in enumerate(seq_a) if v > A_aligned)

    # find for which capability the alignment threshold is crossed
    coeff = lin_equ([seq_c[index_cross-1],seq_a[index_cross-1]], [seq_c[index_cross],seq_a[index_cross]])
    risk = (A_aligned - coeff[1])/coeff[0] - C_agi
    
    return risk

def output_path(i_order, df, C_agi, A_aligned):
    
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
    
    # compute the risk score 
    risk = compute_risk(C_list, A_list, C_agi, A_aligned)
       
    return C_list, A_list, round(risk,3), alligned 


def plot_innovations(i,seq_c, seq_a, C_agi, A_aligned, sort = False):
    
    start_c =  np.zeros(len(seq_c))
    start_a =  np.zeros(len(seq_a))
    
    if sort:
        start_c = [0] + seq_c
        start_a = [0] + seq_a
        
        start_c.pop()
        start_a.pop()

    plt.figure()
    plt.ylabel('Alignment (a)')
    plt.xlabel('Capabilities (c)')
    ax = plt.gca()

    for nr in range(len(i)):
        ax.annotate(f'i{i[nr-1]}', [seq_c[nr-1],seq_a[nr-1]], fontsize=14)

    ax.quiver(start_c, start_a, np.subtract(seq_c,start_c), np.subtract(seq_a,start_a),angles='xy', scale_units='xy',color=np.random.rand(len(seq_c),3),scale=1)

    if sort:
        ax.set_xlim([0, seq_c[-1] + 2])
        ax.set_ylim([0, seq_a[-1] + 2])
    else:
        ax.set_xlim([0, max(seq_c) + 2])
        ax.set_ylim([0, max(seq_a) + 2])

    ax.set_xlim([0, C_agi + 10])
    ax.set_ylim([0, A_aligned + 10])

    ax.axhline(y=A_aligned, color='g')
    ax.axvline(x=C_agi, color='r')

    # plt.grid()
    plt.draw()
    plt.show()
    
def optimize_path(ini_path, df, C_agi, A_aligned):
    
    # initialize path
    all_paths = list(it.permutations(ini_path))
    
    # safe the path with the lowerst risk: 
    risk_lowest = 0
    safest_path = ini_path
        
    for path in all_paths:
        seq_c, seq_a, risk, aligned = output_path(path, df, C_agi, A_aligned)

        print(risk)
        
        if risk < risk_lowest:
            safest_path = path 
            risk_lowest = risk
    
    seq_c, seq_a, risk, aligned = output_path(safest_path, df, C_agi, A_aligned)
    plot_innovations(safest_path, seq_c, seq_a, C_agi, A_aligned, True)
    
    return [safest_path, risk_lowest]


def optimal_inno(ini_path, df, C_agi, A_aligned, stage):
    
    if stage == 1:
        return 1
    else:
        return optimal_inno(ini_path, df, C_agi, A_aligned, stage - 1)
    
    
    
# find the optimal path using the best decision in each stage

def score(innovation,df):
    return df.iloc[innovation-1]["a"]/df.iloc[innovation-1]["c"]

def best_step(check_innos, df):
    
    highest_score = 0
    score_value = 0
    best_inno = 0
    for i in check_innos:
        score_value = score(i,df)
        if score_value > highest_score:
            highest_score = score_value
            best_inno = i
    
    return best_inno

def optimal_path_rc(df, C_agi, A_aligned, t):
    
    if t == 1:
        return [best_step(list(range(1,len(df)+1)), df)]   
        
    else:
        
        current_path = optimal_path_rc(df, C_agi, A_aligned, t - 1)
        left_innovations = list(set(range(1,len(df)+1)) - set(current_path))
        current_path.append(best_step(left_innovations, df))
        return current_path
    

def genr_game_needs(stages, scale):
    # Generate capabilities and alignment values 

    # define start values and predefined knowledge 
    # table amount of innovations, capabilities & alignment
    inno = list(range(1,stages+1))
    c = gen_list_val(stages,scale)
    a = gen_list_val(stages,scale)

    df = pd.DataFrame({'c':c, 'a':a}, index = inno)

    # threshold values
    C_agi = sum(c)*0.8 
    A_aligned = sum(a)*0.8

    # print the capabilities 
    print('Capabilities vector:')
    print(c)
    print('Alignment vector')
    print(a)
    print('AGI threshold')
    print(C_agi)
    print('Alignment threshold')
    print(A_aligned)

    return df, c, a, C_agi, A_aligned 