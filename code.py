# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 19:12:35 2017

@author: Pranav
"""

from math import log                                        #Log for entropy
from math import fabs                                       #For Absolute Values
import numpy as np                                             

   
def splitting_attribute(index, value, D):                  #Split tree Left and Right
        left_split = [row for row in D if row[index] <= value]          # left side computed 
        right_split = [row for row in D if row[index] > value]          # Right side; r is row
        return left_split, right_split
             

def IG(D,index,value):
    
    data_d1 = float(len(D))    
    information_gain = 0.0
    c1 = [row for row in D if row[index] <= value]                               #class c1
    c2 = [row for row in D if row[index] > value]                                #class c2
    d_class_n = [row for row in D if row[10]==0.0]                              #this is dn values geater than;r[10] last_attribute
    d_class_y = [row for row in D if row[10]==1.0]                              #this is dy values less than
    
    probab_c1 = float(len(c1))/float(data_d1)                               #class1 for Information Gain
    probab_c2 = float(len(c2))/float(data_d1)                               #class2 for Information Gain
    part_1 = float(probab_c1*np.log2(probab_c1)) + float(probab_c2*np.log2(probab_c2)) #Information gain part_1
    p_part_1 =  part_1 * -1                                                 #multiply -1 Make it positive
    
    prob1_classY = [row for row in d_class_y if row[index] <= value]                    
    p1new_Yes = float(len(prob1_classY))/data_d1
    prob2_Y = [row for row in d_class_y if row[index] > value]                       
    p2n_Y = float(len(prob2_Y))/data_d1
    p1_No = [row for row in d_class_n if row[index] <= value]
    p1n_N = float(len(p1_No))/data_d1
    p2_No = [row for row in d_class_n if row[index] > value]
    p2n_N = float(len(p2_No))/data_d1
    part_2 = float(len(d_class_y))/float(data_d1)*((p1new_Yes)*np.log2(p1new_Yes) +(p2n_Y)*np.log2(p2n_Y)) + float(len(d_class_n))/float(data_d1)*((p1n_N)*np.log2(p1n_N) +(p2n_N)*np.log2(p2n_N))
    part_2 =  part_2 * -1
    information_gain = float(p_part_1) - float(part_2)                               #Main information gain
    return information_gain

def G(D,index,value):
    observation_l = float(len(D))                                       #length of observation of data set
    GINI = 0.0                                                               #initialize GINI with 0     
    c1,c2 = splitting_attribute(index, value, D)                              #c1 c2 classes
    prob_c1 = float(float(len(c1))/float(observation_l))                    #Probability of class1
    prob_c2 = float(float(len(c1))/float(observation_l))                    #probability of class2                                                             
    GINI = float(1 - (float(prob_c1)**2) + (float(prob_c2)**2))             #Gini Val
    return GINI

def CART(D, index, value):
    c1,c2 = splitting_attribute(-1,0.0, D)
    
    N_Yes = float(len(c1))                                                #points_yes
    
    N_No= float(len(c2))                                                 #points_no 
    n = float(len(D))                                                   #Overall Length of matrix
    cart_1 = float(2*(N_Yes/n)*(N_No/n))
    data_yes = N_Yes                                                              #Formulae mentioned in book eq(19.18)
    data_no = N_No                                                              
    
    ci = [row for row in D if row[index] == value]                            #class                            
    prob_ci_in_y = float(len(ci))/float(data_yes)                             #class in region yes
    prob_ci_in_n = float(len(ci))/float(data_no)                             #Class in region no
    sum_probability = float(fabs(prob_ci_in_y-prob_ci_in_n))            #probability in absolute value
    CART = float(cart_1*sum_probability)
    
    return CART
    
def bestSplit(D, criterion):
    
    dict_2 = {'index': 0, 'best_attribute' : 0, 'comparator': 0}             #dictionary takes data initialized 0 with respect to index
    
    access_loop = (len(D.T)) - 1                                           #transpose of Data matrix                                          
    if criterion == 1:                                                      #split 1
        
        GINI_value_choice = 1.0                                                 
        for ind in range(access_loop):                                      #indexing in loop for iteration
            for row in D:
                value = row[ind]
                ge_new = float(G(D, ind, value))
                if ge_new < GINI_value_choice:
                    GINI_value_choice = ge_new
                    print ge_new
                    dict_2['index'] = ind
                    dict_2['best_attribute'] = value
                    dict_2['comparator'] = GINI_value_choice
        return dict_2['index'], dict_2['best_attribute'], dict_2['comparator']
    if criterion == 2:                                                          #split 2
        cart_2 = 0.0
        for ind in range(access_loop):
            for row in D:
                value = row[ind]
                c = float(CART(D, ind, value))
                if c > cart_2:
                    cart_2 = c
                    dict_2['index'] = ind
                    dict_2['best_attribute'] = value
                    dict_2['comparator'] = cart_2
        return dict_2['index'], dict_2['best_attribute'],dict_2['comparator']
    if criterion ==3:                                                               #split 3
        information_gain = 0.0
        for ind in range(access_loop):
            for row in D:
                value = row[ind]
                ig_comparator = (IG(D,ind,value))
                if ig_comparator > information_gain:
                    information_gain = ig_comparator
                    dict_2['index'] = ind
                    dict_2['best_attribute'] = value
                    dict_2['comparator'] = information_gain
        return dict_2['index'], dict_2['best_attribute'],dict_2['comparator']
    

def classifyIG(D,T):
    value_1,value_2,value_3 = bestSplit(D,3)      #values for IG
    k1,k2,k3 = bestSplit(T,3)
    h= IG(T,value_1,value_2)
    error = h-value_3
    print "Classification error for Info Gain", error
    
  
def classifyG(D,T):
    
    value_1,value_2,value_3 = bestSplit(D,2)              #value for Gini
    k1,k2,k3 = bestSplit(T,2)
    h1= G(T,value_1,value_2)
    error = h1-value_3
    print "Classification error for Gini", error
    
         

def classifyCART(D,T):
    value_1,value_2,value_3 = bestSplit(D,2)
    k1,k2,k3 = bestSplit(T,2)
    h2= CART(T,value_1,value_2)                         #value for CART
    error = h2-value_3
    print "Classification error for CART", error
    
            
def load(filename):
    D = np.loadtxt(filename, delimiter = ',')
    
    return D



def main():
    
    filename = 'train.txt'
    D = load(filename)
    
    v1,v2,v3 = bestSplit(D,1)
    print "gini_split"
    print "best index,value ,measure",v1,v2,v3
    
    
    v1,v2,v3 = bestSplit(D, 2)
    print "CART"
    print "best index, value, measure",v1,v2,v3
    
    
    
    v1,v2,v3 = bestSplit(D, 3)
    print "info_gain"
    print "best measure, value, measure",v1,v2,v3
    

    
    filename_T = 'test.txt'
    T = load(filename_T)

    classifyCART(D,T)
    classifyG(D,T)
    classifyIG(D,T)
    
    
if __name__=="__main__": 
	"""__name__=="__main__" when the python script is run directly, not when it 
	is imported. When this program is run from the command line (or an IDE), the 
	following will happen; if you <import HW2>, nothing happens unless you call
	a function.
	"""
	main()
