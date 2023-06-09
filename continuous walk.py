# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:59:18 2023

@author: ronan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import sympy
import networkx as nx
import matplotlib.animation as ani
import functools as fu
from operator import itemgetter
import random as random

'''
variables
'''
nodes = 50  #number of nodes for the generated graphs
edges = nodes*4  #number of edges for the random graph
degree = 4 #degree of new nodes in barabasi albert graph



'''
graphs used for this project
'''
circle = nx.cycle_graph(nodes)          #cycle graph
lines = nx.path_graph(nodes)            #line graph (integer line)
full = nx.complete_graph(nodes)         #complete graph 

rnd = nx.gnm_random_graph(nodes,edges)  #random graph with a set number of edges
rnd_large  = rnd.subgraph(max(nx.connected_components(rnd), key=len)).copy()    #largest fully connected part of the random graph above
rnd2 = nx.gnm_random_graph(nodes,edges) #second random graph
rnd_large2  = rnd2.subgraph(max(nx.connected_components(rnd2), key=len)).copy() #second fully connected random graph

barabasi = nx.barabasi_albert_graph(nodes,degree)       #barabasi albert graph
erdos = nx.erdos_renyi_graph(nodes,0.5)                 #erdos renyi graph

square_cross = nx.Graph()       #square graph with a central vertex
square_cross.add_nodes_from([0,1,2,3,4])
square_cross.add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)])

test_graph_1 = nx.Graph()       #generating two graphs from kronecker sum of small graphs in different way for isomorph check
test_graph_1.add_nodes_from([0,1,2,3,4,5,6,7,8,9,10,11])
test_graph_1.add_edges_from([(0,1),(0,4),(1,2),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,8),(5,6),(5,7),(5,9),(6,7),(6,10),(7,11),(8,9),(9,10),(9,11),(10,11)])
test_graph_2 = nx.Graph()
test_graph_2.add_nodes_from([0,1,2,3,4,5,6,7,8,9,10,11])
test_graph_2.add_edges_from([(0,1),(0,3),(1,2),(1,4),(2,5),(3,4),(3,6),(3,9),(4,5),(4,7),(4,10),(5,8),(5,11),(6,7),(6,9),(7,8),(7,10),(8,11),(9,10),(10,11)])

graph_a = nx.Graph()        #graphs used for plotting
graph_a.add_nodes_from([0,1,2])
graph_a.add_edges_from([(0,1),(1,2)])
graph_b = nx.Graph()
graph_b.add_nodes_from([0,1,2,3])
graph_b.add_edges_from([(0,1),(1,2),(1,3),(2,3)])

dark_state_cross = np.array([0,1,-1,-1,1])*(np.sqrt(2)/4) - np.array([0,1,1,-1,-1])*(np.sqrt(2)/4) #eigenstate of the square graph with central node







'''
functions used for this project
'''

#random walk functions
def continuous_rw(graph,startpos,time):
    '''
    classical continuous random walk with a single starting position
    '''
    hamiltonian = nx.laplacian_matrix(graph).toarray()      #making the hamiltonian as the laplacian of the graph
    start = np.zeros(graph.number_of_nodes())
    start[startpos] = 1
    return scipy.sparse.linalg.expm_multiply(-time*hamiltonian,start)

def continuous_qw(graph,startpos,time): 
    '''
    continuous time quantum random walk with a single starting position
    '''
    hamiltonian = nx.laplacian_matrix(graph).toarray()      #making the hamiltonian as the laplacian of the graph
    start = np.zeros(graph.number_of_nodes())
    start[startpos] = 1
    return scipy.sparse.linalg.expm_multiply(complex(0,-1)*time*hamiltonian,start)

def continuous_qw_ad(graph,startpos,time): 
    '''
    continuous time quantum random walk that uses the adjacency matrix instead of the laplacian for the hamiltonian
    '''
    hamiltonian = nx.adjacency_matrix(graph).toarray()
    start = np.zeros(graph.number_of_nodes())
    start[startpos] = 1
    return scipy.sparse.linalg.expm_multiply(complex(0,-1)*time*hamiltonian,start)

def continuous_qw_ad_startmix(graph,start,time): 
    '''
    continuous time quantum random walk with a starting position that is a superposition with adjacency matrix as hamiltonian
    '''
    hamiltonian = nx.adjacency_matrix(graph).toarray()
    return scipy.sparse.linalg.expm_multiply(complex(0,-1)*time*hamiltonian,start)

def probability_plot_qw(position,lab): 
    '''
    function that returns and plots the proabability distribution of a quantum random walk over the vertices
    ''' 
    probability = np.real(position*np.conjugate(position))
    plt.plot(np.linspace(1,nodes,nodes),probability,label = lab)
    return probability

def probability_qw(position): 
    '''
    function that returns the probability distribution without plotting
    ''' 
    probability = np.real(position*np.conjugate(position))
    return probability

def non_perfect_continuous_qw(graph,startpos,time,std): 
    '''
    continuous time quantum random walk with an uncertainty in the evolution time
    '''
    hamiltonian = nx.laplacian_matrix(graph).toarray()
    start = np.zeros(graph.number_of_nodes())
    start[startpos] = 1
    evolution_time = np.random.normal(time,std)
    if evolution_time >0: 
        return scipy.sparse.linalg.expm_multiply(complex(0,-1)*evolution_time*hamiltonian,start)
    else:
        return scipy.sparse.linalg.expm_multiply(complex(0,-1)*0*hamiltonian,start)

def non_perfect_continuous_qw_ad_startmix(graph,start,time,std): 
    '''
    continuous time quantum random walk with a starting position that is a superposition with an uncertainty in the evolution time
    '''
    hamiltonian = nx.adjacency_matrix(graph).toarray()
    evolution_time = np.random.normal(time,std)
    if evolution_time >0: 
        return scipy.sparse.linalg.expm_multiply(complex(0,-1)*evolution_time*hamiltonian,start)
    else:
        return scipy.sparse.linalg.expm_multiply(complex(0,-1)*0*hamiltonian,start)



#global measure functions
def total_measure(graph,startpos,freq,time): 
    '''
    function that returns the probability distribution after a time with a periodic measurement
    (slow version of this function a faster one is given later)
    ''' 
    if time<=freq:
        return probability_qw(continuous_qw(graph, startpos, time))
    else:
        prob_old = probability_qw(continuous_qw(graph, startpos, freq))
        prob_new = np.zeros(graph.number_of_nodes())
        for i in range(time//freq-1):
            for a in range(graph.number_of_nodes()):
                prob_new = prob_new + probability_qw(continuous_qw(graph, a, freq))*prob_old[a]
            prob_old = prob_new.copy()
            prob_new = np.zeros(graph.number_of_nodes())
        if time%freq == 0:
            return prob_old
        else: 
            for i in range(graph.number_of_nodes()):
                prob_new = prob_new + probability_qw(continuous_qw(graph, i, time%freq))*prob_old[i]
            return prob_new
        
        
def G_matrix(graph,freq): 
    '''
    returns the G operator from which we can directly find the probability distribution after n measurements
    (slow version of this function faster one is given later)
    '''      
    G = np.zeros((graph.number_of_nodes(),graph.number_of_nodes()))
    hamiltonian_exp = scipy.sparse.linalg.expm(complex(0,-1)*freq*nx.adjacency_matrix(graph).toarray())
    for i in range(graph.number_of_nodes()):
        for a in range(graph.number_of_nodes()):
            xprime_vec = np.zeros((graph.number_of_nodes(),1))
            xprime_vec[a,0] = 1
            x_trans = np.zeros((1,graph.number_of_nodes()))
            x_trans[0,i] = 1
            G = G + np.real(hamiltonian_exp[a][i]*np.conjugate(hamiltonian_exp[a][i]))*xprime_vec@x_trans
    return G

def G_matrix_fast(graph,freq): 
    '''
    efficient function that returns the G operator from which we can directly find the probablity distribution after n measurements
    '''
    G = scipy.sparse.linalg.expm(complex(0,-1)*freq*nx.adjacency_matrix(graph).toarray())*np.conjugate(scipy.sparse.linalg.expm(complex(0,-1)*freq*nx.adjacency_matrix(graph).toarray()))
    return np.real(G)

def non_perfect_G_matrix(graph,freq,std): 
    '''
    efficient function that returns the G operator with an uncertainty in the measurement time
    '''
    matrix = scipy.sparse.linalg.expm(complex(0,-1)*np.random.normal(freq,std)*nx.adjacency_matrix(graph).toarray())
    G = matrix*np.conjugate(matrix)
    return np.real(G)

def Total_measure_fast_plot(graph,startpos,freq,measurements,lab): 
    '''
    fast and efficient function that returns and plots the probablity distribution after n grlobal measurements with a specific frequency
    ''' 
    x = np.zeros(graph.number_of_nodes())
    x[startpos] = 1
    G = G_matrix_fast(graph, freq)
    plt.plot(np.linalg.matrix_power(G,measurements)@x,label = lab)
    return np.linalg.matrix_power(G,measurements)@x

def non_perfect_total_measure_plot(graph,startpos,freq,std,measurements,lab): 
    '''
    function that returns and plots the probablity distribution after n global measuements with an uncertainty in the measurement frequency
    '''
    x = np.zeros(graph.number_of_nodes())
    x[startpos] = 1
    G = non_perfect_G_matrix(graph, freq, std)
    for i in range(measurements-1):
        G = non_perfect_G_matrix(graph, freq, std)@G    
    plt.plot(G@x,label = lab,linestyle = "--")
    return G@x

def Total_measure_fast_prob(graph,startpos,freq,measurements): 
    '''
    fast and efficient function that returns the probablity distribution after n grlobal measurements with a specific frequency without plotting
    ''' 
    x = np.zeros(graph.number_of_nodes())
    x[startpos] = 1
    G = G_matrix_fast(graph, freq)
    return np.linalg.matrix_power(G,measurements)@x

def non_perfect_total_measure_prob(graph,startpos,freq,std,measurements): 
    '''
    function that returns the probablity distribution after n global measuements with an uncertainty in the measurement frequency without plotting
    '''
    x = np.zeros(graph.number_of_nodes())
    x[startpos] = 1
    G = non_perfect_G_matrix(graph, freq, std)
    for i in range(measurements-1):
        G = non_perfect_G_matrix(graph, freq, std)@G    
    return G@x

def detect_prob(number, graph, freq,target,start): 
    '''
    detection probability in a single target vertex after n global measuements with a specific frequency
    '''  
    target_trans = np.zeros((1,graph.number_of_nodes()))
    target_trans[0,target] = 1
    target_vec = np.zeros((graph.number_of_nodes(),1))
    target_vec[target,0] = 1
    start_vec = np.zeros((graph.number_of_nodes(),1))
    start_vec[start,0] = 1
    Fn = target_trans@np.linalg.matrix_power(G_matrix_fast(graph, freq)@(np.identity(graph.number_of_nodes())-target_vec@target_trans),number-1)@G_matrix_fast(graph, freq)@start_vec
    return Fn[0][0]

def non_perfect_detect_prob(number,graph,freq,std,target,start): 
    '''
    detection probability in a single target vertex after n global measurements with an uncertainty in the measurement frequency
    '''
    target_trans = np.zeros((1,graph.number_of_nodes()))
    target_trans[0,target] = 1
    target_vec = np.zeros((graph.number_of_nodes(),1))
    target_vec[target,0] = 1
    start_vec = np.zeros((graph.number_of_nodes(),1))
    start_vec[start,0] = 1
    G = np.identity(graph.number_of_nodes())
    D_ = (np.identity(graph.number_of_nodes())-target_vec@target_trans)
    for i in range(number-1):
        G = (non_perfect_G_matrix(graph, freq, std)@D_)@G  
    Fn = target_trans@G@non_perfect_G_matrix(graph, freq, std)@start_vec
    return Fn[0][0]


def detect_plot(number_measure,freq_list,graph,target,start): 
    '''
    plotting the detection probability against the number of measurements for different frequencies
    '''
    for a in freq_list:
        y = np.zeros(number_measure)
        for i in range(number_measure):
            y[i] = detect_prob(i+1, graph, a, target, start)
        plt.plot(y,label=a)
    plt.legend()
    plt.show()
    
def cum_detect_prob(number_measure,freq,graph,target,start): 
    '''
    returning the cumulative detection probability after n measurements
    '''
    y = 0
    for i in range(number_measure):
        y = detect_prob(i+1, graph, freq, target, start) + y
    return y

    
def cum_detect_plot(number_measure,freq_list,graph,target,start): 
    '''
    plotting the cumulative detection probability against the number of measurements for different frequencies
    ''' #cumulative detection probability plotted
    for a in freq_list:
        y = np.zeros(number_measure)
        for i in range(number_measure):
            y[i] = detect_prob(i+1, graph, a, target, start) + y[i-1]
        plt.plot(y,label=a)
    plt.legend()
    plt.show()

def non_perfect_cum_detect_plot(number_measure,freq_list,std,graph,target,start): 
    '''
    plotting the cumulative detecton probability against the number of measurements with an uncertainty in the measurement frequency
    '''
    for a in freq_list:
        y = np.zeros(number_measure)
        for i in range(number_measure):
            y[i] = non_perfect_detect_prob(i+1, graph, a,std, target, start) + y[i-1]
        plt.plot(y,label=a)
    plt.legend()
    plt.show()
    
def detect_plot_freq(measure,graph,maxfreq,lenght,target,start): 
    '''
    plotting the cumulative detection prbability on n measurements against the frequency
    '''
    y = np.zeros(lenght)
    x = np.linspace(0,maxfreq,lenght)
    for i in range(lenght):
        y[i] = cum_detect_prob(measure, i*maxfreq/lenght, graph, target, start)
    plt.plot(x,y)
    plt.show()




#local measure functions
def local_measure_matrix(graph,freq,target): 
    '''
    survival operator for a specific target state for local measurements
    '''    
    target_trans = np.zeros((1,graph.number_of_nodes()))
    target_trans[0,target] = 1
    target_vec = np.zeros((graph.number_of_nodes(),1))
    target_vec[target,0] = 1
    target_matrix = target_vec@target_trans
    G = (np.identity(graph.number_of_nodes())-target_matrix)@scipy.sparse.linalg.expm(complex(0,-1)*freq*nx.adjacency_matrix(graph).toarray())
    return G
    
def non_perfect_measure_matrix(graph,freq,std,target): 
    '''
    survival operator with an uncertainty in the measurement time
    '''
    target_trans = np.zeros((1,graph.number_of_nodes()))
    target_trans[0,target] = 1
    target_vec = np.zeros((graph.number_of_nodes(),1))
    target_vec[target,0] = 1
    target_matrix = target_vec@target_trans
    G = (np.identity(graph.number_of_nodes())-target_matrix)@scipy.sparse.linalg.expm(complex(0,-1)*np.random.normal(freq,std)*nx.adjacency_matrix(graph).toarray())
    return G

def survival_local_measure(number,graph,freq,target,start): 
    '''
    calculating the survival probability on a specific target 
    ''' 
    start_vec = np.zeros((graph.number_of_nodes(),1))
    start_vec[start,0] = 1
    Sn = np.linalg.matrix_power(local_measure_matrix(graph,freq,target), number)@start_vec
    return sum(np.real(Sn*np.conjugate(Sn)))

def non_perfect_survival(number,graph,freq,std,target,start): 
    '''
    calculating the survival probability on a specific target with an uncertainty in the measurement time
    '''
    start_vec = np.zeros((graph.number_of_nodes(),1))
    start_vec[start,0] = 1
    G = non_perfect_measure_matrix(graph, freq, std,target)
    for i in range(number-1):
        G = non_perfect_measure_matrix(graph, freq, std,target)@G 
    Sn = G@start_vec
    return sum(np.real(Sn*np.conjugate(Sn)))

def cum_survival_plot(number_measure,freq_list,graph,target,start): 
    '''
    plotting the cumulative survival probability against the number of measurements
    '''   
    for a in freq_list:
        y = np.zeros(number_measure)
        lab = "T = "+str(a)
        for i in range(number_measure):
            y[i] = survival_local_measure(i, graph, a, target, start)
        plt.plot(y,label=lab)
    plt.legend()
    plt.show()
    
def non_perfect_cum_survival_plot(number_measure,freq_list,std,graph,target,start): 
    '''
    plotting the cumulative survial probability against the number of measurments with an uncertainty in the measurement frequency
    '''
    for a in freq_list:
        y = np.zeros(number_measure)
        for i in range(number_measure):
            y[i] = non_perfect_survival(i, graph, a,std, target, start)
        plt.plot(y,label=a)
    plt.legend()
    plt.show()

def survival_plot_freq(measure,graph,maxfreq,lenght,target,start): 
    '''
    plotting the survival probability after n measurements against the frequency
    '''
    y = np.zeros(lenght)
    x = np.linspace(0,maxfreq,lenght)
    lab = "n = "+str(measure)
    for i in range(lenght):
        y[i] = survival_local_measure(measure,graph,i*maxfreq/lenght,target, start)
    plt.plot(x,y,label = lab)
    plt.show()
 
    
 
#isomorphism functions    
def create_isomporh(graph): 
    '''
    create an isomorphic copy of a graph
    '''
    node_mapping = dict(zip(graph.nodes(), sorted(graph.nodes(), key=lambda k: random.random())))
    graph_new = nx.relabel_nodes(graph, node_mapping)
    return graph_new

def create_similar(graph,num):
    '''
    create a copy of a graph with num extra edges with a built in max tries to avoid infinite looping for complete graphs
    '''
    test = graph.copy()
    for i in range(5*num):
        a = random.randint(0,graph.number_of_nodes()-1)
        b = random.randint(0,graph.number_of_nodes()-1)
        if a != b and ((a,b)and(b,a)) not in graph.edges():
            test.add_edges_from([(a,b)])
        if graph.number_of_edges()+num-1 < test.number_of_edges():
            #print(graph.number_of_edges(),test.number_of_edges()+num)
            return test
    return test

def isomorph_test(graph1,graph2,time): 
    '''
    difference in graph certificate for two graphs at a specific time
    '''
    if graph1.number_of_nodes() != graph2.number_of_nodes():
        print("not isomorph")
        return
    else :
        n = graph1.number_of_nodes()
        prob1 = np.zeros(n)
        prob2 = np.zeros(n)
        for i in range(n):
            prob1[i] = probability_qw(continuous_qw(graph1, i, time))[i]
            prob2[i] = probability_qw(continuous_qw(graph2, i, time))[i]
        graphcert1 = np.sort(prob1)
        graphcert2 = np.sort(prob2)
        return 0.5*np.sum((abs(graphcert1-graphcert2)))

def isomorph_plot_isomorph(tests,time): 
    '''
    plotting the difference in graph certificate for random graphs that are isomorph for a number of tests
    '''
    result = np.zeros(tests)
    for i in range(tests):
        rnd = nx.gnm_random_graph(nodes,edges)
        rnd_large  = rnd.subgraph(max(nx.connected_components(rnd), key=len)).copy()
        result[i] = isomorph_test(rnd_large, create_isomporh(rnd_large), time)
    plt.plot(result,label="isomorph")
    
def isomorph_plot_similar(tests,time,num):  
    '''
    plotting the difference in graph certificate for random graphs that are similar for a number of tests
    '''
    result = np.zeros(tests)
    for i in range(tests):
        rnd = nx.gnm_random_graph(nodes,edges)
        rnd_large  = rnd.subgraph(max(nx.connected_components(rnd), key=len)).copy()
        result[i] = isomorph_test(rnd_large, create_similar(rnd_large,num[i]), time)
    plt.plot(result,label="similar")
    
def isomorph_plot_random(tests, time): 
    '''
    plotting the difference in graph certificate for random graphs that are random for a number of tests
    '''
    result = np.zeros(tests)
    for i in range(tests):
        rnd1 = nx.gnm_random_graph(nodes,edges)
        rnd_large1  = rnd1.subgraph(max(nx.connected_components(rnd1), key=len)).copy()
        rnd2 = nx.gnm_random_graph(nodes,edges)
        rnd_large2  = rnd2.subgraph(max(nx.connected_components(rnd2), key=len)).copy()
        result[i] = isomorph_test(rnd_large1, rnd_large2, time)
    plt.plot(result,label="random")

def isomorph_plot_times(tests,interval,graph1,graph2,lab): 
    '''
    plotting the difference in graph certificate for different times
    '''
    result = np.zeros(tests)
    for i in range(tests):
        result[i] = isomorph_test(graph1, graph2,(i+1)*interval)
    plt.plot(result,label=lab)

def non_perfect_isomorph_test(graph1,graph2,time,std): 
    '''
    calculating the difference in graph certificate with an uncertainty in the time
    '''
    if graph1.number_of_nodes() != graph2.number_of_nodes():
        print("not isomorph")
        return
    else :
        n = graph1.number_of_nodes()
        prob1 = np.zeros(n)
        prob2 = np.zeros(n)
        for i in range(n):
            prob1[i] = probability_qw(non_perfect_continuous_qw(graph1, i, time,std))[i]
            prob2[i] = probability_qw(non_perfect_continuous_qw(graph2, i, time,std))[i]
        graphcert1 = np.sort(prob1)
        graphcert2 = np.sort(prob2)
        return 0.5*np.sum((abs(graphcert1-graphcert2)))

def non_perfect_isomorph_plot_times(tests,interval,graph1,graph2,std,lab): 
    '''
    plotting the difference in graph certificate for different times with uncertainty
    '''
    result = np.zeros(tests)
    for i in range(tests):
        result[i] = non_perfect_isomorph_test(graph1, graph2,(i+1)*interval,std)
    plt.plot(result,label=lab)




#centrality functions
def centrality_test(graph,time,measures): 
    '''
    centrality measure based on the continuous time quantum walk
    '''
    start = np.ones(graph.number_of_nodes())/np.sqrt(graph.number_of_nodes())
    result = np.zeros(graph.number_of_nodes())
    for i in range(1,measures+1):
        a = continuous_qw_ad_startmix(graph, start, time/i)
        result = result + a*np.conjugate(a)/measures
    return np.real(result)

def eigenvector_central(graph): 
    '''
    classical centrality measure based on the eigenvector of the adjacency matrix
    '''
    test = np.linalg.eig(nx.adjacency_matrix(graph).toarray())[1]
    central = np.zeros(graph.number_of_nodes())
    for i in range(graph.number_of_nodes()):
        central[i] = test[i][0]
    return abs(central)

def non_pefect_centrality_test(graph,time,measures,std): 
    '''
    centrality measure based on the continuous time quantum walk with an uncertainty
    '''
    start = np.ones(graph.number_of_nodes())/np.sqrt(graph.number_of_nodes())
    result = np.zeros(graph.number_of_nodes())
    for i in range(1,measures+1):
        a = non_perfect_continuous_qw_ad_startmix(graph, start, time/i,std)
        result = result + a*np.conjugate(a)/measures
    return np.real(result)




#spatial search functions
def spatial_search(graph,target,start,time): 
    '''
    detection probability with set target and initial states at a specific time
    '''
    startpos = np.identity(graph.number_of_nodes())*0
    startpos_vec = np.zeros(graph.number_of_nodes())
    targetpos = np.identity(graph.number_of_nodes())*0
    jumprate = 1/max(np.linalg.eig(nx.adjacency_matrix(graph).toarray())[0])
    for i in target:
        targetpos[i,i] = 1
    for i in start:
        startpos[i,i] = 1
        startpos_vec[i] = 1/np.sqrt(len(start))
    hamiltonian = - startpos - targetpos - nx.adjacency_matrix(graph).toarray()*jumprate
    probdistr = scipy.sparse.linalg.expm_multiply(complex(0,-1)*time*hamiltonian,startpos_vec)
    probability = np.real(probdistr*np.conjugate(probdistr))
    P = 0
    for i in target:
        P = P + probability[i]
    return P

def spatial_search_plot(graph,target,start,maxtime,times,lab): 
    '''
    plot the detection probability in the target states against the time
    '''
    lin = np.linspace(0,maxtime,times)
    prob = np.zeros(times)
    for i in range(times):
        prob[i] = spatial_search(graph, target, start, lin[i])
    plt.plot(lin,prob,label = lab)
    return prob

def non_perfect_spatial_search_plot(graph,target,start,maxtime,times,std,lab): 
    '''
    plot the detection probability in the target states against the time with an uncertainty in each calculation
    '''
    lin = np.linspace(0,maxtime,times)
    prob = np.zeros(times)
    for i in range(times):
        prob[i] = spatial_search(graph, target, start, np.random.normal(lin[i],std))
    plt.plot(lin,prob,label = lab)
    return prob

def spatial_search_kappa(graph,target,start,time,kappa): 
    '''
    detection probability for different values of kappa in the hamiltonian
    '''
    startpos = np.identity(graph.number_of_nodes())*0
    startpos_vec = np.zeros(graph.number_of_nodes())
    targetpos = np.identity(graph.number_of_nodes())*0
    for i in target:
        targetpos[i,i] = 1
    for i in start:
        startpos[i,i] = 1
        startpos_vec[i] = 1/np.sqrt(len(start))
    hamiltonian = - startpos - targetpos - nx.adjacency_matrix(graph).toarray()*kappa
    probdistr = scipy.sparse.linalg.expm_multiply(complex(0,-1)*time*hamiltonian,startpos_vec)
    probability = np.real(probdistr*np.conjugate(probdistr))
    P = 0
    for i in target:
        P = P + probability[i]
    return P

def spatial_search_plot_kappa(graph,target,start,maxtime,times,kappa,lab): 
    '''
    plot the detection probability in the target states against the time for different values of kappa in the hamiltonian
    '''
    lin = np.linspace(0,maxtime,times)
    prob = np.zeros(times)
    for i in range(times):
        prob[i] = spatial_search_kappa(graph, target, start, lin[i],kappa)
    plt.plot(lin,prob,label = lab)
    return prob

def spatial_search_prob(graph,target,start,maxtime,times): 
    '''
    returns detection probability against the time without plotting
    '''
    lin = np.linspace(0,maxtime,times)
    prob = np.zeros(times)
    for i in range(times):
        prob[i] = spatial_search(graph, target, start, lin[i])
    return prob

def spatial_search_time(target,start,times,freq,maxtime): 
    '''
    returns the times it takes for a particle to be detected with non collapsing measurements at a set frequency
    '''
    results = np.zeros(times)
    for i in range(times):
        graph = nx.erdos_renyi_graph(nodes,0.5)
        a = 0
        measure = 0
        while a == 0:
            if random.random() <= spatial_search(graph, target, start, measure) or measure > maxtime:
                results[i] = measure
                a = 1
            else:
                measure = measure + freq
    return results




#effect of uncertainty functions    
def av_diff_test_total_measure(graph,start,measures,freq,std,tests): 
    '''
    returns the absolut difference between theoretical measurement and measurement with uncertainty for n global measurements
    '''
    result = np.zeros(tests)
    a = Total_measure_fast_prob(graph, start, freq, measures)
    for i in range(tests):
        result[i] = np.sum(abs(a-non_perfect_total_measure_prob(graph, start, freq, std, measures)))
    return np.mean(result),np.std(result)

def av_diff_network_central(time,measures,std,tests): 
    '''
    returns the abolute difference and error between the theoretical measurement and measurement with uncertainty in network centrality
    '''
    result = np.zeros(tests)
    test =  np.zeros(tests)
    for i in range(tests):
        graph = nx.barabasi_albert_graph(nodes,degree)
        a = centrality_test(graph, time, measures)
        b = non_pefect_centrality_test(graph, time, measures, std)
        result[i] = np.sum(abs(a-b))
        test[i] = np.sum(abs(a-b)/a)/len(a)
#        test[i] = np.sum(abs(a-b))/np.sum(a)
    return np.mean(result),np.std(result),np.mean(test),np.std(test)

def av_diff_graph_isomorph(graph1,graph2,time,std,tests): 
    '''
    returns the abolute difference and error between the theoretical measurement and measurement with uncertainty in graph isomorphism
    '''
    result = np.zeros(tests)
    test =  np.zeros(tests)
    a = isomorph_test(graph1, graph2, time)
    c = np.sum(a)
    for i in range(tests):
        b = non_perfect_isomorph_test(graph1, graph2, time, std)
        result[i] = np.sum(abs(a-b))
#        test[i] = np.sum(abs(a-b)/a)/len(a)
        if c != 0:
            test[i] = np.sum(abs(a-b))/np.sum(a)
    return np.mean(result),np.std(result),np.mean(test),np.std(test)

def opt_search(maxtime,states,steps,tests): 
    '''
    returns the optimal search time for an erdos renyi graph with set number of nodes
    '''
    result = np.zeros(tests)
    for i in range(tests):
        sample = random.sample(range(0,nodes-1), 2*states) 
        start = sample[:states]       
        target = sample[states:]
        graph = nx.erdos_renyi_graph(nodes,0.5)
        a = spatial_search_prob(graph, target, start, maxtime, steps)
        result[i] = np.argmax(a)*maxtime/steps
    return np.mean(result),np.std(result)

def av_diff_search(states,time,std,tests): 
    '''
    returns the abolute difference and error between the theoretical measurement and measurement with uncertainty in spatial search
    '''
    result = np.zeros(tests)
    test =  np.zeros(tests)
    test_2 = np.zeros(tests)
    for i in range(tests):
        graph = nx.erdos_renyi_graph(nodes,0.5)
        sample = random.sample(range(0,nodes-1), 2*states) 
        start = sample[:states]       
        target = sample[states:]
        a = spatial_search(graph, target, start, time)
        b = spatial_search(graph, target, start, np.random.normal(time,std))
        result[i] = a
#        test[i] = np.sum(abs(a-b)/a)/len(a)
        test[i] = b
        test_2[i] = abs(a-b)/a
#    plt.hist(test_2)
    return np.mean(result),np.std(result),np.mean(test),np.std(test),np.mean(test_2)





'''
animations
'''

fig, ax = plt.subplots()
#plt.figure(figsize=(5,3),dpi = 600)
line, = ax.plot([])

def animate_plot(frame_num,graph,speed,startpos): 
    '''
    animated plot with probabilities on the vertices
    '''  
    y = probability_qw(continuous_qw(graph,startpos,frame_num/speed))
    x = np.linspace(0,graph.number_of_nodes(),graph.number_of_nodes())
    ax.set_ylim(0,1.1*max(y))
    ax.set_xlim(0, graph.number_of_nodes())
    line.set_data((x, y))
    return line

def animate_plot_startmix(frame_num,graph,speed,start): 
    '''
    animated plot with probabilities on the vertices with superposition start positon
    '''  
    y = probability_qw(continuous_qw_ad_startmix(graph,start,frame_num/speed))
    x = np.linspace(0,graph.number_of_nodes()-1,graph.number_of_nodes())
    ax.set_ylim(0,1.1*max(y))
    ax.set_xlim(0, graph.number_of_nodes()-1)
    line.set_data((x, y))
    return line


def animate_graph(frame_num,graph,seed_,speed,startpos): 
    '''
    animated plot with probabilities on the vertices visualized on the graph with a heatmap
    ''' 
    ax.clear()
    nx.draw(graph,pos = nx.spring_layout(graph,seed=seed_),node_color=probability_qw(continuous_qw(graph,startpos,frame_num/speed))*100,cmap=plt.cm.Blues)

def animate_graph_startmix(frame_num,graph,seed_,speed,start): 
    '''
    animated plot with probabilities on the vertices visualized on the graph with a heatmap with a superpositon start position
    '''  
    ax.clear()
    nx.draw(graph,pos = nx.spring_layout(graph,seed=seed_),node_color=probability_qw(continuous_qw_ad_startmix(graph,start,frame_num/speed))*100,cmap=plt.cm.Blues)







# commands used for plotting an calculating values
'''
probability_plot_qw(continuous_qw(rnd_large, nodes//2, 6))          #plot of the probability over the nodes
print(probability_plot_qw(continuous_qw(lines, nodes//2, 10)))      #plot and print of the probabilities
nx.draw(rnd_large)                                                  #plot of the graph structure
Total_measure_fast_plot(lines,nodes//2,10,3)                        #plot after measurements with frequency 10

print(G_matrix_fast(lines,10))                                      #print of the G matrix for global measurement

cum_detect_plot(200,[0.1,0.5,1,5],lines,4,4)                        #detection plot for different frequencies for global measures

cum_survival_plot(100,[0.1,1,5,10],circle,0,2)                      #survival plot for different frequencies for local measures

cum_survival_plot(100,[0.1,1,5,10,20,100],square_cross,0,2)
cum_survival_plot(50,[0.1,1,5,np.pi/np.sqrt(5)],square_cross,0,2)
survival_plot_freq(5,square_cross,20,200,0,2)
survival_local_measure(30,square_cross,np.pi/np.sqrt(5),0,2)

survival_plot_freq(5,circle,10,400,0,1)
survival_plot_freq(5,circle,10,400,0,2)
survival_plot_freq(5,circle,10,400,0,3)

probability_plot_qw(continuous_qw(lines, nodes//2, 10))
plt.plot(continuous_rw(lines, nodes//2, 10))
plt.show()

isomorph = create_isomporh(rnd_large)
print(isomorph_test(rnd_large,isomorph,3))
print(non_perfect_isomorph_test(rnd_large,isomorph,3,0.1))
print(non_perfect_isomorph_test(rnd_large,isomorph,3,0.03))

non_perfect_cum_survival_plot(40,[0.1,1,np.pi/np.sqrt(5)],0.05,square_cross,0,2)
cum_survival_plot(40,[0.1,1,np.pi/np.sqrt(5)],square_cross,0,2)

non_perfect_cum_detect_plot(40,[0.1,1,5],0.05,lines,nodes//2,nodes//2)
cum_detect_plot(40,[0.1,1,5],lines,nodes//2,nodes//2)

non_perfect_total_measure_plot(lines,nodes//2,5,0.05,3)
Total_measure_fast_plot(lines,nodes//2,5,3)

probability_plot_qw(non_perfect_continuous_qw(lines, nodes//2, 20,0.2))
probability_plot_qw(continuous_qw(lines, nodes//2, 20))

nx.draw(rnd_large,node_color=centrality_test(rnd_large,500,5000)*(1/max(centrality_test(rnd_large,500,5000))),cmap=plt.cm.Blues)

centrality_test(barabasi,500,10000)*(1/max(centrality_test(barabasi,500,10000)))
barabasi.degree()

non_pefect_centrality_test(square_cross,5,100,5/100/100)
centrality_test(square_cross,5,100)

detect_plot_freq(50,square_cross,7,100,4,3)
[val for (node, val) in erdos.degree()]

#plt.plot(eigenvector_central(barabasi)*(1/np.sqrt(sum(eigenvector_central(barabasi)**2))),label = "eigenvector centrality")
#plt.plot(centrality_test(barabasi,300,2000)*(1/np.sqrt(sum(centrality_test(barabasi,300,2000)**2))),label = "CTQW centrality")
#plt.plot(np.array([val for (node, val) in barabasi.degree()])*(1/np.sqrt(sum(np.array([val for (node, val) in barabasi.degree()])**2))),label = "degree centrality")
#isomorph_plot_isomorph(20, 4)
#isomorph_plot_similar(20, 4,[1,1,2,1,2,3,2,4,1,1,3,1,2,1,1,1,2,3,1,2])
#isomorph_plot_random(20,4)
#isomorph_plot_times(15,1,rnd_large,create_isomporh(rnd_large),"isomorph")
#isomorph_plot_times(15,1,rnd_large,create_similar(rnd_large, 2),"similar")
#isomorph_plot_times(15,1,rnd_large,rnd_large2,"random")
#cum_detect_plot(100,[0.5,1,np.pi,5],circle,1,4)  
#cum_detect_plot(200,[0.1,0.5,1,5],lines,0,2)  
#plt.figure(figsize=(2.85,2.85),dpi = 600)
#cum_survival_plot(100,[0.1,1,1.35,5,10],square_cross,0,2)
#print(mean(spatial_search_plot(erdos, [12,34,2], [1,18,22], 25, 500,"optimal")*linspace(0,25,500)))
#non_perfect_spatial_search_plot(erdos, [12,34,2], [1,18,22], 25, 500,0.05,"uncertain")
#spatial_search_plot_kappa(erdos,[12,34,2], [1,18,22], 25, 500,0.03,"0.03")
#spatial_search_plot_kappa(erdos,[12,34,2], [1,18,22], 25, 500,0.01,"0.01")
#spatial_search_plot_kappa(erdos,[12,34,2], [1,18,22], 25, 500,0.1,"0.1")
#spatial_search_plot_kappa(erdos,[12,34,2], [1,18,22], 25, 500,1,"1")
#Total_measure_fast_plot(lines,nodes//2,5,3,"no uncertainty")
#non_perfect_total_measure_plot(lines,nodes//2,5,0.1,3,"deviation = 0.1")
#non_perfect_total_measure_plot(lines,nodes//2,5,0.1,3,"deviation = 0.1")
#non_perfect_total_measure_plot(lines,nodes//2,5,0.1,3,"deviation = 0.1")
#non_perfect_total_measure_plot(lines,nodes//2,5,0.1,3,"deviation = 0.1")

#spatial_search_plot(barabasi, [12,34,2], [1,18,22], 25, 500,"Barabasi-Albert")
#spatial_search_plot(rnd_large, [12,34,2], [1,18,22], 25, 500,"Random graph")
#plt.xlabel("vertex")
#plt.ylabel("probability")
#nx.draw(square_cross,pos = nx.spring_layout(square_cross,seed=5))
#plt.legend()
#plt.xlim(10,90)
#nx.draw(graph_b)
#plt.savefig("measure_uncertain_01.pdf",dpi = 600)
#plt.show()
isomorph_plot_times(10, 1, rnd_large, create_isomporh(rnd_large), "isomorph theory")
isomorph_plot_times(10, 1, rnd_large, create_similar(rnd_large,1), "similar theory")
isomorph_plot_times(10, 1, rnd_large, rnd_large2, "random theory")
non_perfect_isomorph_plot_times(10, 1, rnd_large, create_isomporh(rnd_large),0.1, "isomorph 0.1")
non_perfect_isomorph_plot_times(10, 1, rnd_large, create_similar(rnd_large,1),0.1, "similar 0.1")
non_perfect_isomorph_plot_times(10, 1, rnd_large, rnd_large2,0.1, "random 0.1")
plt.legend()
plt.xlabel("time step")
plt.ylabel("difference in graph certificate")
plt.savefig("gi_sigma01.pdf",dpi = 600)

plt.show()



#animations

anim = ani.FuncAnimation(fig, fu.partial(animate_plot, graph = lines,speed = 1,startpos = lines.number_of_nodes()//2), frames=200, interval=150)
anim = ani.FuncAnimation(fig, fu.partial(animate_graph,graph = rnd_large, seed_ = 9,speed = 10, startpos = rnd_large.number_of_nodes()//2), frames=200, interval=150)
anim = ani.FuncAnimation(fig, fu.partial(animate_graph,graph = barabasi, seed_ = 9,speed = 10,startpos = max(list(barabasi.degree()),key=itemgetter(1))[0]), frames=200, interval=150)
anim = ani.FuncAnimation(fig, fu.partial(animate_plot, graph = barabasi,speed = 10, startpos = max(list(barabasi.degree()),key=itemgetter(1))[0]), frames=200, interval=150)
plt.show()
'''
