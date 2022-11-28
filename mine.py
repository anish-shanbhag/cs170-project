## there are n team members
## the variable 'conflict' is a number that represents the conflict between two people
## conflict[2][3] is the conflict between person 2 and person 3

## split n people in k teams such that the total conflict is minimized
## return a list object such that list[0] is the team of person 0, list[1] is the team of person 1, etc.

import networkx as nx
import pulp as pl
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn import metrics

from starter import *

from scipy import sparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys

from sklearn.cluster import KMeans
from networkx.algorithms.community import kernighan_lin_bisection

def team_split_0(n, k, conflict):
    ## output a list partitioning n people into k groups 
    ## minimizing the internal edge weights
    # Assign a team to v with G.nodes[v]['team'] = team_id
    # Access the team of v with team_id = G.nodes[v]['team']
    # the weight of an edge is G.edges[i, j]['weight']
    
    # create a graph G with n nodes
    G = nx.Graph()
    G.add_nodes_from([i for i in range(n)])
    
    # add edges to G
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j, weight = conflict[i][j])
    
    # use k-means to partition G into k groups
    labels = KMeans(n_clusters=k).fit_predict(G)
    
    return labels

def team_split_1(n, k, conflict):
    # initialize the model
    model = pl.LpProblem("TeamSplit", pl.LpMinimize)

    # initialize the decision variables
    # x[i][j] is 1 if person i is on team j and 0 otherwise
    x = pl.LpVariable.dicts("x", [(i, j) for i in range(n) for j in range(k)], 0, 1, pl.LpInteger)

    # objective function
    model += pl.lpSum([conflict[i][j] * x[i][j] for i in range(n) for j in range(k)])

    # constraint: each person is on exactly one team
    for i in range(n):
        model += pl.lpSum([x[i][j] for j in range(k)]) == 1

    # constraint: the number of people on each team is the same
    for j in range(k):
        model += pl.lpSum([x[i][j] for i in range(n)]) == n / k

    # solve the model
    model.solve()

    # get the output
    output = [0] * n
    for i in range(n):
        for j in range(k):
            if x[i][j].value() == 1:
                output[i] = j
                break

    return output


def team_split_3(n, k, conflict):
    # create a graph with n nodes, each representing a person
    # each node has a weight equal to the sum of the conflicts with all other nodes
    # each edge has a weight equal to the conflict between the two nodes
    # find a partitioning of the graph into k groups with minimum total edge weight
    # the partitioning is a list of length n, where list[i] is the team of person i
    
    # create a graph with n nodes, each representing a person
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # each node has a weight equal to the sum of the conflicts with all other nodes
    for i in range(n):
        G.nodes[i]['weight'] = sum(conflict[i])
    
    # each edge has a weight equal to the conflict between the two nodes
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=conflict[i][j])
    
    # find a partitioning of the graph into k groups with minimum total edge weight
    # the partitioning is a list of length n, where list[i] is the team of person i
    partition = nx.algorithms.community.kernighan_lin_bisection(G, k)
    
    # convert the partitioning into a list
    output = [0] * n
    for i in range(k):
        for j in partition[i]:
            output[j] = i
    
    return output