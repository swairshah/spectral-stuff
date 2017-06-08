import argparse
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import matrix_rank, svd
import networkx as nx
import math
import sys
import os
import re
from conjugate_gs import *
from matrix_stuff import *

def GenDumbbellGraph(n1, n2):
    """ generates graph with two clusters of n1-clique
    and n2-clique with one node from each clique 
    connected to each other
    """
    G = nx.complete_graph(n1)
    H = nx.complete_graph(n2)

    mapping = {}
    for i in range(n2):
        mapping[i] = i+n1
    H = nx.relabel_nodes(H, mapping=mapping)

    I = nx.union(G,H)
    I.add_edge(n1-1,n1)
    I.weighted = False
    #set weight to 1
    for e in I.edges_iter():
        I.add_edge(e[0],e[1], weight = 1)

    return I

def IncidenceMatrix(G):
    n = len(G.nodes())
    e = len(G.edges())
    X = np.zeros((n,e))
    for idx, edge in enumerate(G.edges(data = True)):
        X[edge[0], idx] = 1
        X[edge[1], idx] = -1
    return X

def DegreeMatrix(G):
    degrees = nx.degree(G).values()
    matrix = np.diag(list(degrees))
    return matrix

def AdjacancyMatrix(G):
    return nx.attr_matrix(G)[0]

def Laplacian(G):
    E = IncidenceMatrix(G)
    L = E.dot(E.T)
    # Sanity Check
    # L should be same as 
    # DegreeMatrix(G) - AdjacancyMatrix(G)
    # Laplacian generated with nx : 
    # scipy.sparse.csr_matrix.todense(nx.laplacian_matrix(G))
    return L

def Draw(G):
    pos = nx.fruchterman_reingold_layout(G)
    nx.draw_networkx(G, node_color = 'orange', alpha = 0.6, pos = pos)
    plt.show()


def test_pinv():
    G = GenDumbbellGraph(7,8)
    L = Laplacian(G)
    E = IncidenceMatrix(G)
    L_plus = np.linalg.pinv(L)

    print("with L ", norm(ConjugatePinv(L, L) - L_plus))
    print("with E ", norm(ConjugatePinv(E, L) - L_plus))

    ES = E[:, sort_cols_by_norm(E)]
    print("sorted E ", norm(ConjugatePinv(ES, L) - L_plus))

    EL = np.zeros(E.shape)
    n = E.shape[1]; ES = E[:, np.random.randint(n, size = n)]
    print("shuffled E ", norm(ConjugatePinv(EL, L) - L_plus))

    EC = E[:, sort_cols_conjugate(E, L)]
    print("conj sorted E ", norm(ConjugatePinv(EC, L) - L_plus))

    I = np.eye(15)
    R = np.random.uniform(size = (15,50))
    print("Identity",norm(ConjugatePinv(I, L) - L_plus))
    print("Random", norm(ConjugatePinv(R, L) - L_plus))

    RL = np.zeros(R.shape)
    n = RL.shape[1]; RS = RL[:, np.random.randint(n, size = n)]
    print("Random shuffled", norm(ConjugatePinv(RS, L) - L_plus))

    n = L.shape[1]; RL[:,range(n)] = L[:,range(n)]
    print("Random + L cols (first)", norm(ConjugatePinv(RL, L) - L_plus)) 

    n = L.shape[1]; RL[:,range(20,n+20)] = L[:,range(n)]
    print("Random + L cols (middle)", norm(ConjugatePinv(RL, L) - L_plus)) 

if __name__ == "__main__":
    #G = GenDumbbellGraph(7,8)
    G = nx.complete_graph(10)
    L = Laplacian(G)
    E = IncidenceMatrix(G)
    L_plus = np.linalg.pinv(L)

    print(np.linalg.svd(L)[1])

    #for i in range(E.shape[1]):
    #    print(norm(proj(E[:,i], L)))

    for i in range(E.shape[1]):
        print(norm(proj_ortho_to_null(E[:,i], L)))

    print()
    for i in range(L.shape[1]):
        print(norm(proj_ortho_to_null(L[:,i], L)))


