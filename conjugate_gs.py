import numpy as np
from numpy.linalg import norm
from gram_schmidt import *
from matrix_stuff import *
import networkx as nx
from laplacian import *
tol = 1E-8

def swap_cols(A, i, j):
    tmp = A[:,i].copy()
    A[:,i] = A[:,j].copy()
    A[:,j] = tmp.copy()

def max_relative_score(P, L, e_idx):
    scores = []
    P_updated = np.zeros(P.shape)
    e = P[:, e_idx]
    for i in range(P.shape[1]):
        print(i, e_idx)
        if i == e_idx:
            P_updated[:, i] = P[:, e_idx]
            continue
        f = P[:, i]
        denom = f.T @ L @ f
        if denom > tol:
            numer = f.T @ L @ e
            f = f - (numer/denom)*e
            P_updated[:, i] = f
            scores.append(norm(f))
        else:
            scores.append(0)
    idx = np.argmax(-np.array(scores))
    return idx, P_updated

def conjugate_gs_ordered(E, L):
    P = E.copy()
    n, m = E.shape
    scores = np.diag(E.T @ L @ E)
    max_idx = np.argmax(scores)
    final_idx = np.zeros(m)

    swap_cols(P, 0, max_idx)
    final_idx[0] = max_idx
    for i in range(m-1):
        p = P[:, i]
        max_idx, P = max_relative_score(P, L, i)
        swap_cols(P, i+1, max_idx)
    final_idx[0] = max_idx
    return P

def conjugate_gs(X, A):
    #U = remove_zero_cols(modified_gs(X))
    idx = np.arange(X.shape[1])
    np.random.shuffle(idx)
    U = X[:, idx]
    r, c = U.shape
    P = U.copy()
    D = np.zeros(P.shape[1])
    for i in range(c):
        p = P[:, i]
        for j in range(i):
            q = P[:, j]
            denom = q.T.dot(A.dot(q))
            D[idx[i]] = denom
            if denom > tol:
                numer = q.T.dot(A.dot(p))
                p = p - (numer/denom)*q
                print(p)
    return P, D

def conjugate_pinv(X, A):
    P, _ = conjugate_gs(X, A)
    A_pinv = np.zeros(A.shape)
    for i in range(P.shape[1]):
        p = P[:, i]
        d = p.T.dot(A).dot(p)
        if d > tol:
            A_pinv += np.outer(p,p)/d
    return A_pinv

if __name__ == "__main__":
    G = GenDumbbellGraph(4, 3)
    L = Laplacian(G)
    E = IncidenceMatrix(G)
    k = 4
    #L_plus = np.linalg.pinv(L)
    P, D = conjugate_gs(E, L)
    idx = np.argsort(-D)
    E_h = E[:, idx[:k]]
    H = GraphFromIncidenceMatrix(E_h)
    Draw(H)
    print(E_h)
