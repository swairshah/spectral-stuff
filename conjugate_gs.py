import numpy as np
from numpy.linalg import norm
from gram_schmidt import *
from matrix_stuff import *
tol = 1E-8

def conjugate_gs(X, A):
    #U = X
    U = remove_zero_cols(modified_gs(X))
    r, c = U.shape
    P = U.copy()
    for i in range(1,c):
        p = P[:, i]
        for j in range(i):
            q = P[:, j]
            denom = q.T.dot(A.dot(q))
            if denom > tol:
                numer = q.T.dot(A.dot(p))
                p = p - (numer/denom)*q

    return P

def conjugate_pinv(X, A):
    P = conjugate_gs(X, A)
    A_pinv = np.zeros(A.shape)
    for i in range(P.shape[1]):
        p = P[:, i]
        d = p.T.dot(A).dot(p)
        if d > tol:
            A_pinv += np.outer(p,p)/d
    return A_pinv

