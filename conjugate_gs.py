#!/usr/bin/python
import numpy as np
from numpy.linalg import norm
from gram_schmidt import *
from matrix_stuff import *

tol = 1E-8

def isPD(X):
    return np.all(np.linalg.eigvals(X.dot(X.T)) > 0)

def isPSD(X):
    return np.all(np.linalg.eigvals(X.dot(X.T)) - 0 >= tol)

def CReduceSingle(x, p, B):
    return x - x.T.dot(B).dot(p)

def CReduce(x, P, B):
    r, c = P.shape
    t = x
    for j in range(c):
        t = t - (t.T.dot(B).dot(P[:,j])) * P[:,j]
    return t

def CNormalize(x, B):
    #print(x.shape)
    #assert len(x.shape) == 1
    norm_sq = (x.T.dot(B).dot(x))
    if norm_sq < tol:
        return x
    else:
        return x/(norm_sq)**0.5

def ConjugateGS(X, B):
    r, c = X.shape
    P = np.zeros(X.shape)
    p0 = CNormalize(X[:,0], B)
    P[:,0] = p0;

    for j in range(1,c):
        p = CReduce(X[:,j], P, B)
        #P[:, j] = CNormalize(p, B)
        norm_sq = (p.T.dot(B).dot(p))
        if norm_sq > tol:
            P[:, j] = p/(norm_sq)**0.5

    return P

def SelfConjugatePinv(X, k = 0):
    # `k` denotes the number of
    # vectors to use to construct
    # pseudoinverse, 0 means use all
    P = ConjugateGS(X, X)
    r, c = P.shape
    X_pinv = np.zeros((r, r))
    if k == 0:
        k = r
    for i in range(k):
        p = P[:,i]
        norm_ = p.T.dot(X).dot(p)
        if norm_ > tol:
            X_pinv += np.outer(p,p)/norm_

    return X_pinv

def ConjugatePinv(X, B):
    P = ConjugateGS(X, B)
    r, c = P.shape
    #k = np.linalg.matrix_rank(B)
    B_pinv = np.zeros((r,r))
    for i in range(r):
        p = P[:, i]
        d = p.T.dot(B).dot(p)
        if d > tol:
            B_pinv += np.outer(p,p)/d
    return B_pinv

def test():
    y = np.array([
        [1,1,1,1],
        [1,2,1,2],
        [1,2,3,1]
        ])

    I = np.eye(3)

    #print(ConjugateGS(y, I))
    #print(GramSchmidt(y))
    #print(norm(ConjugateGS(y, I) - GramSchmidt(y)))

    A = np.random.uniform(size = (50,100))
    A = A.dot(A.T)
    B = np.random.uniform(size = (50,50))
    A_p1 = np.linalg.pinv(A)

    A_p2 = SelfConjugatePinv(A)
    #print(norm(A_p1 - A_p2))

    A_ = SelfConjugatePinv(A_p2)
    #print(norm(A - A_))

    P = ConjugateGS(A, A)
    r, c = P.shape
    for i in range(c):
        print(P[:,0].T.dot(A.dot(P[:,i])))

if __name__ == "__main__":
    test()
