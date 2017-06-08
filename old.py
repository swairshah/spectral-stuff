import numpy as np
from numpy.linalg import norm
from gram_schmidt import *

def CReduceSingle(x, p, B):
    return x - x.T.dot(B).dot(p)

def CReduce(x, P, B):
    r, c = P.shape
    t = x
    for j in range(c):
        t = t - (t.T.dot(B).dot(P[:,j])) * P[:,j]
    return t

def CNormalize(x, B):
    norm = (x.T.dot(B).dot(x))**0.5
    if norm == 0:
        return np.zeros(x.size)
    else:
        return x/norm

def ConjugateGS(X, B):
    r, c = X.shape
    P = np.zeros(X.shape)
    p0 = CNormalize(X[:,0], B)
    P[:,0] = p0;

    for j in range(1,c):
        p = CReduce(X[:,j], P, B)
        P[:, j] = CNormalize(p, B)

    return P

def ConjuagtePinv(X, k = 0):
    P = ConjugateGS(X, X)
    r, c = P.shape
    X_pinv = np.zeros((r, r))
    if k == 0:
        k = r
    for i in range(k):
        p = P[:,i]
        norm = p.T.dot(X).dot(p)
        X_pinv += np.outer(p,p)/norm

    return X_pinv

def test():
    y = np.array([
        [1,1,1,1],
        [1,2,1,2],
        [1,2,3,1]
        ])

    I = np.eye(3)

    print(ConjugateGS(y, I))
    print(GramSchmidt(y))

if __name__ == "__main__":
    A = np.random.normal(size = (50,100))
    A = A.dot(A.T)
    A_p1 = np.linalg.pinv(A)
    #print(norm(A.dot(A_p1), ord = 'fro'))
    #print(norm(A_p1.dot(A), ord = 'fro'))

    A_p2 = ConjuagtePinv(A)
    print(norm(A_p1 - A_p2))
    #print(norm(A.dot(A_p2), ord = 'fro'))
    #print(norm(A_p2.dot(A), ord = 'fro'))

    A_ = ConjuagtePinv(A_p2)
    print(norm(A - A_))
