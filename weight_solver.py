import numpy as np
"""
Given X (mxn) and Y(mxr), solve for w1..wr such that,
\sum x_i x_i^T = \sum w_j (y_j y_j^T)
"""

def vectorize(A):
    ret = A.reshape((-1,1))
    ret = ret.flatten()
    return ret

def solve_for_w(X, Y):
    m, n = X.shape
    assert(Y.shape[0] == m)
    r = Y.shape[1]

    Ax = np.zeros((m*m, n))
    Ay = np.zeros((m*m, r))

    for i in range(n):
        xxt = np.outer(X[:,i],X[:,i])
        Ax[:,i] = vectorize(xxt)

    for i in range(r):
        yyt = np.outer(Y[:,i],Y[:,i])
        Ay[:,i] = vectorize(yyt)

    Aw = np.linalg.pinv(Ay) @ Ax

    w = Aw.sum(axis = 1)
    return w

def solve_for_w_laplacian(L, Y):
    m, m = L.shape
    assert(Y.shape[0] == m)
    r = Y.shape[1]

    Ax = np.zeros((m*m, n))
    Ay = np.zeros((m*m, r))

    for i in range(n):
        xxt = np.outer(X[:,i],X[:,i])
        Ax[:,i] = vectorize(xxt)

    for i in range(r):
        yyt = np.outer(Y[:,i],Y[:,i])
        Ay[:,i] = vectorize(yyt)

    Aw = np.linalg.pinv(Ay) @ Ax

    w = Aw.sum(axis = 1)
    return w

def iterative_column_removal(E, F, k, tol=1e-10):
    m, n = E.shape
    r = F.shape[1]
    w = np.ones(r)
    while r > k:
        w = solve_for_w(E, F)
        idx = np.argmin(w)
        F = np.delete(F, idx, axis = 1)
        r = F.shape[1]

    return F, w

if __name__ == "__main__":
    m = 5
    n = 20
    r = 5
    X = np.random.randn(m,n)
    Y = np.random.randn(m,r)
    #print(np.linalg.matrix_rank(X))
    #print(np.linalg.matrix_rank(Y))
    #Y = X.copy()
    iterative_column_removal(X, Y, k = 3)
