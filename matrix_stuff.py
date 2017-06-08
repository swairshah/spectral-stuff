import numpy as np
from numpy.linalg import norm

tol = 1E-8

def sort_cols_by_norm(X):
    r, c = X.shape
    norm_list = []
    for i in range(c):
        n = norm(X[:, i])
        norm_list.append(-n)
    sorted_idx = np.argsort(np.array(norm_list))
    return sorted_idx 

def sort_cols_conjugate(X, B):
    r, c = X.shape
    norm_list = []
    for i in range(c):
        x = X[:, i]
        n = norm(x.T.dot(B.dot(x)))
        norm_list.append(-n)
    sorted_idx = np.argsort(np.array(norm_list))
    return sorted_idx 

def proj(x, A):
    inv = np.linalg.pinv(A.T.dot(A))
    return (A.dot(inv.dot(A.T))).dot(x)

def proj_on_null(x, A):
    return (x - proj(x, A))

def proj_ortho_to_null(x, A):
    U, s, _ = np.linalg.svd(A)
    indices = s > tol
    Uk = U[:, indices]
    return proj(x, Uk)
