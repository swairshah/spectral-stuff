import numpy as np
from numpy.linalg import norm, qr

tol = 1e-8

def gram_schmidt(X):
    r, c = X.shape
    Q = np.zeros((r,c))
    for j in range(c):
        q = X[:,j]
        for i in range(j):
            proj = np.dot(Q[:, i], X[:, j])
            q = q - proj*Q[:,i]

        q_norm = norm(q)
        if q_norm != 0:
           Q[:,j] = q / q_norm
        else:
           Q[:,j] = q
    return Q

def modified_gs(X):
    r, c = X.shape
    Q = X.copy()
    for i in range(c):
        q_norm = norm(Q[:, i])
        if q_norm > tol:
            Q[:, i] = Q[:, i]/q_norm
        else:
            Q[:, i] = np.zeros(r)

        for j in range(i+1, c):
            proj = np.dot(Q[:, i], Q[:, j])
            Q[:, j] = Q[:, j] - proj*Q[:, i]
    return Q

def remove_zero_cols(X):
    return X[:,[norm(i) > tol for i in X.T]]

#XXX:Fix these, some issue.
#def PivotedGS(X):
#    r, c = X.shape
#    Q = X.copy()
#    col_norms = [(i, norm(X[:,i])) for i in range(c)]
#    col_norms = sorted(col_norms, key = lambda tup : tup[1])
#    indices = [i[0] for i in col_norms]
#    for i in range(c):
#        val_i = indices[i]
#        q_norm = norm(Q[:, val_i])
#        if q_norm != 0:
#            Q[:, val_i] = Q[:, val_i]/q_norm
#
#        for j in range(i+1, c):
#            val_j = indices[j]
#            proj = np.dot(Q[:, val_i], Q[:, val_j])
#            Q[:, val_j] = Q[:, val_j] - proj*Q[:, val_i]
#    return Q

if __name__ == "__main__":
    x = np.random.normal(size = (3000,300))

    print(norm(qr(x)[0], ord = 'fro'))
    print(norm(gram_schmidt(x), ord = 'fro'))

    y = np.array([
        [1,1,1,1],
        [1,2,1,2],
        [1,2,3,1]
        ])

    print(gram_schmidt(y))
