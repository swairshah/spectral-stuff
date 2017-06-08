import numpy as np
from numpy.linalg import norm
from numpy.linalg import qr

def GramSchmidt(X):
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

#XXX:Fix these, some issue.
#def ModifiedGS(X):
#    r, c = X.shape
#    Q = X.copy()
#    for i in range(c):
#        q_norm = norm(Q[:, i])
#        if q_norm != 0:
#            Q[:, i] = Q[:, i]/q_norm
#
#        for j in range(i+1, c):
#            proj = np.dot(Q[:, i], Q[:, j])
#            Q[:, j] = Q[:, j] - proj*Q[:, i]
#    return Q
#
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
    print(norm(GramSchmidt(x), ord = 'fro'))

    y = np.array([
        [1,1,1,1],
        [1,2,1,2],
        [1,2,3,1]
        ])

    print(GramSchmidt(y))
