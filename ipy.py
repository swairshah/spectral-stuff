from gram_schmidt import *
from matrix_stuff import *
from conjugate_gs import *
from laplacian import *

A = np.random.rand(10, 15)
B = np.random.rand(4, 5)

G = nx.complete_graph(4)
L = Laplacian(G)
E = IncidenceMatrix(G)
L_plus = np.linalg.pinv(L)

