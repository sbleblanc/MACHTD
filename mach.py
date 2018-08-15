from scipy import sparse as sa
from scipy.sparse import linalg as sla
from scipy import linalg as la
import numpy as np
import random
from tensorly import tenalg as ta
import tensorly


def mach_td(X, rank, p):
    X_s = np.zeros(X.shape).flatten()

    for idx, v in enumerate(X.flatten()):
        coinToss = random.uniform(0,1)        
        if coinToss <= p:
            X_s[idx] = v/p

    X_s = X_s.reshape(X.shape)
    factors = []
    for i in range(X.ndim):
        if rank[i] < X.shape[i]:
           A_s = sa.csr_matrix(tensorly.unfold(X_s,i))
           #print(A_s.nnz)
           factors.append(sla.svds(A_s, k=rank[i], return_singular_vectors='u')[0])
        else:
            U, _, _ = la.svd(tensorly.unfold(X_s,i), full_matrices=False)
            factors.append(U)

    core = ta.multi_mode_dot(X_s,factors, transpose=True)
    return core, factors


