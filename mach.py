from scipy import sparse as sa
from scipy import linalg as la
import numpy as np
import random

def mach_td(X, rank, p):
    X_p = sa.coo_matrix(shape=X.shape, dtype=X.dtype) 

    for idx, v in enumerate(X):
        coinToss = random.uniform(0,1)        
        if coinToss <= p:
            X_p[idx] = v/p

    factors = []
    for i in range(X.ndim):

