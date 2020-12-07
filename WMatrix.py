# coding=utf-8
"""
Calculate the input feature: precision matrix,and calculation method of ResPRE(L. Yang, H. Jun, Z. Chengxin,
Y. Dong-Jun, Z. Yang, ResPRE: high-accuracy protein contact prediction by coupling precision matrix with deep residual neural networks,
Bioinformatics, 35 (2019) 4647-4655.) is adopted.
"""
import numpy as np
import numpy.linalg as na
from numba import jit

@jit
def ROPE(input_matrix, rho):

    p = input_matrix.shape[0]
    S = input_matrix

    LM = na.eigh(S)
    L = LM[0]
    M = LM[1]

    for i in range(len(L)):
        if L[i] < 0:
            L[i] = 0

    lamda = 2.0 / (L + np.sqrt(np.power(L, 2) + 8 * rho))

    indexlamda = np.argsort(-lamda)
    lamda = np.diag(-np.sort(-lamda)[:p])
    hattheta = np.dot(M[:, indexlamda], lamda)
    hattheta = np.dot(hattheta, M[:, indexlamda].transpose())
    return hattheta


@jit
def reshape2DTo3D(mtx, dim=21):
    """
    :param mtx: size: (L*d) * (L*d)
    :param dim: d
    :return: size: (d*d)*(L)*(L)
    """
    p = mtx.shape[0] // dim
    reshape = np.zeros([dim*dim, p, p])
    for i in range(p):
        for j in range(p):
            reshape[:, i, j] = mtx[i*dim:i*dim+dim, j*dim:j*dim+dim].flatten()
    return reshape


@jit
def calculatePreMtxFromCovMtx(cov_mtx):
    rho2 = np.exp((np.arange(80)-60)/5.0)[30]
    pre_mtx = ROPE(cov_mtx, rho2)
    return pre_mtx