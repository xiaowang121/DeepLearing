# coding=utf-8
"""...................................................
                author:Xiaofei Wang
                date:2020.12.6
                Pytorch 1.6.1
                python 3.7
    ..................................................
"""
"""
This program is used to generate input features
"""

import os
import re
import math
import numpy as np
from numba import jit
from torch.utils.data import Dataset

from WMatrix import reshape2DTo3D, calculatePreMtxFromCovMtx

aa_dictionary = {
    'A': 1,
    'B': 0,
    'C': 2,
    'D': 3,
    'E': 4,

    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 0,

    'K': 9,
    'L': 10,
    'M': 11,
    'N': 12,
    'O': 0,

    'P': 13,
    'Q': 14,
    'R': 15,
    'S': 16,
    'T': 17,
    'U': 0,

    'V': 18,
    'W': 19,
    'X': 0,
    'Y': 20,
    'Z': 0,
    '-': 0,
    '*': 0,
}


def loadA3M(a3m_path):

    with open(a3m_path, 'r') as f:
        lines = f.readlines()
    lines = filter(lambda x: '>' not in x, lines)
    msa = [re.sub(r'[a-z''\n]', '', x) for x in lines]

    return msa


def transferMSA(msa):

    N = len(msa)
    L = len(msa[0])  # the length of query protein
    numeric_msa = np.zeros([N, L], dtype=int)
    for i in range(N):
        aline = msa[i]
        for j in range(L):
            numeric_msa[i, j] = aa_dictionary[aline[j]]

    return numeric_msa


@jit
def get_weights(numeric_msa):
    N = numeric_msa.shape[0]
    L = numeric_msa.shape[1]
    weights = np.zeros(N, dtype=np.float32)
    for i in range(N):
        mn = 0
        for j in range(N):
            count = 0
            for k in range(L):
                if numeric_msa[i, k] == numeric_msa[j, k]:
                    count += 1
            if count >= L * 0.8:
                mn += 1
        weights[i] = 1 / mn
    return weights


@jit
def generate21Lx21LCovMatrix(numeric_msa, weights):
    """
    Calculate the covariance matrix according to the sequence information
    """
    Lambda = 1  # 超参数lambda
    q = 21  # 20种氨基酸和空位
    N = numeric_msa.shape[0]
    L = numeric_msa.shape[1]

    sigma_weights = np.zeros((L, q))
    pa = np.zeros((L, q))  # 生成一个L*21的空矩阵
    pab = np.zeros((q, q))
    cov = np.zeros((L * q, L * q))
    for i in range(L):
        neff = 0
        for j in range(N):
            sigma_weights[i, numeric_msa[j, i]] += weights[j]
            neff += weights[j]
        for aa in range(q):
            pa[i, aa] = (sigma_weights[i, aa] + Lambda / q) / (neff + Lambda)
            # pa[i, aa] = (sigma_weights[i, aa] + Lambda*neff / q) / (neff + Lambda*neff)
    for i in range(L):
        for j in range(i, L):
            for a in range(q):
                for b in range(q):
                    if i == j:
                        if a == b:
                            pab[a, b] = (sigma_weights[i, a] + Lambda/ math.pow(q, 2)) / (neff + Lambda)
                            # pab[a, b] = (sigma_weights[i, a] + Lambda * neff / math.pow(q, 2)) / (neff + Lambda * neff)
                        else:
                            pab[a, b] = (Lambda/ math.pow(q, 2)) / (neff + Lambda)
                            # pab[a, b] = (Lambda * neff / math.pow(q, 2)) / (neff + Lambda * neff)
            if i != j:
                for k in range(N):
                    a = numeric_msa[k, i]
                    b = numeric_msa[k, j]
                    tmp = weights[k]
                    pab[a, b] += tmp
                for a in range(q):
                    for b in range(q):
                        pab[a, b] = (pab[a, b] + Lambda / math.pow(q, 2)) / (neff + Lambda)
                        # pab[a, b] = (pab[a, b] + Lambda*neff / math.pow(q, 2)) / (neff + Lambda*neff)
            for a in range(q):
                for b in range(q):
                    if i != j or a == b:
                        if pab[a, b] > 0:
                            cov[i * 21 + a][j * 21 + b] = pab[a][b] - pa[i][a] * pa[j][b]
                            cov[j * 21 + b][i * 21 + a] = cov[i * 21 + a][j * 21 + b]
    return cov


@jit
def cal_large_matrix1(msa, weight):
    # output:21*l*21*l
    ALPHA = 21
    pseudoc = 1
    M = msa.shape[0]
    N = msa.shape[1]
    pab = np.zeros((ALPHA, ALPHA))
    pa = np.zeros((N, ALPHA))
    cov = np.zeros([N * ALPHA, N * ALPHA])
    for i in range(N):
        for aa in range(ALPHA):
            pa[i, aa] = pseudoc
        neff = 0.0
        for k in range(M):
            pa[i, msa[k, i]] += weight[k]
            neff += weight[k]
        for aa in range(ALPHA):
            pa[i, aa] /= pseudoc * ALPHA * 1.0 + neff

    # print(pab)
    for i in range(N):
        for j in range(i, N):
            for a in range(ALPHA):
                for b in range(ALPHA):
                    if i == j:
                        if a == b:
                            pab[a, b] = pa[i, a]
                        else:
                            pab[a, b] = 0.0
                    else:
                        pab[a, b] = pseudoc * 1.0 / ALPHA
            if (i != j):
                neff2 = 0;
                for k in range(M):
                    a = msa[k, i]
                    b = msa[k, j]
                    tmp = weight[k]
                    pab[a, b] += tmp
                    neff2 += tmp
                for a in range(ALPHA):
                    for b in range(ALPHA):
                        pab[a, b] /= pseudoc * ALPHA * 1.0 + neff2
            for a in range(ALPHA):
                for b in range(ALPHA):
                    if (i != j or a == b):
                        if (pab[a][b] > 0.0):
                            cov[i * 21 + a][j * 21 + b] = pab[a][b] - pa[i][a] * pa[j][b]
                            cov[j * 21 + b][i * 21 + a] = cov[i * 21 + a][j * 21 + b]

    return cov

"""
To reduce the covariance matrix
"""
@jit
def suojiancov(cov):
    lambda1 = 0.5
    c = 0.0
    sjcov = np.zeros((cov.shape[0], cov.shape[1]))
    for i in range(cov.shape[0]):
        c += (cov[i, i])
    for i in range(cov.shape[0]):
        sjcov[i, i] = c / cov.shape[0]
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            sjcov[i, j] = lambda1 * sjcov[i, j] + (1-lambda1) * cov[i, j]
    return sjcov

def feaByMSA(msa_path):
    msa = loadA3M(msa_path)
    numeric_msa = transferMSA(msa)
    weights = get_weights(numeric_msa)
    cov_mtx = generate21Lx21LCovMatrix(numeric_msa, weights)
    # cov_mtx = cal_large_matrix1(numeric_msa, weights)
    # sjcov=suojiancov(cov_mtx)
    # pre_mtx = calculatePreMtxFromCovMtx(sjcov)
    pre_mtx = calculatePreMtxFromCovMtx(cov_mtx)
    fea = reshape2DTo3D(pre_mtx)
    return fea

class WContactDataset(Dataset):


    def __init__(self, msa_root_dir, transform=None):
        super(WContactDataset, self).__init__()

        self.msa_root_dir = msa_root_dir
        self.transform = transform
        self.filename_list1 = []
        self.msa_list = os.listdir(self.msa_root_dir)  # Obtaining all files in this directory
        for filename in self.msa_list:
            a = filename.split('.')
            self.filename_list1.append(a[0])

        self.filename_list = self.filename_list1
    def __len__(self):
        return len(self.filename_list)

    def getfilename(self,ind):
        return self.filename_list[ind]

    def getIthProteinLen(self, index):
        file_name = self.filename_list[index]  # obtaining the name of the msa file according to the index value
        print(file_name)
        a3m_path = os.path.join(self.msa_root_dir, file_name+'.a3m')  # obtaining the absolution path of the msa file
        msa = loadA3M(a3m_path)
        length = len(msa[0])
        return length

    def __getitem__(self, index):
        file_name = self.filename_list[index]  # obtaining the name of the msa file according to the index value
        msa_path = os.path.join(self.msa_root_dir, file_name+'.a3m')  # obtaining the absolution path of the msa file
        fea = feaByMSA(msa_path)
        sample = {'fea': fea}  # construct the dictionary
        return sample

    def getIthSampleFeaLab(self, index):
        sample = self.__getitem__(index)
        fea = np.expand_dims(sample['fea'], 0)
        return fea


