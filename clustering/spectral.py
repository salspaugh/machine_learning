#!/usr/bin/env python

import clustering
import kmedoids
import numpy as np
import scipy
import scipy.linalg as la
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import svds

from scipy.cluster.vq import whiten, kmeans

import matplotlib.pyplot as plt

def cluster(distances, k=4, sigma=2.):
    L = compute_special_L(distances, sigma)
    X = compute_special_eigenmatrix(L, k)
    Y = renormalize(X)
    return cluster_rows_in_k(Y, k)

def compute_special_L(distances, sigma):
    affinity = compute_affinity(distances, sigma)
    s = affinity.sum(axis=1) + 1e-10
    
    # While seemingly sensible, the exponentiation of a large negative number
    # and underflow results in a singular matrix. So instead we rely on the 
    # fact that D is diagonal to invert it. We do not do:
    #       Dinvsqrt = scipy.real(la.sqrtm(la.inv(D)))
    # Also note that D = np.diagflat(s) but we don't need this, we compute 
    # D^(-1/2) directly below.
    
    Dinvsqrt = np.diagflat(np.sqrt(1./s))
    
    # If you don't cast those to matrices first, and the matrices are stored
    # as ndarrays, it will multiply pairwise.
    return np.mat(Dinvsqrt) * np.mat(affinity) * np.mat(Dinvsqrt)

def compute_affinity(distances, sigma):
    affinity = np.exp(-1.*(np.multiply(distances, distances)) / (2.*(sigma**2.)))
    np.fill_diagonal(affinity, 0.)
    return affinity 

def compute_special_eigenmatrix(L, k):
    return eigsh(scipy.real(L), k=k)[1] # Freakin' magic ...

def renormalize(X):
    Xnorm = (np.sqrt(X**2)).sum(axis=1)
    for row in range(X.shape[0]): # TODO(salspaugh): Change this implementation -- looping is inefficient.
        X[row,:] = X[row,:] / Xnorm[row]
    return X

def cluster_rows_in_k(kdata, k):
    distances = compute_distances_in_k(kdata)
    return kmedoids.cluster(distances, k=k)

def l2norm(a, b):
    return np.linalg.norm(a - b)

def compute_distances_in_k(data):
    return clustering.build_distance_matrix(data, l2norm)
