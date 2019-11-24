# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:45:03 2019

@author: amitabh.gunjan
"""

import numpy as np
import sklearn.datasets, sklearn.decomposition

X = sklearn.datasets.load_iris().data
mu = np.mean(X, axis=0)

pca = sklearn.decomposition.PCA()
pca.fit(X)

nComp = 4
Xhat = np.dot(pca.transform(X)[:,:nComp], pca.components_[:nComp,:])
Xhat += mu

print(Xhat[0,])

pcomps = pca.components_
pca.n_components


cov = pca.get_covariance()
from numpy import linalg as LA
evals, eigenvectors = LA.eig(cov)
eigenvectors = eigenvectors.T
