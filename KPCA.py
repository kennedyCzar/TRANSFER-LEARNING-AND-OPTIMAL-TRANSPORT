#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:19:22 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np
from Utils.kernels import Kernels

class kPCA(Kernels):
    def __init__(self, k = None, kernel = None):
        super().__init__()
        if not k:
            k = 2
            self.k = k
        else:
            self.k = k
        if not kernel:
            kernel = 'rbf'
            self.kernel = kernel
        else:
            self.kernel = kernel
        return
    
    def explained_variance_(self):
        '''
        :Return: explained variance.
        '''
        self.total_eigenvalue = np.sum(self.eival)
        self.explained_variance = [x/self.total_eigenvalue*100 for x in sorted(self.eival, reverse = True)[:self.k]]
        return self.explained_variance
    
    def kernelize(self, x1, x2):
        '''
        :params: x1: NxD
        :params: x2: NxD
        '''
        if self.kernel == 'linear':
            return Kernels.linear(x1, x2)
        elif self.kernel == 'rbf':
            return Kernels.rbf(x1, x2)
        elif self.kernel == 'sigmoid':
            return Kernels.sigmoid(x1, x2)
        elif self.kernel == 'polynomial':
            return Kernels.polynomial(x1, x2)
        elif self.kernel == 'cosine':
            return Kernels.cosine(x1, x2)
        elif self.kernel == 'correlation':
            return Kernels.correlation(x1, x2)
        elif self.kernel == 'linrbf':
            return Kernels.linrbf(x1, x2)
        elif self.kernel == 'rbfpoly':
            return Kernels.rbfpoly(x1, x2)
        elif self.kernel == 'rbfcosine':
            return Kernels.rbfpoly(x1, x2)
        elif self.kernel == 'etakernel':
            return Kernels.etakernel(x1, x2)
        elif self.kernel == 'alignment':
            return Kernels.alignment(x1, x2)
        elif self.kernel == 'laplace':
            return Kernels.laplacian(x1, x2)
        elif self.kernel == 'locguass':
            return Kernels.locguass(x1, x2)
        elif self.kernel == 'chi':
            return Kernels.chi(x1)
        
    def fit(self, X):
        '''
        param: X: NxD
        '''
        self.X = X
        #normalized kernel
        N_N = 1/self.X.shape[0]*np.ones((self.X.shape[0],self.X.shape[0]))
        self.normKernel = self.kernelize(X, X) - N_N.dot(self.kernelize(X, X)) - self.kernelize(X, X).dot(N_N) + N_N.dot(self.kernelize(X, X).dot(N_N))
#        self.normKernel = self.kernelize(X, X) - 2*1/X.shape[0]*np.ones((X.shape[0], X.shape[0])).dot(self.kernelize(X, X)) + \
#                            1/X.shape[0]*np.ones((X.shape[0], X.shape[0])).dot(np.dot(1/X.shape[0]*np.ones((X.shape[0], X.shape[0])), self.kernelize(X, X)))
        self.eival, self.eivect = np.linalg.eig(self.normKernel)
        self.eival, self.eivect = self.eival.real, self.eivect.real
        #sort eigen values and return explained variance
        self.sorted_eigen = np.argsort(self.eival[:self.k])[::-1]
        self.explained_variance = self.explained_variance_()
        #return eigen value and corresponding eigenvectors
        self.eival, self.eivect = self.eival[:self.k], self.eivect[:, self.sorted_eigen]
        self.components_ = self.eivect.T
        return self
    
    def fit_transform(self):
        '''
        Return: transformed data
        '''
        return self.normKernel.dot(self.eivect)
    
    def inverse_transform(self):
        '''
        :Return the inverse of input data
        '''
        self.transformed = self.normKernel.dot(self.eivect)
        return self.transformed.dot(self.components_)
    
    

