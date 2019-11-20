#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 08:05:26 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np


class PCA:
    def __init__(self, k = None):
        '''
        :param: k: Number of principal components to select
                    Default value is 2.
        '''
        if not k:
            k = 2
            self.k = k
        else:
            self.k = k
        return
    
    def explained_variance_(self):
        '''
        :Return: explained variance.
        '''
        self.total_eigenvalue = np.sum(self.eival)
        self.explained_variance = [x/self.total_eigenvalue*100 for x in sorted(self.eival, reverse = True)[:self.k]]
        return self.explained_variance
    
    def fit(self, X):
        '''
        :param: X: NxD
        '''
        self.X = X
        self.Xcopy = X
        self.mean = np.mean(self.X, axis = 0)
        #centered mean
        self.X = self.X - np.mean(self.X, axis = 0)
        #covariance
        self.cov = (1/self.X.shape[1])* np.dot(self.X.T, self.X)
        self.eival, self.eivect = np.linalg.eig(self.cov)
        self.eival, self.eivect = self.eival.real, self.eivect.real
        self.sorted_eigen = np.argsort(self.eival[:self.k])[::-1]
        #sort eigen values and return explained variance
        self.explained_variance = self.explained_variance_()
        #return eigen value and corresponding eigenvectors
        self.eival, self.eivect = self.eival[:self.k], self.eivect[:, self.sorted_eigen]
        self.components_ = self.eivect.T
        return self
    
    def fit_transform(self):
        '''
        :Return: transformed datapoints
        '''
        return self.X.dot(self.eivect)
    
    def inverse_transform(self):
        '''
        :Return the inverse of input data
        '''
        self.transformed = self.X.dot(self.eivect)
        return self.transformed.dot(self.components_) + self.mean



