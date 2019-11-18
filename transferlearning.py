#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:13:47 2019

@author: kenneth
"""


from __future__ import absolute_import
import os
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from KPCA import kPCA
from sklearn.neighbors import KNeighborsClassifier
from PCA import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

path = '/home/kenneth/Documents/MLDM M2/ADVANCE_ML/TRANSFER LEARNING/DATASET'
data = loadmat(os.path.join(path, 'webcam.mat'))
X_w = data['fts']
y_w = data['labels']

data_d = loadmat(os.path.join(path, 'dslr.mat'))
X_d = data['fts']
y_d = data['labels']

#%%

class subspacealignment(object):
    def __init__(self):
        '''
        Domain Adaptation
        :References: https://hal.archives-ouvertes.fr/hal-00869417/document
                    https://arxiv.org/abs/1409.5241
        '''
        return
    
    def fit(self, ds_x = None, ds_y = None, dt_x = None, \
            dt_y = None, d = None, type = None, m_kernel = None):
        '''Domain Adaptation using Subspace Alignment
        :param: ds_x: NxD
        :param: ds_y: Dx1
        :param: dt_x: NxD
        :param: dt_y: Dx1
        :param: d: Number of principal components
        '''
        if ds_x is None:
            raise IOError('Source Input data in required')
        else:
            self.ds_x = ds_x
        if ds_y is None:
            raise IOError('Source Input data in required')
        else:
            self.ds_y = ds_y
        if dt_x is None:
            raise IOError('Source Input data in required')
        else:
            self.dt_x = dt_x
        if dt_y is None:
            raise IOError('Source Input data in required')
        else:
            self.dt_y = dt_y
        if d is None:
            d = 2
            self.d = d
        else:
            self.d = d
        if not m_kernel:
            m_kernel = 'linear'
            self.m_kernel = m_kernel
        else:
            self.m_kernel = m_kernel
        #find PCA for Source domain after scaling
        X_w = MinMaxScaler().fit_transform(self.ds_x) #scale source data
        if not type:
            X_s = kPCA(k = self.d, kernel = self.m_kernel).fit(X_w.T) #perform PCA
        else:
            X_s = PCA(k = self.d).fit(X_w)
        X_s = X_s.components_.T #get components
        
        #PCA for target domain after scaling
        X_d = MinMaxScaler().fit_transform(self.dt_x) #scale target data
        if not type:
            X_t = kPCA(k = self.d, kernel = self.m_kernel).fit(X_d.T) #perform PCA
        else:
            X_t = PCA(k = self.d).fit(X_d)
        X_t = X_t.components_.T #get components
        
        #compute source and target projections
        X_a = X_s.dot(X_s.T.dot(X_t))
        S_a = self.ds_x.dot(X_a) #source projection
        T_a = self.dt_x.dot(X_t) #target projection
        
        #perform classification
        '''
        Fit a 1-NN classifier on S_a and make predictions on T_a
        '''
        self.classifier = KNeighborsClassifier(n_neighbors = 1)
        self.classifier.fit(S_a, self.ds_y)
        self.accuracy = accuracy_score(self.dt_y, self.classifier.predict(T_a))
        return self.accuracy
        


#%% Testing
        
subalignacc = subspacealignment().fit(X_w, y_w, X_d, y_d, d = 10, m_kernel = 'linear')



