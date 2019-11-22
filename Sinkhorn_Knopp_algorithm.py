#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:34:48 2019

@author: kenneth
"""

from __future__ import absolute_import
import time
import numpy as np
from scipy.spatial.distance import cdist
from utils import EvalC
from sklearn.neighbors import KNeighborsClassifier

class sinkhorn(EvalC):
    def __init__(self):
        '''Sinkhorn-Knopp Algorithm for Optimal Transport
            :Reference: https://arxiv.org/pdf/1306.0895.pdf
        :Returntype: d-matrix
        '''
        super().__init__()
        return
        
    def fit(self, ds_x = None, ds_y = None, dt_x = None, dt_y = None, \
            a = None, b = None, reg = None, iteration=None, thresh = None):
        '''
        :param: a: uniform distribution of 1
        :param b: uniform distribution of 1
        :param M: Cost Matrix
        :param reg: regularizaton term. Default is 1
        '''
#        start = time.time()
        if ds_x is None:
            raise IOError('Source Input data in required')
        else:
            self.ds_x = ds_x
        if ds_y is None:
            raise IOError('Source Input data in required')
        else:
            self.ds_y = ds_y.ravel()
        if dt_x is None:
            raise IOError('Source Input data in required')
        else:
            self.dt_x = dt_x
        if dt_y is None:
            raise IOError('Source Input data in required')
        else:
            self.dt_y = dt_y.ravel()
        self.M = cdist(self.ds_x, self.dt_x)
        N_mn, N_md = self.M.shape
        # init data
        if not reg:
            reg = 5
            self.reg = reg
        else:
            self.reg = reg
        if not iteration:
            iteration = 1000
            self.iteration = iteration
        else:
            self.iteration = iteration
        if not thresh:
            thresh = 1e-9
            self.thresh = thresh
        else:
            self.thresh = thresh
        if not a:
            a = np.ones(N_mn)
            self.a = a
        else:
            self.a = a
        if not b:
            b = np.ones(N_md)
            self.b = b
        else:
            self.b = b
        if not reg:
            reg = 1
            self.reg = 1
        else:
            self.reg = 1
        #define u and v vectors
        
        self.u = np.ones(N_mn)
        self.v = np.ones(N_md)
        #compute Kernel K
        self.K = np.exp(-self.M/reg)
        tmp2 = np.empty(b.shape, dtype=self.M.dtype)
        Kp = (1 / a).reshape(-1, 1) * self.K
        epsilon = 1
        for iterr in range(self.iteration):
            if (epsilon > self.thresh):
                uprev = self.u
                vprev = self.v
    #            KtransposeU = np.dot(K.T, u)
                self.v = np.divide(b, self.K.T.dot(self.u))
                self.u = 1. / np.dot(Kp, self.v)
                if (np.any(self.K.T.dot(self.u) == 0) or np.any(np.isnan(self.u)) or np.any(np.isnan(self.v))
                    or np.any(np.isinf(self.u)) or np.any(np.isinf(self.v))):
                    self.u = uprev
                    self.v = vprev
                    break
            if (iterr % 10) == 0:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = self.u.dot(self.K.dot(self.v))
                epsilon = np.linalg.norm(tmp2 - b)
        self.G = self.u.reshape((-1, 1)) * self.K * self.v.reshape((1, -1))
#        print(f'Finnihsed running sinkhorn algorithm\nTime: {round(time.time() - start, 3)}secs')
        print('*'*40)
        self.S_a = self.G.dot(self.dt_x)
        print('>>>> Transported Source to target domain using coupling matrix')
        print('*'*40)
        print('Initializing 1-Nearest Neighbour classifier')
        self.classifier = KNeighborsClassifier(n_neighbors = 1)
        self.classifier.fit(self.S_a, self.ds_y)
        print('>>>> Done fitting source domain >>>>')
        self.ypred = self.classifier.predict(self.dt_x)
        print(f'Accuracy: {EvalC.accuary_multiclass(self.dt_y, self.ypred)}')
        return self
    
    

#%%


#ot1 = sinkhorn().fit(X_w, y_w, X_d, y_d)
