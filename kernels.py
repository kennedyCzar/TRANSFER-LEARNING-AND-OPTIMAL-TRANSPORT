#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:11:48 2019

@author: kenneth
"""

from __future__ import absolute_import
import numpy as np


class Kernels:
    '''
    Kernels are mostly used for solving
    non-lineaar problems. By projecting/transforming
    our data into a subspace, making it easy to
    almost accurately classify our data as if it were
    still in linear space.
    '''
    def __init__(self):
        return
    
    @staticmethod
    def linear(x1, x2, c = None):
        '''
        Linear kernel
        ----------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :return type: kernel(Gram) matrix
        '''
        if not c:
            c = 0
        else:
            c = c
        return x1.dot(x2.T) + c
    
    
    @staticmethod
    def linear_svdd(x1, x2, c = None):
        '''
        Linear kernel
        ----------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :return type: kernel(Gram) matrix
        '''
        if not c:
            c = 0
        else:
            c = c
        return x1.T.dot(x2) + c
    
    @staticmethod
    def rbf(x1, x2, gamma = None):
        '''
        RBF: Radial basis function or guassian kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if x1.ndim == 1 and x2.ndim == 1:
            return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
            return np.exp(-gamma * np.linalg.norm(x1 - x2, axis = 1)**2)
        elif x1.ndim > 1 and x2.ndim > 1:
            return np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 2)**2)
       
    @staticmethod
    def laplacian(x1, x2, gamma = None):
        '''
        RBF: Radial basis function or guassian kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if x1.ndim == 1 and x2.ndim == 1:
            return np.exp(-gamma * np.linalg.norm(x1 - x2))
        elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
            return np.exp(-gamma * np.linalg.norm(x1 - x2, axis = 1))
        elif x1.ndim > 1 and x2.ndim > 1:
            return np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 2))
        
    
    @staticmethod
    def locguass(x1, x2, d = None, gamma = None):
        '''
        :local guassian
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if not d:
            d = 5
        else:
            d = d
        if x1.ndim == 1 and x2.ndim == 1:
            return (np.maximum(0, 1 - gamma*np.linalg.norm(x1 - x2)/3)**d)*np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
            return (np.maximum(0, 1 - gamma*np.linalg.norm(x1 - x2, axis = 1)/3)**d)*np.exp(-gamma * np.linalg.norm(x1 - x2, axis = 1)**2)
        elif x1.ndim > 1 and x2.ndim > 1:
            return (np.maximum(0, 1 - gamma*np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 2)/3)**d) * np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 2)**2)
    
    @staticmethod
    def chi(x):
        '''
        Using Chisquared from sklearn
        '''
        from sklearn.metrics.pairwise import chi2_kernel
        return chi2_kernel(x)
    
    @staticmethod
    def sigmoid(x1, x2, gamma = None, c = None):
        '''
        logistic or sigmoid kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if not c:
            c = 1
        return np.tanh(gamma * x1.dot(x2.T) + c)
    
    
    @staticmethod
    def polynomial(x1, x2, d = None, c = None):
        '''
        polynomial kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: d: polynomial degree
        :return type: kernel(Gram) matrix
        '''
        if not d:
            d = 5
        else:
            d = d
        return (x1.dot(x2.T))**d
    
    @staticmethod
    def cosine(x1, x2):
        '''
        Cosine kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :return type: kernel(Gram) matrix
        '''
        
        return (x1.dot(x2.T)/np.linalg.norm(x1, 1) * np.linalg.norm(x2, 1))
    
    @staticmethod
    def correlation(x1, x2, gamma = None):
        '''
        Correlation kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        return np.exp((x1.dot(x2.T)/np.linalg.norm(x1, 1) * np.linalg.norm(x2, 1)) - 1/gamma)
    
    @staticmethod
    def linrbf(x1, x2, gamma = None, op = None):
        '''
        MKL: Lineaar + RBF kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if not op:
            op = 'multiply' #add seems like the best performning here
        else:
            op = op
        if op == 'multiply':
            return Kernels.linear(x1, x2) * Kernels.rbf(x1, x2, gamma)
        elif op == 'add':
            return Kernels.linear(x1, x2) + Kernels.rbf(x1, x2, gamma)
        elif op == 'divide':
            return Kernels.linear(x1, x2) / Kernels.rbf(x1, x2, gamma)
        elif op == 'subtract':
            return np.abs(Kernels.linear(x1, x2) - Kernels.rbf(x1, x2, gamma))
        elif op == 'dot':
            return Kernels.linear(x1, x2).dot(10000*Kernels.rbf(x1, x2, gamma).T)
        
    @staticmethod
    def rbfpoly(x1, x2, d = None, gamma = None, op = None):
        '''
        MKL: RBF + Polynomial kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
            gamma = gamma
        if not d:
            d = 5
        else:
            d = d
        if not op:
            op = 'multiply'
        else:
            op = op
        if op == 'multiply':
            return Kernels.polynomial(x1, x2, d) * Kernels.rbf(x1, x2, gamma)
        elif op == 'add':
            return Kernels.polynomial(x1, x2, d) + Kernels.rbf(x1, x2, gamma)
        elif op == 'divide':
            return Kernels.polynomial(x1, x2, d) / Kernels.rbf(x1, x2, gamma)
        elif op == 'subtract':
            return np.abs(Kernels.polynomial(x1, x2, d) - Kernels.rbf(x1, x2, gamma))
        elif op == 'dot':
            return Kernels.polynomial(x1, x2, d).dot(10000*Kernels.polynomial(x1, x2, d).T)
        
    @staticmethod
    def rbfcosine(x1, x2, gamma = None, op = None):
        '''
        MKL: RBF + Polynomial kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if not op:
            op = 'multiply'
        else:
            op = op
        if op == 'multiply':
            return Kernels.cosine(x1, x2) * Kernels.rbf(x1, x2, gamma)
        elif op == 'add':
            return Kernels.cosine(x1, x2) + Kernels.rbf(x1, x2, gamma)
        elif op == 'divide':
            return Kernels.cosine(x1, x2) / Kernels.rbf(x1, x2, gamma)
        elif op == 'subtract':
            return np.abs(Kernels.cosine(x1, x2) - Kernels.rbf(x1, x2, gamma))
        elif op == 'dot':
            return Kernels.cosine(x1, x2).dot(10000*Kernels.cosine(x1, x2).T)
        
    @staticmethod
    def etakernel(x1, x2, d = None, gamma = None, op = None):
        '''
        MKL: Pavlidis et al. (2001)
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if not op:
            op = 'etamul'
        else:
            op = op
        if not d:
            d = 5
        else:
            d = d
        if op == 'eta':
            return Kernels.linrbf(x1, x2).dot(Kernels.rbfpoly(x1, x2))
        elif op == 'etasum':
            return Kernels.linrbf(x1, x2) + Kernels.rbfpoly(x1, x2)
        elif op == 'etamul':
            return Kernels.linrbf(x1, x2) * Kernels.rbfpoly(x1, x2)
        elif op == 'etadiv':
            return Kernels.linrbf(x1, x2) / Kernels.rbfpoly(x1, x2)
        elif op == 'etapoly':
            return Kernels.linrbf(x1, x2).dot(Kernels.rbfpoly(x1, x2)) + Kernels.rbfpoly(x1, x2).dot(Kernels.rbfcosine(x1, x2))
        elif op == 'etasig':
            return Kernels.sigmoid(x1, x2).dot(Kernels.rbf(x1, x2)) + Kernels.rbfpoly(x1, x2).dot(Kernels.sigmoid(x1, x2))
        elif op == 'etaalpha':
            return Kernels.rbf(Kernels.linear(x1, x2).dot(Kernels.rbfpoly(x1, x2)), Kernels.sigmoid(x1, x2) + Kernels.polynomial(x1, x2))
            
    @staticmethod
    def alignment(x1, x2, d = None, gamma = None, op = None):
        '''
        MKL: Cortes et al.
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if not op:
            op = 'rbfpoly'
        else:
            op = op
        if not d:
            d = 5
        else:
            d = d
        
        kappa_lin = Kernels.linear(x1, x2) - 1/len(x1)*np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T).dot(Kernels.linear(x1, x2))\
                     - 1/len(x1)*Kernels.linear(x1, x2).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T)) + \
                     1/len(x1)**2*(np.ones((x1.shape[0], x1.shape[0])).T.dot(Kernels.linear(x1, x2))).dot(np.ones((x1.shape[0], x1.shape[0]))).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T))
        kappa_rbf = Kernels.rbf(x1, x2) - 1/len(x1)*np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T).dot(Kernels.rbf(x1, x2))\
                     - 1/len(x1)*Kernels.rbf(x1, x2).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T)) + \
                     1/len(x1)**2*(np.ones((x1.shape[0], x1.shape[0])).T.dot(Kernels.rbf(x1, x2))).dot(np.ones((x1.shape[0], x1.shape[0]))).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T))    
        kappa_poly = Kernels.polynomial(x1, x2) - 1/len(x1)*np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T).dot(Kernels.polynomial(x1, x2))\
                     - 1/len(x1)*Kernels.polynomial(x1, x2).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T)) + \
                     1/len(x1)**2*(np.ones((x1.shape[0], x1.shape[0])).T.dot(Kernels.polynomial(x1, x2))).dot(np.ones((x1.shape[0], x1.shape[0]))).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T))
        kappa_sigmoid = Kernels.sigmoid(x1, x2) - 1/len(x1)*np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T).dot(Kernels.sigmoid(x1, x2))\
                     - 1/len(x1)*Kernels.sigmoid(x1, x2).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T)) + \
                     1/len(x1)**2*(np.ones((x1.shape[0], x1.shape[0])).T.dot(Kernels.sigmoid(x1, x2))).dot(np.ones((x1.shape[0], x1.shape[0]))).dot(np.ones((x1.shape[0], x1.shape[0])).dot(np.ones((x1.shape[0], x1.shape[0])).T))
        if op == 'linrbf':
            return kappa_lin.dot(kappa_rbf)/np.sqrt(np.sum(kappa_lin.dot(kappa_lin))*np.sum(kappa_rbf.dot(kappa_rbf)))
        elif op == 'rbfpoly':
            return kappa_rbf.dot(kappa_poly)/np.sqrt(np.sum(kappa_rbf.dot(kappa_rbf))*np.sum(kappa_poly.dot(kappa_poly)))
        elif op == 'rbfsig':
            return kappa_rbf.dot(kappa_sigmoid)/np.sqrt(np.sum(kappa_rbf.dot(kappa_rbf))*np.sum(kappa_sigmoid.dot(kappa_sigmoid)))
        elif op == 'polysig':
            return kappa_poly.dot(kappa_sigmoid)/np.sqrt(np.sum(kappa_poly.dot(kappa_poly))*np.sum(kappa_sigmoid.dot(kappa_sigmoid)))
        return 
        