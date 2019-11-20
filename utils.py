#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:14:33 2019

@author: kenneth
"""
import numpy as np

class EvalC:
    def __init__(self):
        return
    
    #classification metrics
    '''
                            Actual
                        +ve       -ve
                    ---------------------
                +ve |   TP    |     FP  | +---> Precision
                    --------------------- |
    predicted   -ve |   FN    |   TN    | v
                    --------------------- Recall
    '''
    @staticmethod
    def TP(A, P):
        '''Docstring
        when actual is 1 and prediction is 1
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return np.sum((A == 1) & (P == 1))
    
    @staticmethod
    def FP(A, P):
        '''Docstring
        when actual is 0 and prediction is 1
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return np.sum((A == 0) & (P == 1))
    
    @staticmethod
    def FN(A, P):
        '''Docstring
        when actual is 1 and prediction is 0
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return np.sum((A == 1) & (P == 0))
    
    @staticmethod
    def TN(A, P):
        '''Docstring
        when actual is 0 and prediction is 0
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return np.sum((A == 0) & (P == 0))
    
    @staticmethod
    def confusionMatrix(A, P):
        '''Docstring
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        TP, FP, FN, TN = (EvalC.TP(A, P),\
                          EvalC.FP(A, P),\
                          EvalC.FN(A, P),\
                          EvalC.TN(A, P))
        return np.array([[TP, FP], [FN, TN]])
    
    @staticmethod
    def accuracy(A, P):
        '''Docstring
        :params: A: Actual label
        :params: P: predicted labels
        
        Also: Accuracy np.mean(Y == model.predict(X))
        '''
        return (EvalC.TP(A, P) + EvalC.TN(A, P))/(EvalC.TP(A, P) + EvalC.FP(A, P) +\
                                                EvalC.FN(A, P) + EvalC.TN(A, P))
    
    @staticmethod
    def precision(A, P):
        '''Docstring
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return EvalC.TP(A, P)/(EvalC.TP(A, P) + EvalC.FP(A, P))
    
    @staticmethod
    def recall(A, P):
        '''Docstring
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return EvalC.TP(A, P)/(EvalC.TP(A, P) + EvalC.FN(A, P))
    
    @staticmethod
    def fscore(A, P, beta):
        '''Docstring
        :params: A: Actual label
        :params: P: predicted labels
        :params: beta: positive parameter for rebalancing evaluation task.
        
        Reference: http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=3CB7C5A08700CAF45274C75AABFB75B2?doi=10.1.1.95.9153&rep=rep1&type=pdf
        '''
        return ((np.square(beta) + 1)*EvalC.precision(A, P)*EvalC.recall(A, P))/\
                (np.square(beta) * EvalC.precision(A, P) + EvalC.recall(A, P))
                
    @staticmethod         
    def TPR(A, P):
        '''Docstring
        True Positive rate:
            True Positive Rate corresponds to the 
            proportion of positive data points that 
            are correctly considered as positive, 
            with respect to all positive data points.
        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return EvalC.recall(A, P)
    
    @staticmethod
    def FPR(A, P):
        '''Docstring
        False Positive rate:
            False Positive Rate corresponds to the 
            proportion of negative data points that 
            are mistakenly considered as positive, 
            with respect to all negative data points.

        
        :params: A: Actual label
        :params: P: predicted labels
        '''
        if EvalC.FP(A, P) == 0 & EvalC.FP(A, P) == 0:
            return 0
        else:
            return EvalC.FP(A, P)/(EvalC.FP(A, P) + EvalC.TN(A, P))
    
    @staticmethod
    def TNR(A, P):
        '''Docstring
        True Negative Rate
        '''
        return EvalC.TN(A, P)/(EvalC.TN(A, P) + EvalC.FP(A, P))
    
    @staticmethod
    def f1(A, P):
        '''Docstring
        :params: A: Actual label
        :params: P: predicted labels
        '''
        return (2 * (EvalC.precision(A, P) * EvalC.recall(A, P)))/(EvalC.precision(A, P) + EvalC.recall(A, P))
    
    @staticmethod
    def summary(A, P):
        '''
        :params: A: Actual label
        :params: P: predicted labels
        :return: classification summary
        '''
        print('*'*40)
        print('\t\tSummary')
        print('*'*40)
        print('>> Accuracy: %s'%EvalC.accuracy(A, P))
        print('>> Precision: %s'%EvalC.precision(A, P))
        print('>> Recall: %s'%EvalC.recall(A, P))
        print('>> F1-score: %s'%EvalC.f1(A, P))
        print('>> True positive rate: %s'%EvalC.TPR(A, P))
        print('>> False positive rate: %s'%EvalC.FPR(A, P))
        print('*'*40)
        
class EvalR:
    def __init__(self):
        return
    
    #-Mean Square Error
    @staticmethod
    def MSE(yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: mean square error = average((yh - y)^2)
        '''
        return np.square(yh - y).mean()
    
    #-Root Mean Square Error
    @staticmethod
    def RMSE(yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: square root of mean square error
        '''
        return np.sqrt(EvalR.MSE(yh, y))
    
    #-Mean Absolute Error
    @staticmethod
    def MAE(yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: mean absolute error = average(|yh - y|)
        '''
        return np.abs(yh - y).mean()
    
    #-Median Absolute Error
    @staticmethod
    def MDAE(yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: mean absolute error = median(|yh - y|)
        '''
        return np.median(np.abs(yh - y))
    
    #-Mean Squared Log Error
    @staticmethod
    def MSLE(yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: mean square log error
        '''
        return np.mean(np.square((np.log(y + 1)-np.log(yh + 1))))
    
    #-R-squared Error
    @staticmethod
    def R_squared(yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: R-squared error = 1 - (SS[reg]/SS[total])
        '''
        #-- R_square = 1 - (SS[reg]/SS[total])
        return (1 -(np.sum(np.square(y - yh))/np.sum(np.square(y - y.mean()))))
    
    @staticmethod
    def Adjusted_Rquared(X, yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: Adjusted R-squared error = 1 - [(1 - R^2)(N-1)]/(N- p -1)
                R^2: R-squared 
                N: Total samples
                p: Number of predictors
        '''
        N, p = X.shape
        return 1 - (((1 - EvalR.R_squared(yh, y))*(N - 1))/(N - p - 1))
    
    #--HUber loss
    @staticmethod
    def Huber(yh, y, delta=None):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :param: delta
        :return: Huber loss
        '''
        loss = []
        if not delta:
            delta = 1.0
            loss.append(np.where(np.abs(y - yh) < delta,.5*(y - yh)**2 , delta*(np.abs(y-yh)-0.5*delta)))
        else:
            loss.append(np.where(np.abs(y - yh) < delta,.5*(y - yh)**2 , delta*(np.abs(y-yh)-0.5*delta)))
        return np.array(loss).mean()
    
    #--Explained Variance score
    @staticmethod
    def explainedVariance(yh, y):
        '''
        :param: yh: predicted target
        :param: y: actual target
        :return: Explained variance
        '''
        e = y - yh
        return (1 - ((np.sum(np.square(e - np.mean(e))))/(np.sum(np.square(y - y.mean())))))
    
    def summary(self, X, y, y_hat):
        '''
        :param: y_hat: predicted target
        :param: y: actual target
        :return: Loss summary
        '''
        print('*'*40)
        print('\t\tSummary')
        print('*'*40)
        print('RMSE: %s'%(EvalR.RMSE(y_hat,  y)))
        print('*'*40)
        print('MSE: %s'%(EvalR.MSE(y_hat,  y)))
        print('*'*40)
        print('MAE: %s'%(EvalR.MAE(y_hat,  y)))
        print('*'*40)
        print('MDAE: %s'%(EvalR.MDAE(y_hat,  y)))
        print('*'*40)
        print('R_squared = %s'%(EvalR.R_squared(y_hat,  y)))
        print('*'*40)
        print('Adjusted R_squared = %s'%(EvalR.Adjusted_Rquared(X, y_hat,  y)))
        print('*'*40)
        print('Huber = %s'%(EvalR.Huber(y_hat,  y)))
        print('*'*40)
        print('Explained Variance = %s'%(EvalR.explainedVariance(y_hat,  y)))
        print('*'*40)  
        
        
        
