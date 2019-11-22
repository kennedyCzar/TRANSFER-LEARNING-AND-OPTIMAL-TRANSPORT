#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:33:11 2019

@author: kenneth
"""
import os
import time
from scipy.io import loadmat
from transferlearning import subspacealignment, optimaltransport
from Sinkhorn_Knopp_algorithm import sinkhorn


path = '/home/kenneth/Documents/MLDM M2/ADVANCE_ML/TRANSFER LEARNING/DATASET'
dataset = {'amazon': {'label' : {}, 'data' : {}}, 'caltech10': {'label' : {}, 'data' : {}},
           'webcam': {'label' : {}, 'data' : {}}, 'dslr': {'label' : {}, 'data' : {}}}
for ii in os.listdir(path):
    data = loadmat(os.path.join(path, ii))
    dataset[ii[:-4]]['data'] = data['fts']
    dataset[ii[:-4]]['label'] = data['labels']
    

da = ['w_d', 'c_a', 'd_c', 'c_w', 'a_d', 'w_a', 'c_d']
class_res = {'w_d': {'acc': [], 'time': []}, 'c_a': {'acc': [], 'time': []}, 'd_c': {'acc': [], 'time': []},
            'c_w': {'acc': [], 'time': []}, 'a_d': {'acc': [], 'time': []}, 'w_a': {'acc': [], 'time': []},
            'c_d': {'acc': [], 'time': []}}


#%%

for p, q in dataset.items():
    for ii in da:
        if ii == 'w_d':
            start = time.time()
            subalignacc = subspacealignment().fit_predict(dataset['webcam']['data'], dataset['webcam']['label'], dataset['dslr']['data'], dataset['dslr']['label'], d = 100, m_kernel = 'linear')
            class_res[ii]['acc'].append(subalignacc.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
            start = time.time()
            ot = optimaltransport().fit_predict(dataset['webcam']['data'], dataset['webcam']['label'], dataset['dslr']['data'], dataset['dslr']['label'])
            class_res[ii]['acc'].append(ot.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
        elif ii == 'c_a':
            start = time.time()
            subalignacc = subspacealignment().fit_predict(dataset['caltech10']['data'], dataset['caltech10']['label'], dataset['amazon']['data'], dataset['amazon']['label'], d = 100, m_kernel = 'linear')
            class_res[ii]['acc'].append(subalignacc.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
            start = time.time()
            ot = optimaltransport().fit_predict(dataset['caltech10']['data'], dataset['caltech10']['label'], dataset['amazon']['data'], dataset['amazon']['label'])
            class_res[ii]['acc'].append(ot.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
        elif ii == 'd_c':
            start = time.time()
            subalignacc = subspacealignment().fit_predict(dataset['dslr']['data'], dataset['dslr']['label'], dataset['caltech10']['data'], dataset['caltech10']['label'], d = 100, m_kernel = 'linear')
            class_res[ii]['acc'].append(subalignacc.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
            start = time.time()
            ot = optimaltransport().fit_predict(dataset['dslr']['data'], dataset['dslr']['label'], dataset['caltech10']['data'], dataset['caltech10']['label'])
            class_res[ii]['acc'].append(ot.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
        elif ii == 'c_w':
            start = time.time()
            subalignacc = subspacealignment().fit_predict(dataset['caltech10']['data'], dataset['caltech10']['label'], dataset['webcam']['data'], dataset['webcam']['label'], d = 100, m_kernel = 'linear')
            class_res[ii]['acc'].append(subalignacc.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
            start = time.time()
            ot = optimaltransport().fit_predict(dataset['caltech10']['data'], dataset['caltech10']['label'], dataset['webcam']['data'], dataset['webcam']['label'])
            class_res[ii]['acc'].append(ot.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
        elif ii == 'a_d':
            start = time.time()
            subalignacc = subspacealignment().fit_predict(dataset['amazon']['data'], dataset['amazon']['label'], dataset['dslr']['data'], dataset['dslr']['label'], d = 100, m_kernel = 'linear')
            class_res[ii]['acc'].append(subalignacc.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
            start = time.time()
            ot = optimaltransport().fit_predict(dataset['amazon']['data'], dataset['amazon']['label'], dataset['dslr']['data'], dataset['dslr']['label'])
            class_res[ii]['acc'].append(ot.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
        elif ii == 'w_a':
            start = time.time()
            subalignacc = subspacealignment().fit_predict(dataset['webcam']['data'], dataset['webcam']['label'], dataset['amazon']['data'], dataset['amazon']['label'], d = 100, m_kernel = 'linear')
            class_res[ii]['acc'].append(subalignacc.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
            start = time.time()
            ot = optimaltransport().fit_predict(dataset['webcam']['data'], dataset['webcam']['label'], dataset['amazon']['data'], dataset['amazon']['label'])
            class_res[ii]['acc'].append(ot.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
        elif ii == 'c_d':
            start = time.time()
            subalignacc = subspacealignment().fit_predict(dataset['caltech10']['data'], dataset['caltech10']['label'], dataset['dslr']['data'], dataset['dslr']['label'], d = 100, m_kernel = 'linear')
            class_res[ii]['acc'].append(subalignacc.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
            start = time.time()
            ot = optimaltransport().fit_predict(dataset['caltech10']['data'], dataset['caltech10']['label'], dataset['dslr']['data'], dataset['dslr']['label'])
            class_res[ii]['acc'].append(ot.accuracy)
            class_res[ii]['time'].append(round(time.time() - start, 3))
        
            