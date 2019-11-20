# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:34:47 2019

@author: Jimmy
"""

import csv
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def read_csv(filepath):
    
    with open(filepath,newline='') as csvfile:
        rows = csv.reader(csvfile)
        num = 0
        data = []
        for row in rows:
            if num > 0:
                data.append(row)
            num += 1    
    return data          


def cross_vad_split(data, nfold=5):
    
    keys = list(data.keys())
    random.shuffle(keys)
    shuffledata = dict()
    for key in keys:
        shuffledata.update({key:data[key]})
    
    shuffledataD = list(shuffledata.keys())
    shuffledataT = [c[0] for c in list(shuffledata.values())]
    shuffledataX = [c[1:] for c in list(shuffledata.values())]
    
    dataD_nfold = []
    dataX_nfold = []
    dataT_nfold = []
    
    for i in range(0,len(shuffledataD),len(shuffledataD)//nfold):
        if len(dataD_nfold) == nfold:
            dataD_nfold[-1] += shuffledataD[i:i+len(shuffledataD)//nfold]
            dataX_nfold[-1] += shuffledataX[i:i+len(shuffledataX)//nfold]
            dataT_nfold[-1] += shuffledataT[i:i+len(shuffledataT)//nfold]
            break
        
        dataD_nfold.append(shuffledataD[i:i+len(shuffledataD)//nfold])
        dataX_nfold.append(shuffledataX[i:i+len(shuffledataX)//nfold])
        dataT_nfold.append(shuffledataT[i:i+len(shuffledataT)//nfold])
    
    return dataD_nfold, dataX_nfold, dataT_nfold    
    

def polynomial(x, basis, mu, sigma, M):
    
    x = basic_func(x, basis, mu, sigma)
    
    if M==2:
        xx = []
        for i in range(x.shape[0]):
            for j in range(i, x.shape[0]):
                xx.append(x[i]*x[j])
        xx = np.array(xx)
        
        x_param_item = np.concatenate((x,xx))
    elif M==1:
        x_param_item = x
    
    return x_param_item


def error_fcn(pred, gt, w, lamda):
    
    E = np.sum((pred-gt)**2) / (2*gt.shape[0]) + lamda * np.dot(w.transpose(),w) / 2
    
    return E

def basic_func(x, basis, mu, sigma):

    if basis == 'poly': 
        x_basis = x
    elif basis == 'gauss':
        #x_basis = np.array([math.exp(-(x[i]-np.mean(x))**2/(2*np.std(x)**2)) for i in range(len(x))])
        x_basis = np.array([math.exp(-(x[i]-mu)**2/(2*sigma**2)) for i in range(len(x))])
    elif basis == 'sigmoid':
        #x_basis = np.array([1/(1+math.exp(-(x[i]-np.mean(x))/np.std(x))) for i in range(len(x))])
        x_basis = np.array([1/(1+math.exp(-(x[i]-mu)/sigma)) for i in range(len(x))])
    
    return x_basis

def train(x_train, y_train, M, lamda, basis, mu, sigma):
    
    #if M == 2:    
    x_param = []
    for i in range(x_train.shape[0]):
        x_param.append(polynomial(x_train[i], basis, mu, sigma, M))   
    x_param = np.array(x_param)
    #elif M == 1:
        #x_param = x_train
    
    A = np.dot(x_param.transpose(),x_param)
    w = np.dot(np.dot(np.linalg.inv(lamda * np.eye(A.shape[0],A.shape[1]) + A),x_param.transpose()),y_train)
    y_pred = np.dot(x_param,w)
    
    E = error_fcn(y_pred, y_train, w, lamda)
    RMS_e = math.sqrt(np.sum((y_pred-y_train)**2) / y_train.shape[0])    
    
    return RMS_e, w
 
    
def evaluation(x_valid, y_valid, w, M, lamda, basis, mu, sigma):
    
    #if M == 2:    
    x_param = []
    for i in range(x_valid.shape[0]):
        x_param.append(polynomial(x_valid[i], basis, mu, sigma, M))   
    x_param = np.array(x_param)
    #elif M == 1:
        #x_param = x_valid
    
    y_pred = np.dot(x_param, w)
    E = error_fcn(y_pred, y_valid, w, lamda)
    RMS_e = math.sqrt(np.sum((y_pred-y_valid)**2) / y_valid.shape[0])
    
    return RMS_e


if __name__ == '__main__':
    
    dataX_path = './Dataset/dataset_X.csv'
    dataT_path = './Dataset/dataset_T.csv'
    basis = 'poly'
    M = 2
    lamda = 0
    normalize = True  
    
    dataX = read_csv(dataX_path)
    dataT = read_csv(dataT_path)
    
    dataD = [c[0] for c in dataX] # date of data
    dataX = [c[1:] for c in dataX] # data
    dataT = [c[1:] for c in dataT] # target

    data = {}
    
    for i in range(len(dataD)):
        data[dataD[i]] = dataT[i] + dataX[i]
    
    nfold = 5
    dataD_nfold, dataX_nfold, dataT_nfold = cross_vad_split(data, nfold=nfold)
    
    x_train = np.array([item for i in range(nfold-1) for item in dataX_nfold[i]]).astype(float)
    x_valid = np.array([item for item in dataX_nfold[nfold-1]]).astype(float)
    y_train = np.array([item for i in range(nfold-1) for item in dataT_nfold[i]]).astype(float)
    y_valid = np.array([item for item in dataT_nfold[nfold-1]]).astype(float)
    
    if normalize:
        x_train /= x_train.max(axis=0)
        x_valid /= x_valid.max(axis=0)
        
    # x_train = np.column_stack((x_train[:,2:3],x_train[:,3:4],x_train[:,5:6],x_train[:,6:7],x_train[:,8:9])) 
    # x_valid = np.column_stack((x_valid[:,2:3],x_valid[:,3:4],x_valid[:,5:6],x_valid[:,6:7],x_valid[:,8:9])) 
    
    mu = np.mean(x_train)
    sigma = np.std(x_train)

    RMS_train, w = train(x_train, y_train, M=M, lamda=lamda, basis=basis, mu=mu, sigma=sigma)
    RMS_valid = evaluation(x_valid, y_valid, w, M=M, lamda=lamda, basis=basis, mu=mu, sigma=sigma)