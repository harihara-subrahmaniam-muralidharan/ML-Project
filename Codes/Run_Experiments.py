#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:39:40 2018

@author: harihara
"""


import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ['MKL_NUM_THREADS'] = '16'
#os.environ['GOTO_NUM_THREADS'] = '16'
#os.environ['OMP_NUM_THREADS'] = '16'
#os.environ['openmp'] = 'True'

from copy import deepcopy
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential

#import os

def Reshape_Cols(df, column):
    in_data = df[df.filter(like = 'Lookback').columns.tolist()].values
    out_data = df[[column]].values
    in_data = in_data.reshape((in_data.shape[0],1,in_data.shape[1]))
    return in_data, out_data

def Prepare_Dataset(df, column, lookback = 1, train_len = 40000):
    temp = deepcopy(df[[column]])
    for i in range(1, lookback+1):
        temp.loc[:, str(i)+'-Lookback'] = df[column].shift(i)
    x_train, y_train = Reshape_Cols(temp[lookback:train_len+lookback], column)
    x_test, y_test = Reshape_Cols(temp[train_len+lookback:], column)    
    return x_train, y_train, x_test, y_test

def build_Model(look_back, n_units, dropout_prob, bs, a, b):
    model = Sequential()
    lstm = LSTM(n_units, input_shape=(1, look_back), recurrent_dropout = dropout_prob,
                stateful=False)#, batch_input_shape=(bs, a, b))
    model.add(lstm)
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    
filepath = './Experiments-data/Simulated_Data.csv'
df = pd.read_csv(filepath, index_col = 'Time')

experiments = df.columns.tolist()[5:6]
Look_Back = [1,5,10,25,50]
N_Units = [1,5,10,25,50]
Recurrent_Probability = [0.05, 0.1, 0.2]
bs = 750
History = {}
df_train_results = pd.DataFrame()
df_test_results = pd.DataFrame()
df_Loss = pd.DataFrame()

print str(dt.datetime.now());


for e in experiments:
    for L in Look_Back:
        x_train, y_train, x_test, y_test = Prepare_Dataset(df, e, L)
        y_test = y_test[:len(y_test)/bs*bs]
        #pad = []
        #for i in range(0, L):
        #    pad.append(np.nan)
        if L == 1:
            df_train_results.loc[:,e] = (y_train.reshape(1,len(y_train))[0])
            df_test_results.loc[:,e] = y_test.reshape(1,len(y_test))[0]
        for n in N_Units:
            if L == 50 and n == 50:
                break
            for R in Recurrent_Probability:
                mdl = build_Model(L, n, R, bs, x_train.shape[1], x_train.shape[2])
                hist = mdl.fit(x_train, y_train,  epochs=1000, batch_size=bs, shuffle = False, verbose=0)
                train_predict = mdl.predict(x_train, batch_size=bs)
                test_predict = mdl.predict(x_test[:len(x_test)/bs*bs], batch_size=bs)
                
                model_str = e+'_'+str(L)+'_'+str(n)+'_'+str(R)
                print model_str, '\t------>',str(dt.datetime.now());
                print np.mean(np.abs((y_test - test_predict)/y_test))*100,np.median(np.abs((y_test - test_predict)/y_test))*100
                df_train_results.loc[:,model_str] = train_predict.reshape(1,len(train_predict))[0]
                df_test_results.loc[:,model_str] = test_predict.reshape(1,len(test_predict))[0]
                df_Loss.loc[:,model_str] = hist.history['loss']
                df_train_results.to_csv('Train_Predictions.csv')
                df_test_results.to_csv('Test_Predictions.csv')
                df_Loss.to_csv('Loss.csv')
