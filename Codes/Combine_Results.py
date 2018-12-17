#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:24:55 2018

@author: harihara
"""

import numpy as np
import pandas as pd
from os import listdir, getcwd

suffix = ['0','1','2','3','4','5','6','7']
filepath = getcwd()+'/Results/'

df_Test_Combined = pd.DataFrame()
df_Train_Combined = pd.DataFrame()
df_Loss_Combined = pd.DataFrame()
for s in suffix:
    print s
    
    print filepath+'Test_Predictions-'+s+'.csv'
    df_Test = pd.read_csv(filepath+'Test_Predictions-'+s+'.csv', index_col = 'Unnamed: 0')
    df_Test.index.name = 'Time'
    df_Test_Combined = df_Test_Combined.join(df_Test, how = 'outer',lsuffix='_left', rsuffix='_right')
    
    print filepath+'Train_Predictions-'+s+'.csv'
    df_Train = pd.read_csv(filepath+'Train_Predictions-'+s+'.csv', index_col = 'Unnamed: 0')
    df_Train.index.name = 'Time'
    df_Train_Combined = df_Train_Combined.join(df_Train, how = 'outer',lsuffix='_left', rsuffix='_right')
    
    print filepath+'Loss-'+s+'.csv'
    df_Loss = pd.read_csv(filepath+'Loss-'+s+'.csv', index_col = 'Unnamed: 0')
    df_Loss.index.name = 'Time'
    df_Loss_Combined = df_Loss_Combined.join(df_Loss, how = 'outer',lsuffix='_left', rsuffix='_right')
  
signals = ['TS_ARIMA','TS_ARIMA_LRD','TS_Non_Stationary','TS_LRD_Non_Stationary',
           'Periodic','Periodic_Multiple_Seasonality'	,'SARIMA','SARFIMA','Periodic_Non_Stationary',	
           'Periodic_Non_Stationary_LRD','Exp_Corr_RN','Brownian']
lookahead = ['1','5','10','25','50']
n_units = lookahead
bayes_prob = ['0.05','0.1','0.2']

df_Test_Combined.columns = df_Test_Combined.columns.str.replace("_right", "")
df_Test_Combined.drop(list(df_Test_Combined.filter(regex = '_left')), axis = 1, inplace = True)

df_Loss_Combined.columns = df_Loss_Combined.columns.str.replace("_right", "")
df_Loss_Combined.drop(list(df_Loss_Combined.filter(regex = '_left')), axis = 1, inplace = True)

df_Test_Combined.to_csv('Test_Predictions_All_Expts.csv')
df_Train_Combined.to_csv('Train_Predictions_All_Expts.csv')
df_Loss_Combined.to_csv('Loss_Predictions_All_Expts.csv')