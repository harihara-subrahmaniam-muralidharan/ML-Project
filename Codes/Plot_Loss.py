#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:23:43 2018

@author: harihara
"""

import pandas as pd
import matplotlib.pyplot as plt

signals = ['TS_ARIMA','TS_ARIMA_LRD','TS_Non_Stationary','TS_LRD_Non_Stationary',
           'Periodic','Periodic_Multiple_Seasonality'	,'SARIMA','SARFIMA','Periodic_Non_Stationary',	
           'Periodic_Non_Stationary_LRD','Exp_Corr_RN','Brownian']
lookahead = ['1','5','10','25','50']
n_units = lookahead
bayes_prob = ['0','0.05','0.1','0.2']

loss_data = pd.read_csv('Loss_Predictions_All_Expts.csv',index_col = ['Time'])
for s in signals:
    
    for l in lookahead:
        cols = []
        labels = []
        for n in n_units:
            if n=='50' and l == '50':
                break
            for b in bayes_prob:
                cols.append(s+'_'+l+'_'+n+'_'+b)
                labels.append('_'+l+'_'+n+'_'+b)
    fig, ax = plt.subplots(1,1)
    loss_data[cols].plot(ax = ax, legend = False)
    ax.set_title(s)