#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:49:53 2018

@author: harihara
"""

import numpy as np
import pandas as pd
from os import getcwd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

rcParams = {'font.size': 15 , 'font.weight': 'normal', 'font.family': 'sans-serif',
            'axes.unicode_minus':False, 'axes.labelweight':'bold'}
plt.rcParams.update(rcParams)

signals = ['TS_ARIMA','TS_ARIMA_LRD','TS_Non_Stationary','TS_LRD_Non_Stationary',
           'Periodic','Periodic_Multiple_Seasonality'	,'SARIMA','SARFIMA','Periodic_Non_Stationary',	
           'Periodic_Non_Stationary_LRD','Exp_Corr_RN','Brownian']
lookahead = ['1','5','10','25','50']
n_units = lookahead
bayes_prob = ['0','0.05','0.1','0.2']

def Compute_Error(df):
    df_MAPE_Error = pd.DataFrame()
    df_RRSE_Error = pd.DataFrame()
    for s in signals:
        print s
        actuals = df[s]
        y_ = np.max(actuals) - np.min(actuals)
        for l in lookahead:
            for n in n_units:
                for b in bayes_prob:
                    pred_col = s + '_' + l +'_' + n + '_' + b
                    if l == '50' and n ==  '50':
                        pass
                    else:
                        pred = df[pred_col]
                        mape = np.abs((actuals - pred)/actuals)
                        rrse = np.sqrt(np.abs((pred - actuals)/(y_ - actuals)))
                        df_MAPE_Error.loc[:,pred_col] = mape
                        df_RRSE_Error.loc[:,pred_col] = rrse
    return df_MAPE_Error, df_RRSE_Error

def Return_Error_Table(RMSE_Table, MAPE_Table):
    error_stats = []
    for s in signals:
        print s
        for l in lookahead:
            for n in n_units:
                for b in bayes_prob:
                    pred_col = s + '_' + l +'_' + n + '_' + b
                    if l == '50' and n ==  '50':
                        pass
                    else:
                        RMSE_mu = np.mean(RMSE_Table[pred_col])
                        x = np.array(RMSE_Table[pred_col])
                        drop_med = x[~np.isnan(x)]
                        RMSE_med = np.median(drop_med)
                        MAPE_mu = np.mean(MAPE_Table[pred_col])
                        x = np.array(MAPE_Table[pred_col])
                        drop_med = x[~np.isnan(x)]
                        MAPE_med = np.median(drop_med)
                        d = {'Signal':s, 'LookAhead':l, 'Nunits':n, 'Bayesian_Dropout':b,
                             'Median_APE':MAPE_med, 'Mu_APE':MAPE_mu,
                             'Median_RSE':RMSE_med, 'Mu_RSE':RMSE_mu}
                        error_stats.append(d)
    df_Error_Statistics = pd.DataFrame(error_stats)
    df_Error_Statistics = df_Error_Statistics.set_index(['Signal','LookAhead',
                                                         'Nunits','Bayesian_Dropout'])
    return df_Error_Statistics

def Plot_Time_Series(pred_data, fp, n = '50' ):
    colors = ['black','red','blue','green','orange']
    for s in signals:
        ctr = 0
        fig, ax = plt.subplots(1,1,figsize = (16,12))
        ax.plot(pred_data[s], color = colors[ctr], label = 'Original Signal')
        for b in bayes_prob:
            ctr += 1
            ax.plot(pred_data[s+'_1_'+n+'_'+b], color = colors[ctr], label = 'Dropout='+b)
        ax.set_xlim([0,1000])
        ax.grid()
        ax.set_title(s)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fp+s+'_'+n+'.pdf')
        plt.close()
        
def Plot_Error_Distribution(df_Error, fp):
    for s in signals:
        cols = []
        rename = []
        for l in lookahead:
            for n in n_units:
                if n== '50' and l == '50':
                    pass
                else:
                    for b in bayes_prob:
                        cols.append(s+'_'+l+'_'+n+'_'+b)
                        rename.append(l+'_'+n+'_'+b)
        df_filter = df_Error[cols]
        df_filter = df_filter.rename(columns = dict(zip(cols, rename)))
        print df_filter.head()
        fig,ax = plt.subplots(1,1, figsize = (18,12))
        df_filter.boxplot(ax = ax, showfliers = False, grid = False)
        ax.set_title(s)
        for tick in ax.get_xticklabels():
                tick.set_rotation(90)
        fig.tight_layout()
        fig.savefig(fp+s+'.pdf')
        plt.close()
        
def Plot_Error_Line_Charts(df_Error_Statistics, fp):
    df_Error_Statistics = df_Error_Statistics.reset_index()
    df_Error_Statistics['Nunits'] = df_Error_Statistics['Nunits'].astype(int)
    lookback_colors = dict(zip(lookahead,['red','blue','green','orange','black']))
    bayes_marker = dict(zip(bayes_prob,['-','--',':','-.']))
    
    for s in signals:
        fig, ax = plt.subplots(1,1,figsize = (16,12))
        ax.set_title(s)
        for l in lookback_colors:
            for b in bayes_marker:
                temp = df_Error_Statistics[(df_Error_Statistics['Signal'] == s)]
                temp = temp[temp['LookAhead'] == l]
                temp = temp[temp['Bayesian_Dropout'] == b]
                ax.plot(temp['Nunits'].tolist(), temp['Median_APE'].tolist(), 
                        color = lookback_colors[l],linestyle = bayes_marker[b], 
                        marker = 'o')
                
        red_line = mlines.Line2D([], [], color='red', marker='o', label='Lookback = 1')
        blue_line = mlines.Line2D([], [], color='blue', marker='o', label='Lookback = 5')
        green_line = mlines.Line2D([], [], color='green', marker='o', label='Lookback = 10')
        orange_line = mlines.Line2D([], [], color='orange', marker='o', label='Lookback = 25')
        black_line = mlines.Line2D([], [], color='black', marker='o', label='Lookback = 50')
        
        solid_line = mlines.Line2D([], [], linestyle='-', color = 'black', label='Dropout = 0')
        dash_line = mlines.Line2D([], [], linestyle='--', color = 'black', label='Dropout = 0.05')
        dotted_line = mlines.Line2D([], [], linestyle=':',color = 'black',  label='Dropout = 0.10')
        dot_line = mlines.Line2D([], [], linestyle='-.',  color = 'black', label='Dropout = 0.20')
        
        ax.legend(handles = [red_line, blue_line, green_line, orange_line, black_line,
                   solid_line, dash_line, dotted_line, dot_line], ncol = 3, loc = 1)
        ax.set_xlabel('No of Units')
        ax.set_ylabel('Median Relative Error')
        ax.grid()
        fig.tight_layout()
        fig.savefig(fp+s+'.pdf')
        plt.close()
        
pred_data = pd.read_csv('../All-Data/Test_Predictions_All_Expts.csv',index_col = ['Time'])
MAPE, RRSE = Compute_Error(pred_data)
Err_Stats = Return_Error_Table(RRSE, MAPE)
Plot_Error_Line_Charts(Err_Stats, '../Figures/Line Plots/')
Plot_Time_Series(pred_data, '../Figures/Time Series Plots/')
Plot_Error_Distribution(MAPE, '../Figures/Box Plots/')