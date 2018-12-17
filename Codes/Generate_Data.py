#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:08:15 2018

@author: harihara
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import norm
import statsmodels.api as sm
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsaplots

def brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    r = norm.rvs(size=x0.shape + (n,), scale=delta*np.sqrt(dt))
    if out is None:
        out = np.empty(r.shape)
    np.cumsum(r, axis=-1, out=out)
    out += np.expand_dims(x0, axis=-1)
    return out

def Generate_Fractional_MA(n, d=0.5, l=50):
    n = n+1
    temp = 1
    coeffs = []
    for i in range(0, l-1):
        t = (d+i)
        temp = temp*t
        fact = np.math.factorial(i)
        coeffs.append(temp/fact)
    coeffs = np.array([1] + coeffs)[::-1]
    Wn = np.random.normal(0,1,n+l-1)
    noise = []
    for i in range(l, len(Wn)):
        noise.append(np.dot(Wn[i-l:i],coeffs))
    return noise

def Generate_ARMA_Timeseries_Data(ar_parameters, ma_parameters, n_samples, frac = 1):
    if frac == 1:
        random_noise = Generate_Fractional_MA(n_samples)
    else:
        random_noise = np.random.normal(0, 1, n_samples)
    TS = np.zeros(n_samples)
    P,Q = len(ar_parameters), len(ma_parameters)
    ar_parameters = ar_parameters[::-1]
    ma_parameters = ma_parameters[::-1]    
    for i in range(0, len(TS)):
        if i < P:
            ar = random_noise[0:P]
        else:
            ar = np.multiply(TS[i-P:i], ar_parameters)
        if i < Q:
            ma = random_noise[0:Q]
        else:
            ma = np.multiply(random_noise[i-Q:i], ma_parameters)
        TS[i] = np.sum(ar) + np.sum(ma) + random_noise[i]
    return TS[max(P,Q):]

def Generate_Periodic_Signal(Fs, nsamples):
    x = np.arange(nsamples)
    y = np.sin(2 * np.pi * x / Fs)
    return y

def Simulate_ARIMA_Like_Timeseries(arparams, maparams, num_samples, A1=6, A2=0.5):
    TS = Generate_ARMA_Timeseries_Data(arparams, maparams, num_samples, frac = 0)
    TS_LRD = Generate_ARMA_Timeseries_Data(arparams, maparams, num_samples, frac = 1)
    TS_non_stationary = np.cumsum(TS)
    TS_LRD_non_stationary = np.cumsum(TS_LRD)
    W_Noise = np.random.normal(0,1, len(TS))
    TS_Periodic_F1 = Generate_Periodic_Signal(288, len(TS))
    TS_Periodic_F2 = Generate_Periodic_Signal(356/12.0*288, len(TS))
    corr, uncorr, f = pyasl.expCorrRN(len(TS), 25, mean=4.0, std=2.3, fullOut=True)
    brownian_ser = np.empty((1,len(TS)+1))
    brownian_ser[:, 0] = 50
    brownian(brownian_ser[:,0], len(TS), dt=2e-5, delta=10, out=brownian_ser[:,1:])
    brownian_ser = brownian_ser[0][:-1]
    
    df = pd.DataFrame()
    df.loc[:,'TS_ARIMA'] = TS
    df.loc[:,'TS_ARIMA_LRD'] = TS_LRD
    df.loc[:,'TS_Non_Stationary'] = TS_non_stationary
    df.loc[:,'TS_LRD_Non_Stationary'] = TS_LRD_non_stationary
    df.loc[:,'Periodic'] = A1*TS_Periodic_F1 + W_Noise
    df.loc[:,'Periodic_Multiple_Seasonality'] = (A1*TS_Periodic_F1)*(A2*TS_Periodic_F2) + W_Noise
    df.loc[:,'SARIMA'] = (A2*TS_Periodic_F1)*(A1*TS)
    df.loc[:,'SARFIMA'] = (A1*TS_Periodic_F1)*(TS_LRD)
    df.loc[:,'Periodic_Non_Stationary'] = ((A1*TS_Periodic_F1)*TS_non_stationary
                                           + W_Noise)
    df.loc[:,'Periodic_Non_Stationary_LRD'] = ((A1*TS_Periodic_F1)*TS_LRD_non_stationary 
                                               + W_Noise)
    df.loc[:,'Exp_Corr_RN'] = corr
    df.loc[:,'Brownian'] = brownian_ser
    
    color = ['red','blue','green','orange',
             'black', 'magenta','indigo','violet',
             'brown','gold','olive','gray']
    columns = df.columns.tolist()
    fig_TS, ax_TS = plt.subplots(6,2,figsize = (16,12))
    fig_ACF, ax_ACF = plt.subplots(6,2,figsize = (16,12))
    fig_FFT, ax_FFT = plt.subplots(6,2,figsize = (16,12))
    
    counter = 0
    for i in range(0, 6):
        for j in range(0, 2):
            print columns[counter]
            df[[columns[counter]]].plot(ax = ax_TS[i][j], color = color[counter], legend = False)
            ax_TS[i][j].grid()
            #ax_TS[i][j].legend(loc = 1)
            ax_TS[i][j].set_title(columns[counter])
                        
            if 'Periodic' in columns[counter]:
                lags = 5000
                lim = [0, 0.00005]
                v_lines = False
            else:
                lags = 100
                lim = [0,0.00175]
                v_lines = True
            tsaplots.plot_acf(df[columns[counter]], ax = ax_ACF[i][j], lags = lags,
                              title = 'ACF of '+columns[counter], color = color[counter],
                              marker = 'x', use_vlines = v_lines)
            ax_ACF[i][j].axhline(0)
            ax_ACF[i][j].grid()
            
            f, Pxx_den = signal.periodogram(df[columns[counter]], 1/(5*60.0))
            ax_FFT[i][j].plot(f, np.sqrt(Pxx_den), color = color[counter])
            ax_FFT[i][j].set_title('FFt of '+columns[counter])
            ax_FFT[i][j].grid()
            ax_FFT[i][j].set_xlim(lim)
            counter += 1
            
    fig_TS.tight_layout()
    fig_ACF.tight_layout()
    fig_FFT.tight_layout()
    fig_TS.savefig('TS.pdf')
    fig_ACF.savefig('ACF.pdf')
    fig_FFT.savefig('FFT.pdf')
    return df
    
arparams = np.array([0.85, -0.45,0.34,-0.098])
maparams = np.array([0.65, 0.35, -0.43])
df = Simulate_ARIMA_Like_Timeseries(arparams, maparams, 51004)
df.to_csv('Simulated_Data.csv')