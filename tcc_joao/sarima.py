# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:06:10 2020

@author: acer
"""
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
import pandas as pd
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'G'

tabela = pd.read_excel('dados_linhas_anos.xlsx')

datas = []

for k in range (tabela['Ano'].min(), tabela['Ano'].max()+1):
        for l in range (1,13):
            if l < 10:
                ano_mes = (str(k)+"-0"+str(l))
                datas.append(ano_mes)
            else:
                ano_mes = (str(k)+"-"+str(l))
                datas.append(ano_mes)

datas =  pd.DataFrame({'datas':datas})

eventos = pd.DataFrame(tabela['Linha_2'])
                
df = pd.concat([datas.reset_index(drop=True),eventos.reset_index(drop=True)], axis=1)

df['datas'] = pd.to_datetime(df['datas'],format='%Y-%m')
                

y = df.set_index(['datas'])
print (y.head(5))


"""
y.plot(figsize=(19, 4))
plt.show()
"""

"""from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()
"""




p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter for SARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
        except: 
            continue



        
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
#print(results.summary().tables[1])



pred = results.get_prediction(start=pd.to_datetime('2019-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2019-01':].plot(label='observed')
real = y['2019-01':]
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(5, 4))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Num_Eventos')
plt.legend()
plt.show()

pred_ci = pred_ci.assign(mean=pred_ci.mean(axis=1))
simulado = pred_ci['mean'].values

def sarima_error_func(dados):
    from math import sqrt
    
    from sklearn.metrics import mean_squared_error
    
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['text.color'] = 'G'
    
    tabela = pd.read_excel('c:/Users/acer/Desktop/dados_linhas_anos.xlsx')
    
    datas = []
    
    for k in range (tabela['Ano'].min(), tabela['Ano'].max()+1):
            for l in range (1,13):
                if l < 10:
                    ano_mes = (str(k)+"-0"+str(l))
                    datas.append(ano_mes)
                else:
                    ano_mes = (str(k)+"-"+str(l))
                    datas.append(ano_mes)
    
    datas =  pd.DataFrame({'datas':datas})
    
    eventos = pd.DataFrame(dados)
                    
    df = pd.concat([datas.reset_index(drop=True),eventos.reset_index(drop=True)], axis=1)
    
    df['datas'] = pd.to_datetime(df['datas'],format='%Y-%m')
                    
    
    y = df.set_index(['datas'])
    print (y.head(5))
    
    
    y.plot(figsize=(19, 4))
    plt.show()
    
    
    from pylab import rcParams
    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
    plt.show()
    
    
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter for SARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
            except: 
                continue
            
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=(0, 0, 1),
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])
    
    pred = results.get_prediction(start=pd.to_datetime('2019-01'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = y['2019-01':].plot(label='observed')
    real = y['2019-01':]
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(5, 4))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    pred_ci = pred_ci.assign(mean=pred_ci.mean(axis=1))
    simulado = pred_ci['mean'].values
    erro_sarima = sqrt(mean_squared_error(real, simulado))
    
    
    return erro_sarima

def sarima_func(dados):
    from math import sqrt
    
    from sklearn.metrics import mean_squared_error
    
    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['text.color'] = 'G'
    
    tabela = pd.read_excel('c:/Users/acer/Desktop/dados_linhas_anos.xlsx')
    
    datas = []
    
    for k in range (tabela['Ano'].min(), tabela['Ano'].max()+1):
            for l in range (1,13):
                if l < 10:
                    ano_mes = (str(k)+"-0"+str(l))
                    datas.append(ano_mes)
                else:
                    ano_mes = (str(k)+"-"+str(l))
                    datas.append(ano_mes)
    
    datas =  pd.DataFrame({'datas':datas})
    
    eventos = pd.DataFrame(dados)
                    
    df = pd.concat([datas.reset_index(drop=True),eventos.reset_index(drop=True)], axis=1)
    
    df['datas'] = pd.to_datetime(df['datas'],format='%Y-%m')
                    
    
    y = df.set_index(['datas'])
    print (y.head(5))
    
    
    y.plot(figsize=(19, 4))
    plt.show()
    
    
    from pylab import rcParams
    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
    plt.show()
    
    
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter for SARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
            except: 
                continue
            
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=(0, 0, 1),
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])
    
    pred = results.get_prediction(start=pd.to_datetime('2019-01'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = y['2019-01':].plot(label='observed')
    real = y['2019-01':]
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(5, 4))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    pred_ci = pred_ci.assign(mean=pred_ci.mean(axis=1))
    simulado = pred_ci['mean'].values
    erro_sarima = sqrt(mean_squared_error(real, simulado))
    
    
    return real,simulado
    