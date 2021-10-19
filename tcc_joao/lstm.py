# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 08:12:54 2020

@author: acer
"""
from pandas import DataFrame
from pandas import Series
from pandas import concat
import pandas as pd
import statistics
from pandas import read_csv
from pandas import datetime
import sklearn.metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib
import numpy
from numpy import concatenate
import matplotlib.pyplot as plt
from sarima import pred_ci
from sarima import sarima_func
from sarima import sarima_error_func
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df = df.drop(0)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    return model

# Update LSTM model
def update_model(model, train, batch_size, updates):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    for i in range(updates):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

# run a repeated experiment
def experiment(repeats, series, updates):
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    # split data into train and test-sets
    train, test = supervised_values[0:-12], supervised_values[-12:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    # run experiment
    error_scores = list()
    for r in range(repeats):
        # fit the base model
        lstm_model = fit_lstm(train_scaled, 1, 50, 15)
        #aqui eu salvo o modelo -----> A FAZER
        # forecast test dataset
        train_copy = numpy.copy(train_scaled)
        predictions = list()
        for i in range(len(test_scaled)):
            # update model
            if i > 0:
                update_model(lstm_model, train_copy, 1, updates)
            # predict
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
            # store forecast
            predictions.append(yhat)
            # add to training set
            train_copy = concatenate((train_copy, test_scaled[i,:].reshape(1, -1)))
        # report performance
        rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
        print('%d) Test RMSE: %.3f' % (r+1, rmse))
        error_scores.append(rmse)
        real = raw_values[-12:]
        previsoes = predictions
        return real,previsoes  
    
def experiment_rmse(repeats, series, updates):
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    # split data into train and test-sets
    train, test = supervised_values[0:-12], supervised_values[-12:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    # run experiment
    error_scores = list()
    for r in range(repeats):
        # fit the base model
        lstm_model = fit_lstm(train_scaled, 1, 50, 5)
        # forecast test dataset
        train_copy = numpy.copy(train_scaled)
        predictions = list()
        for i in range(len(test_scaled)):
            # update model
            if i > 0:
                update_model(lstm_model, train_copy, 1, updates)
            # predict
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
            # store forecast
            predictions.append(yhat)
            # add to training set
            train_copy = concatenate((train_copy, test_scaled[i,:].reshape(1, -1)))
        # report performance
        rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
        print('%d) Test RMSE: %.3f' % (r+1, rmse))
        error_scores.append(rmse)
        real = raw_values[-12:]
        previsoes = predictions
        
        return rmse

def lstm_error_run(dataset):
    
    series = dataset

    # experiment
    repeats = 3
    results = DataFrame()
    # run experiment
    updates = 3
    rmse = experiment_rmse(repeats, series, updates)
    
    return rmse

def lstm_data_run(dataset):
    
    series = dataset

    # experiment
    repeats = 10
    results = DataFrame()
    # run experiment
    updates = 20
    real, previsao = experiment(repeats, series, updates)
    
    return real,previsao

def rmse (database):
    
    rmse = DataFrame([])
    
    for i in range (0, len(database.columns), 2):
        
        erro = sqrt(mean_squared_error(database.iloc[:,i], database.iloc[:,i+1]))
        erro = DataFrame([erro])
        rmse = pd.concat([rmse.reset_index(drop=True), erro.reset_index(drop=True)], axis=1)
    
    return rmse
        
          
        
        
    
    

    
    
       
# load dataset
csv = read_csv('dados_linhas_anos.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

#%%


erros_geral_lstm = DataFrame([])
erros_geral_sarima = DataFrame([])
erros_geral_naive = DataFrame([])
erros_geral_ensemble = DataFrame([])

num_linhas = 1

#%%

for i in range (1, num_linhas+1):
    real, prev_lstm = lstm_data_run(csv.iloc[:,i])
    real = DataFrame(real)
    prev_lstm = DataFrame(prev_lstm)
    real.columns=['Real_Linha_'+str(i)]
    prev_lstm.columns=['Prev_Linha_'+str(i)]
    erros_geral_lstm = pd.concat([erros_geral_lstm.reset_index(drop=True), real.reset_index(drop=True)], axis=1)
    erros_geral_lstm = pd.concat([erros_geral_lstm.reset_index(drop=True), prev_lstm.reset_index(drop=True)], axis=1)
#%%
for i in range (1, num_linhas+1):
    real, prev_sarima = sarima_func(csv.iloc[:,i])
    real = DataFrame(real)
    prev_sarima = DataFrame(prev_sarima)
    real.columns=['Real_Linha_'+str(i)]
    prev_sarima.columns=['Prev_Linha_'+str(i)]
    erros_geral_sarima = pd.concat([erros_geral_sarima.reset_index(drop=True), real.reset_index(drop=True)], axis=1)
    erros_geral_sarima = pd.concat([erros_geral_sarima.reset_index(drop=True), prev_sarima.reset_index(drop=True)], axis=1) 
#%%
for i in range (1, num_linhas+1):
    linha_atual = csv['Linha_'+str(i)]
    real, prev_naive = linha_atual[-12:], linha_atual[-25:-13]
    real = DataFrame(real)
    prev_naive = DataFrame(prev_naive)
    real.columns=['Real_Linha_'+str(i)]
    prev_naive.columns=['Prev_Linha_'+str(i)]
    erros_geral_naive = pd.concat([erros_geral_naive.reset_index(drop=True), real.reset_index(drop=True)], axis=1)
    erros_geral_naive = pd.concat([erros_geral_naive.reset_index(drop=True), prev_naive.reset_index(drop=True)], axis=1)
  
#%% 
erros_geral_ensemble = DataFrame([])
for i in range (0, len(erros_geral_lstm.columns), 2):
    real = erros_geral_sarima.iloc[:,i]
    sarima_lstm = pd.concat([erros_geral_lstm.iloc[:,i+1].reset_index(drop=True), erros_geral_sarima.iloc[:,i+1].reset_index(drop=True)], axis=1)
    prev_ensemble = sarima_lstm.mean(axis=1)
    real = DataFrame(real)
    prev_ensemble = DataFrame(prev_ensemble)
    if i==0: 
        real.columns=['Real_Linha_'+str(1)]
        prev_ensemble.columns=['Prev_Linha_'+str(1)]
    elif i==2:
        real.columns=['Real_Linha_'+str(2)]
        prev_ensemble.columns=['Prev_Linha_'+str(2)]
    else:
        real.columns=['Real_Linha'+str(i-1)]
        prev_ensemble.columns=['Prev_Linha'+str(i-1)]
    erros_geral_ensemble = pd.concat([erros_geral_ensemble.reset_index(drop=True), real.reset_index(drop=True)], axis=1)
    erros_geral_ensemble = pd.concat([erros_geral_ensemble.reset_index(drop=True), prev_ensemble.reset_index(drop=True)], axis=1)
#%%
rmse_lstm = rmse(erros_geral_lstm).transpose()
rmse_naive = rmse(erros_geral_naive).transpose()
rmse_sarima = rmse(erros_geral_sarima).transpose()
rmse_ensemble = rmse(erros_geral_ensemble).transpose()

rmse_todos = DataFrame ([])

rmse_todos = pd.concat([rmse_todos.reset_index(drop=True), rmse_lstm.reset_index(drop=True)], axis=1)
rmse_todos = pd.concat([rmse_todos.reset_index(drop=True), rmse_naive.reset_index(drop=True)], axis=1)
rmse_todos = pd.concat([rmse_todos.reset_index(drop=True), rmse_sarima.reset_index(drop=True)], axis=1)
rmse_todos = pd.concat([rmse_todos.reset_index(drop=True), rmse_ensemble.reset_index(drop=True)], axis=1)

medias = rmse_todos.mean()

medias = DataFrame(medias)

medias = medias.transpose()

rmse_todos = rmse_todos.append(medias)

nome_linhas = []

for i in range (0, len(rmse_todos.index)):
    
    if i<len(rmse_todos.index)-1:
        nome_linhas.append("Linha_"+str(i+1))
    else:
        nome_linhas.append("Média")

nome_colunas = ['LSTM','Naive', 'SARIMA', 'Ensemble']


rmse_todos.index = nome_linhas

rmse_todos.columns = nome_colunas

rmse_todos.to_excel(r'c:/Users/acer/Desktop/rmse_todos_2.xlsx', index = True)