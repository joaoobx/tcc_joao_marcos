# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:38:39 2020

@author: acer
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import bisect
import matplotlib.pyplot as plt
import ler_csv_aneel
import ler_csv_dataset
import tratar_dados
import frequencia_linha_mes
import frequencia_linha_mes_graficos

"""aqui eu carrego a tabela de 100000 dados em uma variável da classe 
ler_csv_dataset"""

circuito = ler_csv_dataset.ler_csv_dataset('dados_anos_2.csv')
"""aqui eu carrego a tabela das FECs em uma variável da classe 
ler_csv_aneel"""
fec = ler_csv_aneel.ler_csv_aneel('tabela_itapeva.csv')
#aqui eu carrego os dados anuais das FEcs
tabela_fec = fec.soma_colunas()
#aqui eu embaralho os dados fornecidos, pois, estão separados por linha
tabela_falhas = circuito.embaralha()
#aqui eu transformo essa tabela de FECs em um dataframe
fec_anos = fec.ler_dados()

#aqui eu crio uma variável da classe tratar_dados
falhas_tratar_dados = tratar_dados.tratar_dados(tabela_falhas,tabela_fec,fec_anos,'lista_mudada.txt')

a = falhas_tratar_dados.falhas_todos_anos()

#aqui eu escrevo um aquivo csv da falhas_tratar_dados
data = falhas_tratar_dados.escrever_csv()

#frequencia_linha_mes.frequencia_linha_mes(data,1, len(fec_anos.columns))

nomes_meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']

anos = []

meses = []

    
for k in range (2009 , 2009 + len(fec_anos.columns)):
    
    for l in range (0, 12):
        
        ano, mes = k , nomes_meses[l]
        anos.append(ano)
        meses.append(mes)

anos = pd.DataFrame ({'Ano':anos})
meses = pd.DataFrame ({'Mes':meses})
tabela_concat = pd.concat([anos.reset_index(drop=True), meses.reset_index(drop=True)], axis=1)



numero_linhas = data['Linha_Curto'].max()

#%%

frequencia_linha_mes_graficos.frequencia_linha_mes_graficos(data,2,11)
#%%
df_linhas = pd.DataFrame([])

for i in range (1, numero_linhas+1):
    df_linhas.insert(i-1, "Linha_" + str(i), frequencia_linha_mes.frequencia_linha_mes(data,i, len(fec_anos.columns)).loc[1])
    
frames = pd.concat([tabela_concat.reset_index(drop=True),df_linhas.reset_index(drop=True)], axis=1)

frames.to_csv(r'dados_linhas_anos.csv', index = False)



        
        
