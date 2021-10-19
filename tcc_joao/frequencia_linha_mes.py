# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 19:12:57 2020

@author: acer
"""

import matplotlib.pyplot as plt
import pandas as pd

def frequencia_linha_mes(tabela, numero, numero_anos):
    
    linha = tabela[tabela['Linha_Curto'] == numero]
    
    freq_meses = []
    
    for i in range (1,numero_anos+1):  #colocar uma variável para o número de anos
        for j in range (1,13):
             freq_meses.append(linha[linha['data'] == (str(i)+"-"+str(j))])
             
    array_data = []
    array_freq = []
    
    for k in range (1,numero_anos+1):
        for l in range (0,12):
            mes_ano,freq = (str(k)+"-"+str(l+1)), len(freq_meses[12*(k-1)+l])
            array_data.append(mes_ano)
            array_freq.append(freq)
    
    df_data = pd.DataFrame([array_data,array_freq])
    
    return df_data
