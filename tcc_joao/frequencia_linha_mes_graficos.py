# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 18:25:59 2020

@author: acer
"""


import matplotlib.pyplot as plt
import pandas as pd

def frequencia_linha_mes_graficos(tabela, numero, numero_anos):
    
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
    
    grafico = plt.figure(figsize=(200,30))
    plt.rcParams.update({'font.size': 200})
    eixos = grafico.add_axes([0,0,1,1])
    eixos.set_xlabel('Meses de todos os anos')
    eixos.set_ylabel('Número de Eventos')
    eixos.get_xaxis().set_ticks([])
    eixos.set_title('Eventos na linha '+str(numero))
    eixos.bar(df_data.iloc[0,:],df_data.iloc[1,:],0.5)

    return df_data