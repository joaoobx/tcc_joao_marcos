# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:21:07 2020

@author: acer
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import bisect

class ler_csv_aneel:
    def __init__(self,endereco):
        self.endereco = endereco #endereço do csv contendo todas as fecs
        
    
    def ler_dados(self):
        #esta linha retorna um dataframe com os valores de todas as fecs
        return pd.DataFrame(pd.read_csv(self.endereco, sep = ';',skipinitialspace=True))
        
    
    def soma_colunas(self):
        aneel_tabela = []
        sizeaneel = len(list(map(int, self.ler_dados().columns)))
        for c in range (sizeaneel):
                aneel_tabela.append(np.sum(self.ler_dados().iloc[:,c]))
                c = c + 1
        return aneel_tabela
    
    
