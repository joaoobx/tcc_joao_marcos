# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 07:49:09 2020

@author: acer
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import bisect

class ler_csv_dataset:
    
    def __init__(self,endereco):
        self.endereco = endereco #endereço do csv contendo os 100000 dados
        
    
    def ler_dados(self):
        #esta linha retorna um dataframe com todos os 100000 dados
        return pd.DataFrame(pd.read_csv(self.endereco, sep = ';',skipinitialspace=True))
        
    def embaralha(self):
        #esta linha irá ebaralhar todas as linhas da tabela de 100000 dados
        embaralhado = self.ler_dados()
        return embaralhado.sample(frac=1).reset_index(drop=True)
    
        

    
