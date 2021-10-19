# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:36:09 2020

@author: acer
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import bisect

class tratar_dados:
    
    def __init__(self,tabela,aneel,fec,lista):
        self.tabela = tabela #esta variavel é a tabela de 100000 dados
        self.aneel = aneel #este é um array com as fecs anuais
        self.fec = fec #esta e a tabela de todas das fecs de todos os meses
        self.lista = lista # esta é a lista de todas as linhas e zonas
        
    def falhas_todos_anos(self):
        """esta linha somará todos os valores e dividirá todos por esta soma,
        assim resultará em um array de probabilidades que são menores que um
        e que a soma de todos os valores é um. depois disso todos os valores
        deste array de probabilidades serão multiplicados pelo tamanho da 
        tabela de 100000 dados para formar um array cuja soma dos valores é
        10000"""
        probs = (self.aneel/(np.sum(self.aneel)))*len(self.tabela)
        
        #esta linha arredondará os valores do array probs
        probs_arredondado = np.round(probs)
        
        """esta parte vai evitar que o probs_arredondado gere mais dados do que
        o tamanho do dataframe, se for maior ou menor ele fará mais a diferença
        que será negativa se for maior e positiva se for menor"""
        if (np.sum(probs_arredondado)) != len(self.tabela):
            probs_arredondado[-1] = probs_arredondado[-1] + ((len(self.tabela)) - np.sum(probs_arredondado))
        
        """esta linha somará os valores da probs_arredondado para verificar se 
        este array tem realmente os 10000 dados"""
        
        """esta parte ira usar o probs_arredondado para dividir a tabela de 
        100000 dados em uma lista de listas em que cada uma destas listas contem
        uma porção desta tabela de acordo com cada numero do array probs_arre
        dondado, por exemplo, se o primeiro valor do array for 1200, então o 
        algoritmo pegara as primeiras 1200 linhas da tabela e armazenara na
        primeira lista da lista de listas dados_anos e assim por diante para 
        todos os valores de probs_arredondado"""
        cumsum = 0
        dados_anos = []
        size = len(probs_arredondado)
        for i in range(size):
            if cumsum == 0:
                dados_anos.append(self.tabela.loc[cumsum:(cumsum+probs_arredondado[i])])
                cumsum = cumsum + probs_arredondado[i]
            else:
                dados_anos.append(self.tabela.loc[cumsum+1:(cumsum+probs_arredondado[i])])
                cumsum = cumsum + probs_arredondado[i]
        
        #este for resetará os indices da tabela
        for i in range(size):
            dados_anos[i] = dados_anos[i].reset_index()
        
        """#esta parte irá plotar o grafico do numero de eventos em todos os anos
        anos = list(map(int, self.fec.columns)) 
        y_pos = np.arange(len(anos))
        erros_anos = np.array(probs_arredondado[0:(size)])
        plt.bar(y_pos,erros_anos,align='center',alpha=0.5)
        plt.xticks(y_pos, anos, rotation='vertical')
        plt.ylabel('Eventos')
        plt.title('Ano')
        plt.show()"""
    
        return dados_anos
    
    def falhas_no_ano(self,a):
        dados_anos = self.falhas_todos_anos()
        #nesta parte o usuario escolhera que ano deseja ver
        """print ("Que ano você deseja ver?")
        a = int(input())""" 
        
        """o algoritmo então escolhera o ano na tabela de fecs sendo que este
        ano contem todas as fecs de todos os meses referentes a ele"""
        
        random = self.fec.iloc[:,a]
        
        """esta linha somará todos os valores das fecs e dividirá todos 
        por esta soma,assim resultará em um array de probabilidades que são 
        menores que ume que a soma de todos os valores é um. depois disso todos 
        os valoresdeste array de probabilidades serão multiplicados pelo tamanho
        da lista que o usuario escolheu para formar um array cuja soma dos 
        valores é o tamanho desta lista"""
        probs = (random/(np.sum(random)))*len(dados_anos[a])
        
        #esta linha arredondará os valores do array probs
        probs_ano = np.around(probs)
        
        
        """esta parte ira usar o probs_arredondado para dividir a lista seleionada
        em uma lista de listas em que cada uma destas listas contem
        uma porção desta tabela de acordo com cada numero do array probs_ano,
        por exemplo, se o primeiro valor do array for 120, então o 
        algoritmo pegara as primeiras 120 linhas da tabela e armazenara na
        primeira lista da lista de listas dados_meses e assim por diante para 
        todos os valores de probs_ano"""
        falhas_ano_atual = dados_anos[a]
        cumsum = 0
        dados_meses = []
        sizemeses = len (probs_ano)
        for b in range (sizemeses):
            if cumsum == 0:
                dados_meses.append(falhas_ano_atual.loc[cumsum:(cumsum+probs_ano[b])])
                cumsum = cumsum + probs_ano[b]
            else:
                dados_meses.append(falhas_ano_atual.loc[cumsum+1:(cumsum+probs_ano[b])])
                cumsum = cumsum + probs_ano[b]
        
        """#esta parte irá plotar o grafico do numero de eventos em todos os anos
        meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
        y_pos = np.arange(len(meses))
        erros_meses_1 = np.array(probs_ano[0:sizemeses])
        plt.bar(y_pos,erros_meses_1,align='center',alpha=0.5)
        plt.xticks(y_pos, meses,rotation='vertical')
        plt.ylabel('Eventos')
        plt.title('Meses')
        plt.show()"""
        
        return dados_meses
    
    def escrever_csv (self):
        
        #definição de variaveis
        anos_e_meses = []
        meses_concatenados = []
        tudo_concatenado = []
        
        """este for mostrará uma lista de listas contendo dados dos anos 
        separados em meses"""
        for j in range (11):
            anos_e_meses.append(self.falhas_no_ano(j))
        
        #este for colocará a data em tudo
            for i in range (12):
                anos_e_meses[j][i]['data'] = "{}-{}".format(j+1, i+1)
        
        """este for ira concatenar todos os meses de um ano em um dataframe
        só e isto para todos os anos"""
        for anos in range (11):
            meses_concatenados.append(pd.concat(anos_e_meses[anos]))
        
        """esta linha vai concatenar todos os anos em um dataframe só"""
        tudo_concatenado = pd.concat(meses_concatenados)
        
        """esta parte colocará este dataframe em um arquivo csv"""
        tudo_concatenado.to_csv('dados_tcc_datas.csv')
        tudo_concatenado.to_csv('C:/Users/acer/Desktop/dados_tcc_datas.csv')
        
        return tudo_concatenado
    
    def separar_zonas(self):
        
        #nesta parte o usuario escolhera o mes que deseja ver
        print ("Que mês você deseja ver?")
        
        """esta parte subtraira um do numero digitado para que se o mes 1 for 
        escolhido, o valor selecionado na lista de lista seja o zero"""
        mes = int(input()) - 1
            
        """esta parte ira transformar o arquivo txt contendo as linhas e zonas
        em um dataframe para que possa ser trabalhado pelo algoritmo"""
        results = []
        with open(self.lista) as inputfile:
            for line in inputfile:
                results.append(line.strip().split(',')) 
        
        
        def erros(tabela):
        
            """esta parte separar as linhas da lista selecionada nas zonas corres
            pondentes e armazenara no array dados_zonas em que cada posição é 
            uma zona. apos isso, mostrando qual é, e quantas falhas possui"""
            cumsum4 = 0
            dados_zonas = []
            size_zonas = len(results)
            for d in range(size_zonas):
                dados_zonas.append(np.isin(tabela["Linha_Curto"].ravel(),[np.array(results[d]).astype(int)]).sum())
                print ("Zona ", d+1, ": ", dados_zonas[d], " eventos.")
                d = d + 1
            
            return (dados_zonas)
            
        zona = erros (dados_meses[mes])
        