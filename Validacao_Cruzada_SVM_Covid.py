#Com base nos resultados de exames laboratoriais comumente coletados para um 
#caso suspeito de COVID-19 durante uma visita à sala de emergência, 
#seria possível prever o resultado do teste para SARS-Cov-2 ?
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

base  = pd.read_excel('covid.xlsx')
exclude_columns = [
    'Patient addmited to regular ward (1=yes, 0=no)',
    'Patient addmited to semi-intensive unit (1=yes, 0=no)',
    'Patient addmited to intensive care unit (1=yes, 0=no)',
    'Mycoplasma pneumoniae', 'Influenza B, rapid test', 
    'Influenza A, rapid test',	'Alanine transaminase',	
    'Aspartate transaminase',	
	'Total Bilirubin',	'Direct Bilirubin',	'Indirect Bilirubin',	
    'Magnesium', 'pCO2 (venous blood gas analysis)',	
    'Hb saturation (venous blood gas analysis)',	
    'Base excess (venous blood gas analysis)',
	'pO2 (venous blood gas analysis)',
	'Fio2 (venous blood gas analysis)',	
    'Total CO2 (venous blood gas analysis)',
	'pH (venous blood gas analysis)', 'HCO3 (venous blood gas analysis)'
]
base = base.drop(columns = exclude_columns)
previsores = base.iloc[:, 1:45].values
classe = base.iloc[:, 0].values    

# Pre processamento dos valores faltantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(previsores[:, 0:43])
previsores = imputer.transform(previsores[:, 0:43])

# Efetuando o escalonamento (normalizacao)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisao de previsores de treinamento e testes
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

# Treinamento do algoritmo
from sklearn.svm import SVC
classificador = SVC(kernel='rbf', random_state=1, C=1.0)
classificador.fit(previsores_treinamento,classe_treinamento)

# Efetuar a predicao
previsoes = classificador.predict(previsores_teste)

# Analise do nivel de acertos
from sklearn.metrics import confusion_matrix,accuracy_score
precisao = accuracy_score(classe_teste,previsoes)
matriz = confusion_matrix(classe_teste,previsoes)


a = np.zeros(5)
previsores.shape
previsores.shape[0]
b = np.zeros(shape=(previsores.shape[0], 1))


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
resultados = []
matrizes = []
for indice_treinamento, indice_teste in kfold.split(previsores, 
                                                    np.zeros(shape=(previsores.shape[0], 1))):
    #print('indice treinamento:', indice_treinamento, 'indice teste: ', indice_teste)
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoes)
    matrizes.append(confusion_matrix(classe[indice_teste], previsoes))
    resultados.append(precisao)

matriz_final = np.mean(matrizes, axis = 0)   
resultados = np.asarray(resultados)
media = resultados.mean()
desvio = resultados.std()


