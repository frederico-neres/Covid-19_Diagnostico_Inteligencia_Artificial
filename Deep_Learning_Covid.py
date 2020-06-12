#Com base nos resultados de exames laboratoriais comumente coletados para um 
#caso suspeito de COVID-19 durante uma visita à sala de emergência, 
#seria possível prever o resultado do teste para SARS-Cov-2 ?
# -*- coding: utf-8 -*-
# Importacao das bibliotecas
import keras
import pandas as pd
from keras.utils import  np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

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
imputer = imputer.fit(previsores[:, 0:45])
previsores = imputer.transform(previsores[:, 0:45])

# Efetuando o escalonamento (normalizacao)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Separando dados de treinamento e testes
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.10)

classificador = Sequential()
classificador.add(Dense(units = 43, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 43))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 43, activation = 'relu',
                        kernel_initializer = 'random_uniform'))
classificador.add(Dropout(0.1))
classificador.add(Dense(units = 43, activation = 'relu',
                        kernel_initializer = 'random_uniform'))
classificador.add(Dense(units = 43, activation = 'relu',
                        kernel_initializer = 'random_uniform'))

# Definindo camada de saida
classificador.add(Dense(units = 1, activation = 'sigmoid'))

# Definindo o Otimizador
otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)

# Compilando a Rede Neural
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

# Early Stop e RLR
es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 100, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.25, patience = 50, verbose = 1)

# Treinamento da Rede Neural
classificador.fit(previsores_treinamento, classe_treinamento, callbacks=[es, rlr],
                  batch_size = 55, epochs = 2000)

# Efetuando as previsoes
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz  = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste)