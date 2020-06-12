#Com base nos resultados de exames laboratoriais comumente coletados para um 
#caso suspeito de COVID-19 durante uma visita à sala de emergência, 
#seria possível prever o resultado do teste para SARS-Cov-2 ?
# -*- coding: utf-8 -*-
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
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
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(previsores[:, 0:45])
previsores = imputer.transform(previsores[:, 0:45])

# Efetuando o escalonamento (normalizacao)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

def criarRede():
    # Definindo Neuronios e camadas ocultas
    classificador = Sequential()
    classificador.add(Dense(units = 43, activation = 'relu', 
                            kernel_initializer = 'random_uniform', input_dim = 43))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 43, activation = 'relu', 
                            kernel_initializer = 'random_uniform'))
    
    # Definindo camada de saida
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    
    # Definindo o Otimizador
    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    #otimizador = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-07)
    
    # Compilando a Rede Neural
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                          metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 500,
                                batch_size = 55)

resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()
print("Accuracy: %0.2f (+/- %0.2f)" % (resultados.mean(), resultados.std()))