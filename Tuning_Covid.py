# -*- coding: utf-8 -*-
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
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

# Separando dados de treinamento e testes
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.10)

# Improving the ANN
# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
def build_classifier(optimizer):
    classificador = Sequential()
    classificador.add(Dense(units = 43, activation = 'relu', 
                            kernel_initializer = 'random_uniform', input_dim = 43))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 43, activation = 'relu',
                            kernel_initializer = 'random_uniform'))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                          metrics = ['binary_accuracy'])
    return classificador
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [15, 35, 55],
              'epochs': [500],
              'optimizer': ['adam']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(previsores_teste, classe_teste)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_