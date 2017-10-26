# -*- coding: utf-8 -*-

import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

#from pandas.plotting import scatter_matrix


meteo = pd.read_csv("/Users/florencecanal/Desktop/M2-SID/Apprentissage_donnees_massives/data_meteo/github/code/final_train.csv", sep=";")
meteo = meteo.replace({',': '.'}, regex=True)

meteo = meteo.convert_objects(convert_numeric=True)
meteo.fillna(0, inplace=True)

########################################
######### Regréssion Linéaire ##########
########################################

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import cross_validation


meteo['ecart'] = pd.Series(meteo['tH2_obs']-meteo['tH2'], index=meteo.index)


data_test = pd.read_csv("/Users/florencecanal/Desktop/M2-SID/Apprentissage_donnees_massives/data_meteo/test.csv", sep=";")
data_test = data_test.replace({',': '.'}, regex=True)
data_test = data_test.convert_objects(convert_numeric=True)

X_place = meteo.copy()
y = X_place['ecart']
X_place = X_place[['ddH10_rose4', 'flir1SOL0', 'fllat1SOL0', 'flsen1SOL0', 'hcoulimSOL0', 'huH2', 'tH2', 'tH2_VGrad_2.100', 'tH2_YGrad', 'tpwHPA850', 'vapcSOL0', 'vx1H10']]

reg = linear_model.LinearRegression()
reg_fit = reg.fit(X_place, y)
X_sub = data_test.copy()
X_sub = X_sub[['ddH10_rose4', 'flir1SOL0', 'fllat1SOL0', 'flsen1SOL0', 'hcoulimSOL0', 'huH2', 'tH2', 'tH2_VGrad_2.100', 'tH2_YGrad', 'tpwHPA850', 'vapcSOL0', 'vx1H10']]
X_sub = X_sub.fillna(value=0)
reg = reg.predict(X_sub)

########################################
######### Fichier a soumettre ##########
########################################

X_filled = pd.read_csv("/Users/florencecanal/Desktop/M2-SID/Apprentissage_donnees_massives/data_meteo/test_answer_template.csv", sep=";") #, index_col=0) #, parse_dates=[0], dayfirst=True)
X_filled = X_filled.drop('tH2_obs', 1)
X_filled['tH2_obs'] = pd.Series(data_test['tH2']+reg, index=data_test.index)

X_filled['tH2_obs']= X_filled['tH2_obs'].astype(str)
X_filled = X_filled.replace({'\.': ','}, regex=True)
X_filled.to_csv("/Users/florencecanal/Desktop/M2-SID/Apprentissage_donnees_massives/data_meteo/test_filled.csv", sep=';', index = False)
