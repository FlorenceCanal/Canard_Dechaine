import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from locale import *

#from pandas.plotting import scatter_matrix

def import_data(path, extend, sep):
   '''to import all data into one dataframe
   param : path -> path file
           extend -> file extension
           sep -> variable separator
   exit : One dataframe wich contain all data'''
   filenames = glob.glob(path + '/*.' + extend)
   dfs = []
   for filename in filenames:
       dfs.append(pd.read_csv(filename, sep=sep, index_col=0, parse_dates=[0], dayfirst=True))
   # Concatenate
   big_frame = pd.concat(dfs)#, ignore_index=True)
   big_frame = big_frame.replace({',': '.'}, regex=True)
   big_frame = big_frame.convert_objects(convert_numeric=True)
   return big_frame


data = import_data("/Users/florencecanal/Desktop/M2-SID/Apprentissage_donnees_massives/data_meteo/github/data", "csv", ";")
data_reg = data.copy()
# Ajout de la var ecart temperature
data_reg['ecart'] = pd.Series(data_reg["tH2_obs"]-data_reg["tH2"], index=data_reg.index)
print(data_reg['ecart'])
y=data_reg.corr(method='pearson')
print(y)
from sklearn import linear_model
# Modele par défaut : Regression linéaire
# Ecart de la température
modele = linear_model.LinearRegression()
