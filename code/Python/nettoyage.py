# -*- coding: utf-8 -*-
"""
Created on Mon Apr 3 2017

@author: BRIENS, CANAL, CHAOUNI, GODE, GOURMELON
"""

import glob 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def import_data(path, extend, sep):
   '''to import all data into one dataframe
   param : path -> path file
           extend -> file extension
           sep -> variable separator
   exit : One dataframe wich contain all data'''
   filenames = glob.glob(path + '/*.' + extend)
   dfs = []
   for filename in filenames:
       dfs.append(pd.read_csv(filename, sep = sep))
   # Concatenate
   big_frame = pd.concat(dfs, ignore_index=True)
   return big_frame


def sep_NA(df):
    '''pour isoler les lignes contenant des valeurs manquantes
    df -> dataframe
    return NA dataframe'''
    nans = lambda df: df[df.isnull().any(axis=1)]
    dfNA = nans(df)
    dfNA["count_NaN"] = (dfNA.apply(lambda x: sum(x.isnull().values), axis = 1))/(len(dfNA.columns))
    return dfNA
 
 
def centrer_reduire(data):
	predictors = ['tH2_obs', 'capeinsSOL0', 'ciwcH20', 'clwcH20','ddH10_rose4', 'ffH10', 'flir1SOL0', 'fllat1SOL0', 'flsen1SOL0','flvis1SOL0', 'hcoulimSOL0', 'huH2', 'iwcSOL0', 'nbSOL0_HMoy', 'nH20','ntSOL0_HMoy', 'pMER0', 'rr1SOL0', 'rrH20', 'tH2', 'tH2_VGrad_2.100','tH2_XGrad', 'tH2_YGrad', 'tpwHPA850', 'ux1H10', 'vapcSOL0', 'vx1H10']
	moy = data[predictors].apply(np.nanmean)
	var = data[predictors].apply(np.nanvar)
	data[predictors] = (data[predictors]-moy)/var
	data.fillna(0, inplace=True)
	return data
	
def arranger(data):
	data_final = data.replace({',': '.'}, regex=True)
	data_final = data_final.convert_objects(convert_numeric=True)
	return data_final
   
if __name__ == '__main__':
	chemin = "./data"
	extend = "csv"
	sep = ";"
	data = import_data(chemin,extend,sep)
	data = arranger(data)
	data_NA = sep_NA(data)
	data_NA = data_NA[~(data_NA.values[:,-1] > 0.10)]
	del data_NA["count_NaN"]
	data_NA["tH2_standard"] = pd.Series(data_NA['tH2'], index=data_NA.index)
	data_NA["ecart"] = pd.Series(data_NA['tH2_obs']-data_NA['tH2'], index=data_NA.index)
	data_CR = centrer_reduire(data_NA)
	#data_final = arranger(data_CR)
	data_final = data_CR

	data_final.to_csv('final_train2.csv', sep=';', index=False)

	#df.to_csv('final_train.csv', sep=';', index=False)





