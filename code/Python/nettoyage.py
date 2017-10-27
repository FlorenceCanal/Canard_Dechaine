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
   
if __name__ == '__main__':
	chemin = "./data"
	extend = "csv"
	sep = ";"
	data = import_data(chemin,extend,sep)
	data_NA = sep_NA(data)
	data_NA = data_NA[~(data_NA.values[:,-1] > 0.30)]
	del data_NA["count_NaN"]
	data_final = data_NA.replace({',': '.'}, regex=True)
	data_final = data_final.convert_objects(convert_numeric=True)
	data_final["ecart"] = pd.Series(data_final['tH2_obs']-data_final['tH2'], index=data_final.index)

	data_final.to_csv('final_train.csv', sep=';', index=False)

	#df.to_csv('final_train.csv', sep=';', index=False)




