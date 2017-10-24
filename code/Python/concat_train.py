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
