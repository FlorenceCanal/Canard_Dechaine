# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import datetime
from operator import itemgetter
import matplotlib.pyplot as plt
#from keras.utils import np_utils
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.optimizers import RMSprop, SGD
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, explained_variance_score, pairwise_distances, calinski_harabaz_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, Imputer
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.pipeline import Pipeline
from math import sqrt
import xgboost as xgb
from xgboost import plot_importance



#########################
#                       #
#         SAKHIR        #
#   WEATHER CHALLENGE   #
#                       #
#########################



Train=[]
for i in range(1,37):
    Train.append(pd.read_csv('C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/train/train_{}.csv'.format(i),sep=';',decimal=","))

train = pd.concat(Train)
train['test'] = 0

X_test = pd.read_csv('C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test.csv',sep=';',decimal=",")
X_test['test'] = 1

df = pd.concat([train,X_test])

df['flvis1SOL0'] = df['flvis1SOL0'].apply(lambda x: float(str(x).replace(',','.')))
numeric_features = ['tH2','capeinsSOL0','ciwcH20','clwcH20','nH20','pMER0','rr1SOL0','rrH20','tpwHPA850','ux1H10','vapcSOL0','vx1H10','ddH10_rose4','ffH10','flir1SOL0','fllat1SOL0','flsen1SOL0','flvis1SOL0','hcoulimSOL0','huH2','iwcSOL0','nbSOL0_HMoy','ntSOL0_HMoy','tH2_VGrad_2.100','tH2_XGrad','tH2_YGrad']
qualitative_features = ['insee', 'mois']
df.head()

df['seasons'] = df['mois'].map({'janvier': 'hiver', 'février': 'hiver', 'mars':'printemps', 'avril':'printemps','mai':'printemps','juin':'été','juillet':'été',"août":"été","septembre":"automne","octobre":"automne","novembre":"automne","décembre":"hiver"})
df['year'] = df['date'].apply(lambda x: x[:4])

df['city'] = df['insee'].map({6088001:'Nice', 31069001: 'Toulouse', 33281001:'Bordeaux', 35281001:'Rennes',59343001:'Lille',67124001:'Strasbourg',75114001:'Paris'})
df['lat'] = df['city'].map({'Nice':43.66510,'Toulouse':43.63799,'Bordeaux':44.83137,'Rennes':48.06560,'Lille':50.57178,'Strasbourg':48.53943,'Paris':48.82231})
df['lon'] = df['city'].map({'Nice':7.2133184,'Toulouse':1.3719825,'Bordeaux':-0.6920608,'Rennes':-1.7232451,'Lille':3.1060870,'Strasbourg':7.6280226,'Paris':2.3378563})
df['fleuve'] = df['city'].map({'Nice':0,'Toulouse':1,'Bordeaux':1,'Rennes':0,'Lille':0,'Strasbourg':1,'Paris':1})
df['distmer'] = df['city'].map({'Nice':0,'Toulouse':150,'Bordeaux':65,'Rennes':70,'Lille':80,'Strasbourg':540,'Paris':200})

# Cf. http://www.linternaute.com/voyage/climat/comparaison/toulouse/rennes/2015/ville-31555/ville-35238
df['ensoleillement16'] = df['city'].map({'Nice':2771,'Toulouse':2100,'Bordeaux':2022,'Rennes':1679,'Lille':1653,'Strasbourg':1625,'Paris':1726})
df['ensoleillement15'] = df['city'].map({'Nice':2917,'Toulouse':2238,'Bordeaux':2151,'Rennes':1851,'Lille':1739,'Strasbourg':1886,'Paris':1953})
df['precipitations16'] = df['city'].map({'Nice':484,'Toulouse':577,'Bordeaux':922,'Rennes':625,'Lille':836,'Strasbourg':730,'Paris':658})
df['precipitations15'] = df['city'].map({'Nice':669,'Toulouse':511,'Bordeaux':592,'Rennes':596,'Lille':786,'Strasbourg':475,'Paris':507})

# get_dummies() allow factorial features to be part of the method
df = pd.concat([df, pd.get_dummies(df[['seasons','year']])], axis=1)
df = df.drop('seasons',1)
df = df.drop('mois',1)
df = df.drop('year',1)
df = df.drop('city',1)

df.head()
print('Size of df : '+str(len(df)))



df.isnull().sum()

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(df[numeric_features])
df[numeric_features] = imp.transform(df[numeric_features])

df.isnull().sum()

numeric_features.remove('tH2')

scaler = StandardScaler()
scaler.fit(df[numeric_features])
df[numeric_features] = scaler.transform(df[numeric_features])
df.head()

X_test = df[df["test"]==1].drop('test',1)
train = df[df["test"]==0].drop('test',1)

X_test_date = X_test['date']
X_test_ech = X_test['ech']
X_test_insee = X_test['insee']
X_test = X_test.drop('date',1).drop('insee',1).drop('tH2_obs',1)

train['delta'] = train['tH2_obs'] - train['tH2']
train['delta'] = np.nan_to_num(train['delta'])
print('Size of train : '+str(len(train)))

#tH2_obs=train['tH2_obs']
train = train.drop('date',1).drop('tH2_obs',1).drop('insee',1)

#We will try to predict the delta between the model already created and observations
y_train = train['delta']
X_train = train.drop('delta',1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
X_train.head()





#----------
# I. RANDOM FOREST
#----------


#----------
# Grid search
#----------

param_grid1 = {"n_estimators": np.arange(100, 301, 100),
               "max_depth": np.arange(1, 12, 5),
               "min_samples_split": np.arange(50,151,50),
               "min_samples_leaf": np.arange(1,102,50),
               "max_leaf_nodes": np.arange(2,102,50)
               }

t0 = time()

clf = RandomForestRegressor(random_state=321)
grid_search = GridSearchCV(clf, param_grid=param_grid1,
                           n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print('Best score for data1:', grid_search.best_score_)

grid_results = grid_search.cv_results_
#grid_results.keys()
best_mse = min(-grid_results['mean_train_score'])
print('Best MSE after gridsearch:', best_mse)
print(grid_search.best_params_)

ts = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
print('time end : ',ts)
print('time enlapsed : ',int((time()-t0)/60),'min')

def report(grid_scores, n_top):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f})".format(
              -score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

report(grid_search.grid_scores_, 5)

#print(grid_search.best_params_)
#{'max_depth': 11, 'max_leaf_nodes': 52, 'min_samples_leaf': 1, 'min_samples_split': 150, 'n_estimators': 300}



#----------
# Search for best param
#----------

n_estimators = np.arange(250, 751, 100)
#[1.5689636091118881, 1.5686167605861898, 1.5687449002205123, 1.5686896028263184, 1.5688099763237964, 1.568829569336716]

max_features = ["auto", "sqrt", "log2"]
#[1.570998265346137, 1.6103578571456403, 1.6258394925228259]

max_depth = np.arange(10, df.shape[1], 5)
#[1.5688035989386757, 1.5687824410813596, 1.5687824410813596, 1.5687824410813596, 1.5687824410813596, 1.5687824410813596, 1.5687824410813596, 1.5687824410813596]

min_samples_split = np.arange(50,151,25)
#[1.5687247967758413, 1.5687247967758413, 1.5687824410813596, 1.5688787900495971, 1.5689699196535698]

min_samples_leaf = np.arange(20,71,10)
#[1.5683445954971535, 1.5685175651838483, 1.5687269428149131, 1.5688447604801867, 1.5695678542421843, 1.5706387114833302]

max_leaf_nodes = np.arange(30,101,10)
#[1.686001814585159, 1.640558158208052, 1.6060417359219037, 1.579810703293945, 1.5589583270308549, 1.5419379675473159, 1.5277142454972323, 1.5154583649015128]


cvError=[]
evol = np.arange(1000,5001,250)
nb = len(evol)
with tqdm(total=nb) as pbar:
    for param in evol:
        regr = RandomForestRegressor(n_estimators = 350,
                                     max_depth = 15,
                                     min_samples_split = 40,
                                     min_samples_leaf = 20,
                                     max_leaf_nodes = 1750,
                                     bootstrap = True, oob_score = True,
                                     random_state=321, n_jobs = 4, verbose = 0)
        rmse = sqrt(-np.mean(cross_val_score(regr, X_train, y_train, cv=3, scoring = 'neg_mean_squared_error')))
        print('RMSE : ',rmse)
        cvError.append(rmse)
        pbar.update()
plt.plot(evol, cvError)
plt.title("Random Forest - evol param")
plt.show()

regr = RandomForestRegressor(n_estimators = 350,
                                     max_depth = 15,
                                     min_samples_split = 40,
                                     min_samples_leaf = 20,
                                     max_leaf_nodes = 1750,
                                     bootstrap = True, oob_score = True,
                                     random_state=321, n_jobs = 4, verbose = 0)

regr.fit(X_train, y_train)
predict = regr.predict(X_val)
reality = y_val

plt.hist(predict)
plt.show()
#predict[abs(predict) > 10] = 0
plt.plot(predict, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predict))
# RMSE = 1.1314105964906458
print('RMSE = ' + str(rmse))




#----------
# II. ADABOOST
#----------

evol = ['linear', 'square', 'exponential']

cvError=[]
nb = len(evol)
with tqdm(total=nb) as pbar:
    for param in evol:
        regr = AdaBoostRegressor(base_estimator = RandomForestRegressor(),
                                 n_estimators = 300,
                                 learning_rate = 0.28,
                                 loss = 'linear',
                                 random_state = 321)
        rmse = sqrt(-np.mean(cross_val_score(regr, X_train, y_train, cv=5, scoring = 'neg_mean_squared_error')))
        print('RMSE : ',rmse)
        cvError.append(rmse)
        pbar.update()

plt.plot(range(len(evol)), cvError)
plt.title("Adaboost - learning_rate")
plt.show()

#RMSE = 1.0436002013894718



#----------
# III. GRADIENT BOOSTING
#----------

#from sklearn.datasets import load_boston
#
#df = pd.DataFrame(load_boston().data)
#df.columns = ['crimeRate','zone','indus','charles','nox','rooms','age','distances','highways','tax','teacherRatio','status','medianValue']
#X = df[['crimeRate','zone', 'indus','charles','nox','rooms', 'age',
#        'distances','highways','tax','teacherRatio','status']]
#y = df['medianValue']

cvError=[]
evol = np.arange(100, 1501, 100)
nb = len(evol)
with tqdm(total=nb) as pbar:
    for param in evol:
        regr = GradientBoostingRegressor(learning_rate = 0.6,
                                         n_estimators = 500,
                                         subsample = 0.9,
                                         min_samples_leaf = param,
                                         random_state = 321, verbose = 0)
        rmse = sqrt(-np.mean(cross_val_score(regr, X_train, y_train, cv=3, scoring = 'neg_mean_squared_error')))
        print('RMSE : ',rmse)
        cvError.append(rmse)
        pbar.update()

plt.plot(evol, cvError)
plt.title("Gradient Boosting - param evol")
plt.show()

#subsample :
#evol = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#cvError = [1.6828893312163768,1.2515550685881314,1.2187081498582804,1.1903681738687104,1.1725683464470391,1.1663259610664594,1.160182406692401,1.1501544310895295,1.1563619766601851]

#min_samples_leaf :
#evol = [10,20,30,40,50,60,70,80,90,100]
#cvError = [1.1525818413103583, 1.1507860226217055, 1.1473798630381735, 1.1446122474145521, 1.1438359833464806, 1.1444849198751834, 1.1404409339903856, 1.1420589121180544, 1.141437872365891, 1.1392466590149521]



#GradientBoostingRegressor(loss='ls', # ['ls', 'lad', 'huber', 'quantile']
#                          max_depth=3,
#                          alpha=0.9, # only if loss='huber' or loss='quantile'
#                          random_state=321, verbose=0)





#----------
# X. STOCHASTIC GRADIENT DESCENT
#----------

from sklearn.linear_model import SGDRegressor

cvError=[]
evol = ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
nb = len(evol)
with tqdm(total=nb) as pbar:
    for param in evol:
        regr = SGDRegressor(loss = param,
                            verbose = 0)
        rmse = sqrt(-np.mean(cross_val_score(regr, X_train, y_train, cv=3, scoring = 'neg_mean_squared_error')))
        print('RMSE : ',rmse)
        cvError.append(rmse)
        pbar.update()

plt.plot(range(len(evol)), cvError)
plt.title("SGD_reg - param evol")
plt.show()


#model_sgdr = SGDRegressor(loss='squared_loss', #['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
#                          penalty='l2', #['l2','l1','elasticnet']
#                          alpha=0.0001,
#                          l1_ratio=0.15,
#                          fit_intercept=True,
#                          max_iter=None,
#                          tol=None,
#                          epsilon=0.1,
#                          learning_rate='invscaling',
#                          eta0=0.01,
#                          power_t=0.25,
#                          random_state = 321, verbose=0)





#----------
# IV. SUPPORT VECTOR REGRESSION
#----------

from sklearn.svm import SVR

cvError1=[]
evol1 = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
nb = len(evol1)
with tqdm(total=nb) as pbar:
    for param in evol1:
        regr = SVR(kernel=param,
                   verbose = 0)
        rmse = sqrt(-np.mean(cross_val_score(regr, X_train, y_train, cv=3, scoring = 'neg_mean_squared_error')))
        print('RMSE : ',rmse)
        cvError1.append(rmse)
        pbar.update()

plt.plot(range(len(evol1)), cvError1)
plt.title("SVR - param evol")
plt.show()

#best_param = ?? (minimize cvError MSE)

cvError2=[]
evol2 = np.arange()
nb = len(evol2)
with tqdm(total=nb) as pbar:
    for param in evol2:
        regr = SVR(kernel=best_param,
                   C=param,
                   verbose = 0)
        rmse = sqrt(-np.mean(cross_val_score(regr, X_train, y_train, cv=3, scoring = 'neg_mean_squared_error')))
        print('RMSE : ',rmse)
        cvError2.append(rmse)
        pbar.update()

plt.plot(evol2, cvError2)
plt.title("SVR - param evol")
plt.show()

# ...
# And keep going till param are choosed & error is minimized


#all params =
#model_svr = SVR(kernel='rbf', #['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
#            C=1.0,
#            gamma='auto',
#            degree=3,
#            coef0=0.0,
#            tol=0.001,
#            epsilon=0.1,
#            shrinking=True,
#            cache_size=200,
#            verbose=0)















'''

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, MICE

# X is the complete data matrix
# X_incomplete has the same values as X except a subset have been replace with NaN


from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer, StandardScaler


rng = np.random.RandomState(0)
dataset = load_boston()
X_full, y_full = dataset.data, dataset.target

X = pd.concat([pd.DataFrame(X_full), pd.DataFrame(y_full)], axis=1)

n_samples = X_full.shape[0]
n_features = X_full.shape[1]

X.isnull().sum()

X_incomplete = np.array(X)

missing_rate = 0.15
nb_missing = int(np.floor(missing_rate*n_samples*n_features))
for nb in range(nb_missing):
    x_na = np.random.randint(0,n_samples)
    nb_y_na = np.random.randint(0,n_features/3)
    for i in range(nb_y_na):
        y_na = np.random.randint(0,n_features)
        #print(x_na,y_na)
        X_incomplete[x_na,y_na] = np.nan

X_incomplete = pd.concat([pd.DataFrame(X_incomplete), pd.DataFrame(y_full)], axis=1)
X_incomplete.isnull().sum()


# Use 3 nearest rows which have a feature to fill in each row's missing features
X_filled_knn = pd.DataFrame(KNN(k=3).complete(X_incomplete))

X_filled_mice = MICE(impute_type='pmm').complete(X_incomplete)

X_filled_nnm = NuclearNormMinimization().complete(X_incomplete)


X_filled_softimpute = SoftImpute().complete(X_incomplete_normalized)

knn_mse = ((X_filled_knn - X) ** 2).mean()
print(knn_mse)
#print("Nuclear norm minimization MSE: %f" % knn_mse)
nnm_mse = ((X_filled_nnm - X) ** 2).mean()
print(nnm_mse)
#print("Nuclear norm minimization MSE: %f" % nnm_mse)
softImpute_mse = ((X_filled_softimpute - X) ** 2).mean()
print(softImpute_mse)
#print("SoftImpute MSE: %f" % softImpute_mse)



'''



















