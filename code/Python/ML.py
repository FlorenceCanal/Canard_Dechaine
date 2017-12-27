# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#import math
#import seaborn as sns
from tqdm import tqdm
from time import time
import datetime
from math import sqrt
#from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, explained_variance_score, pairwise_distances, calinski_harabaz_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, Imputer
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
#from sklearn.pipeline import Pipeline
import xgboost as xgb
from xgboost import plot_importance
#from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD
#from keras.wrappers.scikit_learn import KerasClassifier

t0 = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')



#########################
#                       #
#         SAKHIR        #
#   WEATHER CHALLENGE   #
#                       #
#########################

# Cf. https://gigadom.wordpress.com/2017/11/07/practical-machine-learning-with-r-and-python-part-5/





#----------
# I. Load data
#----------

Train=[]
for i in range(1,37):
    Train.append(pd.read_csv('C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/train/train_{}.csv'.format(i),sep=';',decimal=","))

train = pd.concat(Train)
train['test'] = 0

X_test = pd.read_csv('C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test.csv',sep=';',decimal=",")
X_test['test'] = 1

df = pd.concat([train,X_test])

df['flvis1SOL0'] = df['flvis1SOL0'].apply(lambda x: float(str(x).replace(',','.')))
df.head()

#plt.figure(figsize=(15,10))
#sns.pairplot(df[0:200], hue='insee')
#plt.show()





#----------
# II. Add data
#----------

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





#----------
# III. Deal with missing values
#----------

df.isnull().sum()
numeric_features = ['tH2','capeinsSOL0','ciwcH20','clwcH20','nH20','pMER0','rr1SOL0','rrH20','tpwHPA850','ux1H10','vapcSOL0','vx1H10','ddH10_rose4','ffH10','flir1SOL0','fllat1SOL0','flsen1SOL0','flvis1SOL0','hcoulimSOL0','huH2','iwcSOL0','nbSOL0_HMoy','ntSOL0_HMoy','tH2_VGrad_2.100','tH2_XGrad','tH2_YGrad']
qualitative_features = ['insee', 'mois']
imput = Imputer(missing_values='NaN', strategy='mean', axis=0)
imput.fit(df[numeric_features])
df[numeric_features] = imput.transform(df[numeric_features])
#df.isnull().sum()

numeric_features.remove('tH2')

scale_num = StandardScaler()
scale_num.fit(df[numeric_features])
df[numeric_features] = scale_num.transform(df[numeric_features])

df.head()





#----------
# IV. Split train & test & prepare for modeling
#----------

X_test = df[df['test']==1].drop('test',1)
train = df[df['test']==0].drop('test',1)

X_test_date = X_test['date']
X_test_ech = X_test['ech']
X_test_insee = X_test['insee']
X_test = X_test.drop('date',1).drop('insee',1).drop('tH2_obs',1)

train['ecart'] = train['tH2_obs'] - train['tH2']
train['ecart'] = np.nan_to_num(train['ecart'])
print('Size of train : '+str(len(train)))

#tH2_obs=train['tH2_obs']
train = train.drop('date',1).drop('tH2_obs',1).drop('insee',1)





#----------
# V. Delete outliers from train set (isolation forest)
#----------

#clf = IsolationForest(max_samples = 100, random_state = 3)
#clf.fit(train)
#y_noano = clf.predict(train)
#y_noano = pd.DataFrame(y_noano, columns = ['Top'])
#y_noano[y_noano['Top'] == 1].index.values
#
#train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
#train.reset_index(drop = True, inplace = True)
#print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
#print("Number of rows without outliers:", train.shape[0])





#----------
# VI. Split train & validation set
#----------

y_train = train['ecart']
X_train = train.drop('ecart',1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
X_train.head()

print('Ready to start ML part !')





#----------
# VII. Random Forest
#----------

# Cf. http://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/
# Cf. http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# Cf. https://machinelearningmastery.com/implement-random-forest-scratch-python/

# Cf. https://www.kaggle.com/hadend/tuning-random-forest-parameters

model_rf = RandomForestRegressor(n_estimators = 500,
                                 max_depth = 15,
                                 min_samples_split = 40,
                                 min_samples_leaf = 20,
                                 max_leaf_nodes = 1750,
                                 bootstrap = True, oob_score = True,
                                 random_state=321, n_jobs = 4, verbose = 1)

#rmse = sqrt(-np.mean(cross_val_score(model_rf, X_train, y_train, cv=5, scoring = 'neg_mean_squared_error')))
#print('RMSE : ',rmse)

model_rf.fit(X_train, y_train)

def plot_imp_rf(model_rf, X):
    importances = model_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model_rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    names = X.columns[indices]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print(str(f+1) + '. feature ' + str(names[f]) + ' (' + str(importances[indices[f]]) + ')')
    # Plot the feature importances of the forest
    plt.figure(figsize=(15,10))
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), names, rotation=80)
    plt.xlim([-1, X.shape[1]])
    plt.show()

plot_imp_rf(model_rf, X_train)
#oob_error = 1 - model_rf.oob_score_

def verif_valid(model, X_val, y_val):
    reality = y_val
    predictions = model.predict(X_val)
    plt.hist(predictions)
    plt.title('Histogram of Predictions')
    plt.show()
    plt.plot(predictions, reality, 'ro')
    plt.xlabel('Predictions')
    plt.ylabel('Reality')
    plt.title('Predictions x Reality')
    plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
    plt.show()
    print('Explained var (total) = ' + str(explained_variance_score(X_val['tH2']+predictions, X_val['tH2']+reality)))
    print('Explained var (predictions) = ' + str(explained_variance_score(reality, predictions)))
    print('RMSE = ' + str(sqrt(mean_squared_error(reality, predictions))))

verif_valid(model_rf, X_val, y_val)

print('ML part. I : Random Forest, done !')





#----------
# VIII. XGBoost
#----------

model_xgb = xgb.XGBRegressor(base_score=0.5,
                             subsample=0.8,
                             max_delta_step=2,
                             max_depth=7,
                             min_child_weight=3,
                             n_estimators=580,
                             colsample_bytree=0.85,
                             gamma=0,
                             seed=321, silent=0)

#rmse = sqrt(-np.mean(cross_val_score(model_xgb, X_train, y_train, cv=5, scoring = 'neg_mean_squared_error')))
#print('RMSE : ',rmse)

model_xgb.fit(X_train,y_train)

plot_importance(model_xgb)
plt.figure(figsize=(15,15))
plt.show()

verif_valid(model_xgb, X_val, y_val)

print('ML part. II : XGBoost, done !')





#----------
# IX. ADABOOST
#----------

model_adab = AdaBoostRegressor(base_estimator = RandomForestRegressor(),
                               n_estimators = 300,
                               learning_rate = 0.28,
                               loss = 'linear',
                               random_state = 321)

#rmse = sqrt(-np.mean(cross_val_score(model_adab, X_train, y_train, cv=5, scoring = 'neg_mean_squared_error')))
#print('RMSE : ',rmse)

model_adab.fit(X_train,y_train)

verif_valid(model_adab, X_val, y_val)

print('ML part. III : Adaboost, done !')





#----------
# X. GRADIENT BOOSTING
#----------

model_gradb = GradientBoostingRegressor(learning_rate = 0.6,
                                        n_estimators = 500,
                                        subsample = 0.9,
                                        min_samples_leaf = 40,
                                        random_state = 321, verbose = 0)

#rmse = sqrt(-np.mean(cross_val_score(model_gradb, X_train, y_train, cv=5, scoring = 'neg_mean_squared_error')))
#print('RMSE : ',rmse)

model_gradb.fit(X_train,y_train)

verif_valid(model_gradb, X_val, y_val)

print('ML part. IV : Gradient Boosting, done !')





#----------
# XI. STOCHASTIC GRADIENT DESCENT
#----------

from sklearn.linear_model import SGDRegressor

model_sgdr = SGDRegressor(
                          random_state = 321, verbose=0)

#rmse = sqrt(-np.mean(cross_val_score(model_sgdr, X_train, y_train, cv=5, scoring = 'neg_mean_squared_error')))
#print('RMSE : ',rmse)

model_sgdr.fit(X_train,y_train)

verif_valid(model_sgdr, X_val, y_val)

print('ML part. V : Stochastic Gradient Descent, done !')





#----------
# XII. SUPPORT VECTOR REGRESSION
#----------

from sklearn.svm import SVR

model_svr = SVR(
                verbose=0)

#rmse = sqrt(-np.mean(cross_val_score(model_sgdr, X_train, y_train, cv=5, scoring = 'neg_mean_squared_error')))
#print('RMSE : ',rmse)

model_svr.fit(X_train,y_train)

verif_valid(model_svr, X_val, y_val)

print('ML part. VI : SVR, done !')





#----------
# XIII. Neural Network
#----------

X_train_plus = X_train
X_val_plus = X_val

#rf = model_rf.predict(X_train)
#xgb = model_xgb.predict(X_train)
#adab = model_adab.predict(X_train)
#gradb = model_gradb.predict(X_train)
#
#X_train_plus['rf'] = rf
#X_train_plus['xgb'] = xgb
#X_train_plus['adab'] = adab
#X_train_plus['gradb'] = gradb
#X_val_plus = X_val
#
#rf = model_rf.predict(X_val)
#xgb = model_xgb.predict(X_val)
#adab = model_adab.predict(X_val)
#gradb = model_gradb.predict(X_val)
#
#X_val_plus['rf'] = rf
#X_val_plus['xgb'] = xgb
#X_val_plus['adab'] = adab
#X_val_plus['gradb'] = gradb

scale_x = StandardScaler()
scale_x.fit(X_train_plus)
X_train_scale = pd.DataFrame(scale_x.transform(X_train_plus))

def nn_model():
    seed = 321
    np.random.seed(seed)
    rmsprop = RMSprop(lr=0.0001)
    #sgd=SGD(lr=0.1)
    # Create model
    #kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    #for train, test in kfold.split(X, y):
    model_nn = Sequential()
    model_nn.add(Dense(250, input_dim=43, activation='relu', kernel_initializer='normal')) #64
    model_nn.add(Dropout(0.3)) #0.3
    model_nn.add(Dense(300, activation='relu', kernel_initializer='normal')) #128
    model_nn.add(Dropout(0.3)) #0.3
    model_nn.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model_nn.compile(optimizer= rmsprop, loss='mean_squared_error')# optimizer = ['sgd', 'adam']
    return model_nn

model_nn = nn_model()
model_nn.summary()
model_nn.fit(np.array(X_train_scale), np.array(y_train), epochs=350, shuffle=True, verbose=1)

X_val_scale = pd.DataFrame(scale_x.transform(X_val_plus))
verif_valid(model_nn, np.array(X_val_scale), y_val)

print('ML part. VII : Neural Network, done !')





#----------
# XIV. Stacking model
#----------

rf = model_rf.predict(X_train)
xgb = model_xgb.predict(X_train)
adab = model_adab.predict(X_train)
gradb = model_gradb.predict(X_train)
sgdr = model_sgdr.predict(X_train)
svr = model_svr.predict(X_train)
nn = model_nn.predict(np.array(X_train_scale))
X_train_stack = pd.concat([rf, xgb, adab, gradb, sgdr, svr, nn], axis=1)

rf = model_rf.predict(X_val)
xgb = model_xgb.predict(X_val)
adab = model_adab.predict(X_val)
gradb = model_gradb.predict(X_val)
sgdr = model_sgdr.predict(X_val)
svr = model_svr.predict(X_val)
nn = model_nn.predict(np.array(X_val_scale))
X_val_stack = pd.concat([rf, xgb, adab, gradb, sgdr, svr, nn], axis=1)

def stacking_nn_model():
    seed = 321
    np.random.seed(seed)
    rmsprop = RMSprop(lr=0.0001)
    #sgd=SGD(lr=0.1)
    # Create model
    #kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    #for train, test in kfold.split(X, y):
    model_nn = Sequential()
    model_nn.add(Dense(24, input_dim=6, activation='relu', kernel_initializer='normal'))
    model_nn.add(Dropout(0.2))
    model_nn.add(Dense(32, activation='relu', kernel_initializer='normal'))
    model_nn.add(Dropout(0.2))
    model_nn.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model_nn.compile(optimizer= rmsprop, loss='mean_squared_error')# optimizer = ['sgd', 'adam']
    return model_nn

model_stack_nn = stacking_nn_model()
model_stack_nn.summary()
model_stack_nn.fit(np.array(X_train_stack), np.array(y_train), epochs=350, shuffle=True, verbose=1)

print('ML part. VIII : Stacking Model (Neural Network), done !')





#----------
# XV. Predictions on the test set
#----------

model = model_rf
# [model_rf, model_xgb, model_adab, model_gradb, model_sgdr, model_svr, model_nn, model_stack_nn]

#X_test_scale = pd.DataFrame(scale_x.transform(X_test))

predictions = model.predict(X_test)
plt.hist(predictions)
plt.show()

predictions = X_test['tH2'] + predictions
plt.hist(predictions)
plt.show()

res = pd.DataFrame({
        'date':X_test_date,
        'insee':X_test_insee,
        'ech':X_test_ech,
        'tH2_obs':predictions},
        columns=['date','insee','ech','tH2_obs'])

res.to_csv('new_sub.csv', sep=";", index=False)



print('THE END : submission ready !')

tf = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
print('time end : ',tf)
print('time enlapsed : ',int((time()-t0)/60),'min')






































