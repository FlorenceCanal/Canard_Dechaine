# coding: utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, Imputer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from math import sqrt



#########################
#                       #
#         SAKHIR        #
#   WEATHER CHALLENGE   #
#                       #
#########################

# ADD CATEGORICAL :
# - https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/


#----------
# I. Importation
#----------

df = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/final_train.csv", sep=";")
#df = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/train_imputNA.csv", sep=";")
df = df.replace({',': '.'}, regex=True)

df = df.convert_objects(convert_numeric=True)

df['insee'] = df['insee'].astype(object)
df['ddH10_rose4'] = df['ddH10_rose4'].astype(object)
df = df.drop('mois', 1)
df = df.drop('date', 1)

df = df.drop('capeinsSOL0', 1)
df = df.drop('ciwcH20', 1)
df = df.drop('clwcH20', 1)
df = df.drop('nH20', 1)

df_th2obs = df['tH2_obs']
df = df.drop('tH2_obs', 1)
df.info()

df = df.select_dtypes(exclude=['object'])

#df.fillna(0, inplace=True)
cols = df.columns
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)#most_frequent
imp.fit(df)
df = imp.transform(df)

df = pd.DataFrame(df)
df.columns = cols

print(df.isnull().sum())


col = list(df.columns)
col_X = list(df.columns)
col_X.remove('ecart')

X = np.matrix(df.drop('ecart', axis = 1))
Y = np.array(df.ecart).reshape((df.shape[0],1))

scale_x = StandardScaler()
scale_x.fit(X)

X = pd.DataFrame(scale_x.transform(X), columns = col_X)
X['ecart'] = df['ecart'].values



#----------
# II. Pre-processing
#----------


#----------
# III. Train/Test/Valid split
#----------

ids = np.random.rand(len(X)) < 0.85
train = X[ids]
test = X[~ids]


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


data = train.values
classix = data.shape[1]-1
X = data[:,0:classix]
y = data[:,classix]
ids = np.random.rand(data.shape[0]) < 0.8
((X_train,y_train),(X_test,y_test)) = ((X[ids],y[ids]),(X[~ids],y[~ids]))

test = test.values
classix = test.shape[1]-1
X_valid = test[:,0:classix]
y_valid = test[:,classix]



##----------
## IV. Sklearn Neural Network Regressor
##----------
#
#model = MLPRegressor(
#        hidden_layer_sizes=(32,64,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
#        learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, max_iter=1000, shuffle=True,
#        random_state=9, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#        early_stopping=False, validation_fraction=0.3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#
#model.fit(X_train,y_train)
#
#reality = y_test
#predict = model.predict(X_test)
#predict[abs(predict) > 8] = 0
#plt.plot(predict, reality, 'ro')
#plt.xlabel('Predictions')
#plt.ylabel('Reality')
#plt.title('Predictions x Reality on dataset Test')
#plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
#plt.show()
#rmse = sqrt(mean_squared_error(reality, predict))
## 1.06
#print('RMSE = ' + str(rmse))
#
#reality = y_valid
#predict = model.predict(X_valid)
#predict[abs(predict) > 8] = 0
#plt.plot(predict, reality, 'ro')
#plt.xlabel('Predictions')
#plt.ylabel('Reality')
#plt.title('Predictions x Reality on dataset Test')
#plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
#plt.show()
#rmse = sqrt(mean_squared_error(reality, predict))
## 1.05
#print('RMSE = ' + str(rmse))
#
#
#
## Submission
#print((X_train.shape[0] / df.shape[0]) *100)
#
#data_test = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test.csv", sep=";")
#data_test = data_test.replace({',': '.'}, regex=True)
#data_test = data_test.convert_objects(convert_numeric=True)
#data_test['insee'] = data_test['insee'].astype(object)
#data_test['ddH10_rose4'] = data_test['ddH10_rose4'].astype(object)
#data_test = data_test.drop('mois', 1)
#data_test = data_test.drop('date', 1)
#
#data_test = data_test.drop('capeinsSOL0', 1)
#data_test = data_test.drop('ciwcH20', 1)
#data_test = data_test.drop('clwcH20', 1)
#data_test = data_test.drop('nH20', 1)
#
#data_test = data_test.select_dtypes(exclude=['object'])
#data_test.fillna(0,inplace=True)
#
#th2 = np.array(data_test['tH2'], dtype=pd.Series)
#m_test = np.matrix(data_test)
#
##scale_test = StandardScaler()
##scale_test.fit(m_test)
##data_test = pd.DataFrame(scale_test.transform(m_test), columns = col_X)
#data_test = pd.DataFrame(scale_x.transform(m_test), columns = col_X)
#predict = model.predict(data_test.values)
#
##predict[abs(predict) > 8] = 0
#plt.hist(predict)
#plt.show()
#
#X_filled = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test_answer_template.csv", sep=";")
#X_filled = X_filled.drop('tH2_obs', 1)
##predict = predict[:,0]
#X_filled['tH2_obs'] = pd.Series(th2 + predict, index=data_test.index)
#plt.hist(X_filled['tH2_obs'])
#plt.show()
#
#X_filled['tH2_obs']= X_filled['tH2_obs'].astype(str)
#X_filled = X_filled.replace({'\.': ','}, regex=True)
##X_filled.to_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/sub_skMLPregressor1.csv", sep=';', index = False)





#----------
# IV. Keras Deep Learning Regressor (only numerical)
#----------

seed = 37
np.random.seed(seed)

rmsprop = RMSprop(lr=0.0001)
sgd=SGD(lr=0.1)

#kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#for train, test in kfold.split(X, y):
model = Sequential()
model.add(Dense(64, input_dim=22, activation='relu', kernel_initializer='normal')) #22 #64
model.add(Dropout(0.3)) #0.3
model.add(Dense(128, activation='relu', kernel_initializer='normal')) #128
model.add(Dropout(0.5)) #0.3
#model.add(Dense(150, activation='relu', kernel_initializer='normal')) #150
#model.add(Dropout(0.3)) #0.3
model.add(Dense(1, kernel_initializer='normal'))
model.compile(optimizer='rmsprop', loss='mean_squared_error')#, metrics=['mae']) # optimizer=sgd
#model.summary()

model.fit(X_train, y_train, epochs=350, shuffle=True, verbose=1)

factor = 1.3

reality = y_train
predict = model.predict(X_train)
plt.plot(predict*factor, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predict*factor))
# 1.00
print('RMSE = ' + str(rmse))

reality = y_test
predict = model.predict(X_test)
predict[abs(predict) > 8] = 0
plt.plot(predict*factor, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predict*factor))
# 1.00
print('RMSE = ' + str(rmse))

reality = y_valid
predict = model.predict(X_valid)
predict[abs(predict) > 8] = 0
plt.plot(predict*factor, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predict*factor))
# 1.00
print('RMSE = ' + str(rmse))


#loss = model.evaluate(X_train, y_train,  verbose=0)
#print('The RMSE on the training set is ',(loss)) #0.78
#
#loss = model.evaluate(X_test, y_test,  verbose=0)
#print('The RMSE on the testing set is ',(loss)) #1.05
#loss = model.evaluate(X_valid, y_valid,  verbose=0)
#print('The RMSE on the validation set is ',(loss)) #1.06






data_test = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test.csv", sep=";")
data_test = data_test.replace({',': '.'}, regex=True)
data_test = data_test.convert_objects(convert_numeric=True)
data_test['insee'] = data_test['insee'].astype(object)
data_test['ddH10_rose4'] = data_test['ddH10_rose4'].astype(object)
data_test = data_test.drop('mois', 1)
data_test = data_test.drop('date', 1)

data_test = data_test.drop('capeinsSOL0', 1)
data_test = data_test.drop('ciwcH20', 1)
data_test = data_test.drop('clwcH20', 1)
data_test = data_test.drop('nH20', 1)

data_test = data_test.select_dtypes(exclude=['object'])

cols = data_test.columns
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)#most_frequent
imp.fit(data_test)
data_test = imp.transform(data_test)
data_test = pd.DataFrame(data_test)
data_test.columns = cols

print(df.isnull().sum())
#data_test.fillna(0,inplace=True)

th2 = np.array(data_test['tH2'], dtype=pd.Series)
m_test = np.matrix(data_test)

#scale_test = StandardScaler()
#scale_test.fit(m_test)
#data_test = pd.DataFrame(scale_test.transform(m_test), columns = col_X)
data_test = pd.DataFrame(scale_x.transform(m_test), columns = col_X)
data_test = np.array(data_test)

predict = model.predict(data_test) * factor
plt.hist(predict)
plt.show()
#predict[abs(predict) > 8] = 0

X_filled = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test_answer_template.csv", sep=";")
X_filled = X_filled.drop('tH2_obs', 1)
predict = predict[:,0]
X_filled['tH2_obs'] = pd.Series(th2 + predict)
plt.hist(X_filled['tH2_obs'])
plt.show()

X_filled['tH2_obs']= X_filled['tH2_obs'].astype(str)
X_filled = X_filled.replace({'\.': ','}, regex=True)
#X_filled.to_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/sub_skMLPregressor1.csv", sep=';', index = False)









#----------
# IV. Keras Deep Learning Regressor & factors
#----------

df = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/final_train.csv", sep=";")
#df = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/train_imputNA.csv", sep=";")
df = df.replace({',': '.'}, regex=True)

df = df.convert_objects(convert_numeric=True)

df['insee'] = df['insee'].astype(object)
df['ddH10_rose4'] = df['ddH10_rose4'].astype(object)
df = df.drop('mois', 1)
df = df.drop('date', 1)

df = df.drop('capeinsSOL0', 1)
df = df.drop('ciwcH20', 1)
df = df.drop('clwcH20', 1)
df = df.drop('nH20', 1)

df_th2obs = df['tH2_obs']
df = df.drop('tH2_obs', 1)
df.info()


# Add categorical features
encoder = LabelEncoder()
encoder.fit(df['ddH10_rose4'])
vent = encoder.transform(df['ddH10_rose4'])
vent = np_utils.to_categorical(vent)
vent = pd.DataFrame(vent)
vent.columns = ['vent1', 'vent2', 'vent3', 'vent4']

lat = []
lon = []
with tqdm(total=df.shape[0]) as pbar:
    for each in df['insee']:
        if each == 6088001:
            lat.append(43.66510)
            lon.append(7.2133184)
        elif each == 31069001:
            lat.append(43.63799)
            lon.append(1.3719825)
        elif each == 33281001:
            lat.append(44.83137)
            lon.append(-0.6920608)
        elif each == 35281001:
            lat.append(48.06560)
            lon.append(-1.7232451)
        elif each == 59343001:
            lat.append(50.57178)
            lon.append(3.1060870)
        elif each == 67124001:
            lat.append(48.53943)
            lon.append(7.6280226)
        elif each == 75114001:
            lat.append(48.82231)
            lon.append(2.3378563)
        else:
            print("ERROR")
            break
        pbar.update()

print(len(lat) == len(lon) == len(df['insee']))

#"6088001" = "Nice"
#"31069001" = "Toulouse Blagnac"
#"33281001" = "Bordeaux Merignac"
#"35281001" = "Rennes"
#"59343001" = "Lille Lesquin"
#"67124001" = "Strasbourg Entzheim"
#"75114001" = "Paris-Montsouris"

#               villes        lon      lat
#1                Nice  7.2133184 43.66510
#2    Toulouse Blagnac  1.3719825 43.63799
#3   Bordeaux Merignac -0.6920608 44.83137
#4              Rennes -1.7232451 48.06560
#5       Lille Lesquin  3.1060870 50.57178
#6 Strasbourg Entzheim  7.6280226 48.53943
#7    Paris-Montsouris  2.3378563 48.82231

latlon = np.vstack((lat,lon)).T
latlon = pd.DataFrame(latlon)
latlon.columns = ['lat', 'lon']

df = pd.concat([df, latlon, vent], axis=1)

df = df.select_dtypes(exclude=['object'])
df.fillna(0, inplace=True)

col = list(df.columns)
col_X = list(df.columns)
col_X.remove('ecart')

X = np.matrix(df.drop('ecart', axis = 1))
Y = np.array(df.ecart).reshape((df.shape[0],1))

scale_x = StandardScaler()
scale_x.fit(X)

X = pd.DataFrame(scale_x.transform(X), columns = col_X)
X['ecart'] = df['ecart'].values

ids = np.random.rand(len(X)) < 0.85
train = X[ids]
test = X[~ids]

#clf = IsolationForest(max_samples = 100, random_state = 3)
#clf.fit(train)
#y_noano = clf.predict(train)
#y_noano = pd.DataFrame(y_noano, columns = ['Top'])
#y_noano[y_noano['Top'] == 1].index.values
#train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
#train.reset_index(drop = True, inplace = True)
#print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
#print("Number of rows without outliers:", train.shape[0])

data = train.values
classix = data.shape[1]-1
X = data[:,0:classix]
y = data[:,classix]
ids = np.random.rand(data.shape[0]) < 0.8
((X_train,y_train),(X_test,y_test)) = ((X[ids],y[ids]),(X[~ids],y[~ids]))

test = test.values
classix = test.shape[1]-1
X_valid = test[:,0:classix]
y_valid = test[:,classix]


seed = 37
np.random.seed(seed)

rmsprop = RMSprop(lr=0.0001)
sgd=SGD(lr=0.1)

#kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#for train, test in kfold.split(X, y):
model = Sequential()
model.add(Dense(64, input_dim=28, activation='relu', kernel_initializer='normal')) #20
model.add(Dropout(0.3)) #0
model.add(Dense(128, activation='relu', kernel_initializer='normal')) #20
model.add(Dropout(0.4)) #0.5
model.add(Dense(1, kernel_initializer='normal'))
model.compile(optimizer='rmsprop', loss='mean_squared_error')#, metrics=['mae']) # optimizer=sgd
#model.summary()

model.fit(X_train, y_train, epochs=300, verbose=1)

factor = 1

reality = y_test
predict = model.predict(X_test)
predict[abs(predict) > 8] = 0
plt.plot(predict*factor, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predict*factor))
# 1.00
print('RMSE = ' + str(rmse))

reality = y_valid
predict = model.predict(X_valid)
predict[abs(predict) > 8] = 0
plt.plot(predict*factor, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predict*factor))
# 1.00
print('RMSE = ' + str(rmse))


loss = model.evaluate(X_train, y_train,  verbose=0)
print('The RMSE on the training set is ',(loss)) #0.78

loss = model.evaluate(X_test, y_test,  verbose=0)
print('The RMSE on the testing set is ',(loss)) #1.05
loss = model.evaluate(X_valid, y_valid,  verbose=0)
print('The RMSE on the validation set is ',(loss)) #1.06




data_test = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test.csv", sep=";")
data_test = data_test.replace({',': '.'}, regex=True)
data_test = data_test.convert_objects(convert_numeric=True)
data_test['insee'] = data_test['insee'].astype(object)
data_test['ddH10_rose4'] = data_test['ddH10_rose4'].astype(object)
data_test = data_test.drop('mois', 1)
data_test = data_test.drop('date', 1)

data_test = data_test.drop('capeinsSOL0', 1)
data_test = data_test.drop('ciwcH20', 1)
data_test = data_test.drop('clwcH20', 1)
data_test = data_test.drop('nH20', 1)

# Add categorical features
encoder = LabelEncoder()
encoder.fit(data_test['ddH10_rose4'])
vent = encoder.transform(data_test['ddH10_rose4'])
vent = np_utils.to_categorical(vent)
vent = pd.DataFrame(vent)
vent.columns = ['vent1', 'vent2', 'vent3', 'vent4']
lat = []
lon = []
with tqdm(total=data_test.shape[0]) as pbar:
    for each in data_test['insee']:
        if each == 6088001:
            lat.append(43.66510)
            lon.append(7.2133184)
        elif each == 31069001:
            lat.append(43.63799)
            lon.append(1.3719825)
        elif each == 33281001:
            lat.append(44.83137)
            lon.append(-0.6920608)
        elif each == 35281001:
            lat.append(48.06560)
            lon.append(-1.7232451)
        elif each == 59343001:
            lat.append(50.57178)
            lon.append(3.1060870)
        elif each == 67124001:
            lat.append(48.53943)
            lon.append(7.6280226)
        elif each == 75114001:
            lat.append(48.82231)
            lon.append(2.3378563)
        else:
            print("ERROR")
            break
        pbar.update()
print(len(lat) == len(lon) == len(data_test['insee']))
latlon = np.vstack((lat,lon)).T
latlon = pd.DataFrame(latlon)
latlon.columns = ['lat', 'lon']
data_test = pd.concat([data_test, latlon, vent], axis=1)
data_test = data_test.drop('insee', 1)
data_test = data_test.drop('ddH10_rose4', 1)

data_test = data_test.select_dtypes(exclude=['object'])
data_test.fillna(0,inplace=True)

th2 = np.array(data_test['tH2'], dtype=pd.Series)
m_test = np.matrix(data_test)

#scale_test = StandardScaler()
#scale_test.fit(m_test)
#data_test = pd.DataFrame(scale_test.transform(m_test), columns = col_X)
data_test = pd.DataFrame(scale_x.transform(m_test), columns = col_X)
data_test = np.array(data_test)

predict = model.predict(data_test) * factor
plt.hist(predict)
plt.show()
#predict[abs(predict) > 8] = 0

X_filled = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test_answer_template.csv", sep=";")
X_filled = X_filled.drop('tH2_obs', 1)
predict = predict[:,0]
X_filled['tH2_obs'] = pd.Series(th2 + predict)
plt.hist(X_filled['tH2_obs'])
plt.show()

X_filled['tH2_obs']= X_filled['tH2_obs'].astype(str)
X_filled = X_filled.replace({'\.': ','}, regex=True)
#X_filled.to_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/sub_skMLPregressor1.csv", sep=';', index = False)







