# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
df.fillna(0, inplace=True)



#----------
# II. Pre-processing
#----------

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
# III. Train/Test/Valid split
#----------

ids = np.random.rand(len(X)) < 0.9
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



#----------
# IV. Sklearn Neural Network Regressor
#----------


model = MLPRegressor(
        hidden_layer_sizes=(64,128,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=9, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.fit(X_train,y_train)

reality = y_test
predict = model.predict(X_test)
predict[abs(predict) > 8] = 0
plt.plot(predict, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predict))
# 1.06
print('RMSE = ' + str(rmse))

reality = y_valid
predict = model.predict(X_valid)
predict[abs(predict) > 8] = 0
plt.plot(predict, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predict))
# 1.05
print('RMSE = ' + str(rmse))



# Submission
print((X_train.shape[0] / df.shape[0]) *100)

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
data_test.fillna(0,inplace=True)

th2 = np.array(data_test['tH2'], dtype=pd.Series)
m_test = np.matrix(data_test)

#scale_test = StandardScaler()
#scale_test.fit(m_test)
#data_test = pd.DataFrame(scale_test.transform(m_test), columns = col_X)
data_test = pd.DataFrame(scale_x.transform(m_test), columns = col_X)
predict = model.predict(data_test.values)

#predict[abs(predict) > 8] = 0
plt.hist(predict)
plt.show()

X_filled = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test_answer_template.csv", sep=";")
X_filled = X_filled.drop('tH2_obs', 1)
#predict = predict[:,0]
X_filled['tH2_obs'] = pd.Series(th2 + predict, index=data_test.index)
plt.hist(X_filled['tH2_obs'])
plt.show()

X_filled['tH2_obs']= X_filled['tH2_obs'].astype(str)
X_filled = X_filled.replace({'\.': ','}, regex=True)
#X_filled.to_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/sub_skMLPregressor1.csv", sep=';', index = False)







#----------
# IV. Keras Deep Learning Regressor
#----------


#model = Sequential()
#model.add(Dense(64, input_dim=22, activation='relu'))
#model.add(Dense(1))
#model.compile(optimizer='rmsprop', loss='mse', metrics=['mae']) # optimizer=sgd
#model.summary()
#model.fit(X_train, y_train, epochs=100, verbose=1)


seed = 7
np.random.seed(seed)

rmsprop = RMSprop(lr=0.0001)
sgd=SGD(lr=0.1)

#kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#for train, test in kfold.split(X, y):
model = Sequential()
model.add(Dense(64, input_dim=22, activation='relu')) #64
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu')) #128
model.add(Dropout(0.3))
model.add(Dense(150, activation='relu')) #150
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae']) # optimizer=sgd
model.fit(X_train, y_train, epochs=250, verbose=1)


reality = y_test
predict = model.predict(X_test)
predict[abs(predict) > 8] = 0
plt.plot(predict*1.2, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predict*1.2))
# 1.00
print('RMSE = ' + str(rmse))


reality = y_valid
predict = model.predict(X_valid)
predict[abs(predict) > 8] = 0
plt.plot(predict*1.2, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predict*1.2))
# 1.00
print('RMSE = ' + str(rmse))


loss, acc = model.evaluate(X_train, y_train,  verbose=0)
print('The RMSE on the training set is ',(loss),' ',(acc)) #0.78

loss, acc = model.evaluate(X_test, y_test,  verbose=0)
print('The RMSE on the testing set is ',(loss),' ',(acc)) #1.05
loss, acc = model.evaluate(X_valid, y_valid,  verbose=0)
print('The RMSE on the validation set is ',(loss),' ',(acc)) #1.06








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
data_test.fillna(0,inplace=True)

th2 = np.array(data_test['tH2'], dtype=pd.Series)
m_test = np.matrix(data_test)

#scale_test = StandardScaler()
#scale_test.fit(m_test)
#data_test = pd.DataFrame(scale_test.transform(m_test), columns = col_X)
data_test = pd.DataFrame(scale_x.transform(m_test), columns = col_X)
data_test = np.array(data_test)

predict = model.predict(data_test)
#predict[abs(predict) > 8] = 0

plt.hist(predict)
plt.show()

X_filled = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test_answer_template.csv", sep=";")
X_filled = X_filled.drop('tH2_obs', 1)
predict = predict[:,0]
X_filled['tH2_obs'] = pd.Series(th2 + predict)
plt.hist(X_filled['tH2_obs'])
plt.show()

X_filled['tH2_obs']= X_filled['tH2_obs'].astype(str)
X_filled = X_filled.replace({'\.': ','}, regex=True)
#X_filled.to_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/sub_skMLPregressor1.csv", sep=';', index = False)











