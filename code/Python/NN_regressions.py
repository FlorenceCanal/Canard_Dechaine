# coding: utf-8

import os
import math
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest

nb_steps = 3000



#########################
#                       #
#         SAKHIR        #
#   WEATHER CHALLENGE   #
#                       #
#########################

# Cf. https://www.kaggle.com/zoupet/neural-network-model-for-house-prices-tensorflow
# Cf. https://www.kaggle.com/usersumit/tensorflow-dnnregressor/notebook



#----------
# I. Importation
#----------

sess = tf.InteractiveSession()

df = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/final_train.csv", sep=";")
df = df.replace({',': '.'}, regex=True)

df = df.convert_objects(convert_numeric=True)
df['tH2_obs'] = df['tH2_obs'].astype(object)

#df['ech'] = df['ech'].astype(object)
df['insee'] = df['insee'].astype(object)
df['ddH10_rose4'] = df['ddH10_rose4'].astype(object)

df.info()

ids = np.random.rand(len(df)) < 0.8
train = df[ids]
test = df[~ids]

print('Shape of the train data with all features:', train.shape)
train = train.select_dtypes(exclude=['object'])
print('Shape of the train data with numerical features:', train.shape)
train.fillna(0,inplace=True)

print('')
test = test.select_dtypes(exclude=['object'])
print('Shape of the test data with numerical features:', test.shape)
test.fillna(0,inplace=True)

print('')
print('List of features contained our dataset:', list(train.columns))



#----------
# II. Outliers
#----------

clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(train)
y_noano = clf.predict(train)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
y_noano[y_noano['Top'] == 1].index.values

train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop = True, inplace = True)
print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", train.shape[0])

train.head(10)



#----------
# III. Rescaling
#----------

import warnings
warnings.filterwarnings('ignore')

col_train = list(train.columns)
col_train_bis = list(train.columns)
col_train_bis.remove('ecart')
train_ecart = train['ecart']
test_ecart = test['ecart']
test = test.drop('ecart', 1)


m_train = np.matrix(train)
m_test  = np.matrix(test)

m_x_train = np.matrix(train.drop('ecart', axis = 1))
m_y_train = np.array(train.ecart).reshape((train.shape[0],1))

scale_y_train = MinMaxScaler()
scale_y_train.fit(m_y_train)

scale_train = MinMaxScaler()
scale_train.fit(m_train)

scale_x = MinMaxScaler()
scale_x.fit(m_x_train)

train = pd.DataFrame(scale_x.transform(m_x_train), columns = col_train_bis)
train['ecart'] = train_ecart
test  = pd.DataFrame(scale_x.transform(m_test), columns = col_train_bis)

#train.head()

plt.hist(train['ecart'])
plt.show()



# List of features
COLUMNS = col_train
FEATURES = col_train_bis
LABEL = "ecart"

# Columns for tensorflow
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

# Training set and Prediction set with the features to predict
training_set = train[COLUMNS]
prediction_set = train.ecart

# Train and Test 
x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES] , prediction_set, test_size=0.3) #, random_state = 75)
y_train = pd.DataFrame(y_train, columns = [LABEL])
training_set = pd.DataFrame(x_train, columns = FEATURES).merge(y_train, left_index = True, right_index = True)
training_set.head()

# Training for submission
#training_sub = training_set[col_train]

# Same thing but for the test set
y_test = pd.DataFrame(y_test, columns = [LABEL])
testing_set = pd.DataFrame(x_test, columns = FEATURES).merge(y_test, left_index = True, right_index = True)
testing_set.head()



#----------
# IV. Deep NN for continuous features
#----------

# Model
#tf.logging.set_verbosity(tf.logging.ERROR)

regressor1 = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          activation_fn = tf.nn.relu,
                                          hidden_units=[200, 50]#, 25, 12]#,
                                          #optimizer = tf.train.GradientDescentOptimizer( learning_rate= 0.1 )
                                          )

# Reset the index of training
training_set.reset_index(drop = True, inplace = True)

def input_fn(data_set, pred = False):
    if pred == False:
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        labels = tf.constant(data_set[LABEL].values)
        return feature_cols, labels
    if pred == True:
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        return feature_cols

# Deep Neural Network Regressor with the training set which contain the data split by train test split
regressor1.fit(input_fn=lambda: input_fn(training_set), steps=nb_steps)

# Evaluation on the test set created by train_test_split
ev = regressor1.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

# Display the score on the testing set
# 1.1541004
loss_score1 = ev["loss"]
print("Final Loss on the testing set: {0:f}".format(loss_score1))



#----------
# V. Predictions
#----------

print('prportion of training set according to complete dataset = ' +
      str((training_set.shape[0] / df.shape[0])))

y = regressor1.predict(input_fn=lambda: input_fn(testing_set))
predictions = list(itertools.islice(y, testing_set.shape[0]))
predictions = pd.DataFrame(predictions)
predictions = np.array(predictions, dtype=pd.Series)
#predictions = scale_y_train.inverse_transform(np.array(predictions).reshape(len(predictions),1))
#reality = pd.DataFrame(scale_train.inverse_transform(testing_set), columns = [COLUMNS]).ecart
reality = testing_set.ecart
plt.hist(predictions)
plt.show()

matplotlib.rc('xtick')
matplotlib.rc('ytick')
plt.plot(predictions, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()

rmse = sqrt(mean_squared_error(reality, predictions))
# 1.0742908972192025
print('RMSE = ' + str(rmse))



#----------
# VI. Check with validation set
#----------

test['ecart'] = np.zeros(test.shape[0])
y = regressor1.predict(input_fn=lambda: input_fn(test))
predictions = list(itertools.islice(y, test.shape[0]))
predictions = pd.DataFrame(predictions)
#predictions = scale_y_train.inverse_transform(np.array(predictions).reshape(len(predictions),1))
#reality = pd.DataFrame(scale_train.inverse_transform(test), columns = [COLUMNS]).ecart
reality = test_ecart

predictions[abs(predictions) > 8] = 0

plt.hist(predictions)
plt.show()

matplotlib.rc('xtick')
matplotlib.rc('ytick')
plt.plot(predictions, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()

rmse = sqrt(mean_squared_error(reality, predictions))
# 1.1333419253504484
print('RMSE = ' + str(rmse))



#----------
# VII. Submission
#----------

data_test = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test.csv", sep=";")
data_test = data_test.replace({',': '.'}, regex=True)
data_test = data_test.convert_objects(convert_numeric=True)
#data_test['ech'] = data_test['ech'].astype(object)
data_test['insee'] = data_test['insee'].astype(object)
data_test['ddH10_rose4'] = data_test['ddH10_rose4'].astype(object)
data_test = data_test.select_dtypes(exclude=['object'])
data_test.fillna(0,inplace=True)
th2 = np.array(data_test['tH2'], dtype=pd.Series)

m_test = np.matrix(data_test)

scale_test = MinMaxScaler()
scale_test.fit(m_test)
data_test = pd.DataFrame(scale_test.transform(m_test), columns = col_train_bis)
#data_test = pd.DataFrame(scale_x.transform(m_test), columns = col_train_bis)
data_test['ecart'] = np.zeros(data_test.shape[0])

y = regressor1.predict(input_fn=lambda: input_fn(data_test))
predictions = list(itertools.islice(y, data_test.shape[0]))
predictions = pd.DataFrame(predictions)
predictions = np.array(predictions, dtype=pd.Series)
#predictions = scale_y_train.inverse_transform(np.array(predictions).reshape(len(predictions),1))
predictions = predictions[:,0]
plt.hist(predictions)
plt.show()

X_filled = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test_answer_template.csv", sep=";")
X_filled = X_filled.drop('tH2_obs', 1)
X_filled['tH2_obs'] = pd.Series(th2 + predictions, index=data_test.index)

plt.hist(X_filled['tH2_obs'])
plt.show()

X_filled['tH2_obs']= X_filled['tH2_obs'].astype(str)
X_filled = X_filled.replace({'\.': ','}, regex=True)
#X_filled.to_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/sub_NN_Relu_cont.csv", sep=';', index = False)





#----------
# VIII. Leaky Relu
#----------



def leaky_relu(x):
    return tf.nn.relu(x) - 0.01 * tf.nn.relu(-x)

# Model
regressor2 = tf.contrib.learn.DNNRegressor(feature_columns = feature_cols, 
                                          activation_fn = leaky_relu,
                                          hidden_units = [300, 150, 50, 25, 12]) #[200, 100, 50, 25, 12])
    
# Deep Neural Network Regressor with the training set which contain the data split by train test split
regressor2.fit(input_fn=lambda: input_fn(training_set), steps=nb_steps)

# Evaluation on the test set created by train_test_split
ev = regressor2.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

# Display the score on the testing set
loss_score2 = ev["loss"]
# 1.183 (steps = 2000)
print("Final Loss on the testing set with Leaky Relu: {0:f}".format(loss_score2))


# Predictions
y = regressor2.predict(input_fn=lambda: input_fn(testing_set))
predictions = list(itertools.islice(y, testing_set.shape[0]))
#predictions = scale_y_train.inverse_transform(np.array(predictions).reshape(len(predictions),1))
#reality = pd.DataFrame(scale_train.inverse_transform(testing_set), columns = [COLUMNS]).ecart
reality = testing_set.ecart

plt.hist(predictions)
plt.show()
matplotlib.rc('xtick')
matplotlib.rc('ytick')
plt.plot(predictions, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predictions))
# 1.0778
print('RMSE = ' + str(rmse))

# Check with validation set
y = regressor2.predict(input_fn=lambda: input_fn(test))
predictions = list(itertools.islice(y, test.shape[0]))
predictions = pd.DataFrame(predictions)
reality = test_ecart

predictions[abs(predictions) > 8] = 0

plt.hist(predictions)
plt.show()

matplotlib.rc('xtick')
matplotlib.rc('ytick')
plt.plot(predictions, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predictions))
# 1.1550
print('RMSE = ' + str(rmse))

# Submission
print((train.shape[0] / df.shape[0]) *100)
data_test = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test.csv", sep=";")
data_test = data_test.replace({',': '.'}, regex=True)
data_test = data_test.convert_objects(convert_numeric=True)
#data_test['ech'] = data_test['ech'].astype(object)
data_test['insee'] = data_test['insee'].astype(object)
data_test['ddH10_rose4'] = data_test['ddH10_rose4'].astype(object)
data_test = data_test.select_dtypes(exclude=['object'])
data_test['ecart'] = np.zeros(data_test.shape[0])
data_test.fillna(0,inplace=True)
th2 = np.array(data_test['tH2'], dtype=pd.Series)
m_test = np.matrix(data_test)
data_test = pd.DataFrame(scale_train.transform(m_test), columns = col_train)

y = regressor2.predict(input_fn=lambda: input_fn(data_test))
predictions = list(itertools.islice(y, data_test.shape[0]))
#predictions = scale_y_train.inverse_transform(np.array(predictions).reshape(len(predictions),1))
#predictions = predictions[:,0]
plt.hist(predictions)
plt.show()

X_filled = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test_answer_template.csv", sep=";")
X_filled = X_filled.drop('tH2_obs', 1)
X_filled['tH2_obs'] = pd.Series(th2 + predictions, index=data_test.index)
plt.hist(X_filled['tH2_obs'])
plt.show()
X_filled['tH2_obs']= X_filled['tH2_obs'].astype(str)
X_filled = X_filled.replace({'\.': ','}, regex=True)
#X_filled.to_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/sub_NN_Lrelu_cont.csv", sep=';', index = False)





#----------
# VIII. Elu
#----------



# Model
regressor3 = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 
                                          activation_fn = tf.nn.elu, hidden_units=[200, 100, 50, 25, 12])
    
# Deep Neural Network Regressor with the training set which contain the data split by train test split
regressor3.fit(input_fn=lambda: input_fn(training_set), steps=nb_steps)

# Evaluation on the test set created by train_test_split
ev = regressor3.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

loss_score3 = ev["loss"]

print("Final Loss on the testing set with Elu: {0:f}".format(loss_score3))

# ...







#----------
# IX. Deep NN for continuous & factorial features
#----------



# Import and split

df = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/final_train.csv", sep=";")
df = df.replace({',': '.'}, regex=True)

df = df.convert_objects(convert_numeric=True)
df = df.drop('date', 1)
df = df.drop('mois', 1)
df = df.drop('tH2_obs', 1)
#df['ech'] = df['ech'].astype(object)
df['insee'] = df['insee'].astype(object)
df['ddH10_rose4'] = df['ddH10_rose4'].astype(object)
ids = np.random.rand(len(df)) < 0.8
train = df[ids]
test = df[~ids]

train_numerical = train.select_dtypes(exclude=['object'])
train_numerical.fillna(0, inplace = True)
train_categoric = train.select_dtypes(include=['object'])
train_categoric.fillna('NONE',inplace = True)
train = train_numerical.merge(train_categoric, left_index = True, right_index = True)
train['insee'] = train['insee'].astype(object)
train['ddH10_rose4'] = train['ddH10_rose4'].astype(object)

test_numerical = test.select_dtypes(exclude=['object'])
test_numerical.fillna(0,inplace = True)
test_categoric = test.select_dtypes(include=['object'])
test_categoric.fillna('NONE',inplace = True)
test = test_numerical.merge(test_categoric, left_index = True, right_index = True)
test['insee'] = test['insee'].astype(object)
test['ddH10_rose4'] = test['ddH10_rose4'].astype(object)

# Removie the outliers
clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(train_numerical)
y_noano = clf.predict(train_numerical)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
y_noano[y_noano['Top'] == 1].index.values

train_numerical = train_numerical.iloc[y_noano[y_noano['Top'] == 1].index.values]
train_numerical.reset_index(drop = True, inplace = True)
train_categoric = train_categoric.iloc[y_noano[y_noano['Top'] == 1].index.values]
train_categoric.reset_index(drop = True, inplace = True)
train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop = True, inplace = True)
print("Number of Outliers (numerical):", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", train.shape[0])

#train.head(10)

col_train_cat = list(train_categoric.columns)
col_train_num = list(train_numerical.columns)
col_train_num_bis = list(train_numerical.columns)
col_train_num_bis.remove('ecart')

train_ecart = train['ecart']
test_ecart = test['ecart']
test = test.drop('ecart', 1)
test_numerical = test_numerical.drop('ecart', 1)

m_train = np.matrix(train_numerical)
m_x_train = np.matrix(train_numerical.drop('ecart',axis = 1))
m_y_train = np.array(train.ecart).reshape((train.shape[0],1))

m_test  = np.matrix(test_numerical)
m_x_test = m_test

scale_y_train = MinMaxScaler()
scale_y_train.fit(m_y_train.reshape(train.shape[0],1))

scale_train = MinMaxScaler()
scale_train.fit(m_train)

scale_x = MinMaxScaler()
scale_x.fit(m_x_train)

train_num_scale = pd.DataFrame(scale_x.transform(m_x_train), columns = col_train_num_bis)
train[col_train_num_bis] = pd.DataFrame(scale_x.transform(m_x_train), columns = col_train_num_bis)
train_num_scale['ecart'] = train_ecart.values
train['ecart'] = train_ecart.values

test_num_scale = pd.DataFrame(scale_x.transform(m_x_test),columns = col_train_num_bis)
test = pd.concat([test[col_train_cat].reset_index(drop=True), test_num_scale], axis=1)
test_num_scale['ecart'] = np.zeros(test.shape[0])
test['ecart'] = np.zeros(test.shape[0])



# List of features
COLUMNS = col_train_num
FEATURES = col_train_num_bis
LABEL = "ecart"

FEATURES_CAT = col_train_cat

engineered_features = []

for continuous_feature in FEATURES:
    engineered_features.append(
        tf.contrib.layers.real_valued_column(continuous_feature))

for categorical_feature in FEATURES_CAT:
    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
            categorical_feature,
            hash_bucket_size=1000)
    engineered_features.append(
            tf.contrib.layers.embedding_column(
                    sparse_id_column=sparse_column,
                    dimension=16,
                    combiner="sum"))

# Training set and Prediction set with the features to predict
training_set = train[FEATURES + FEATURES_CAT]
prediction_set = train.ecart

# Train and Test 
x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES + FEATURES_CAT],
                                                    prediction_set,
                                                    test_size=0.3) #, random_state=42)
y_train = pd.DataFrame(y_train, columns = [LABEL])
training_set = pd.DataFrame(x_train, columns = FEATURES + FEATURES_CAT).merge(
        y_train,
        left_index = True,
        right_index = True)

# Training for submission
#training_sub = training_set[FEATURES + FEATURES_CAT]
#testing_sub = test[FEATURES + FEATURES_CAT]

# Same thing but for the test set
y_test = pd.DataFrame(y_test, columns = [LABEL])
testing_set = pd.DataFrame(x_test, columns = FEATURES + FEATURES_CAT).merge(
        y_test,
        left_index = True,
        right_index = True)

training_set[FEATURES_CAT] = training_set[FEATURES_CAT].applymap(str)
testing_set[FEATURES_CAT] = testing_set[FEATURES_CAT].applymap(str)

def input_fn_new(data_set, training = True):
    continuous_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(data_set[k].size)],
        values = data_set[k].values,
        dense_shape = [data_set[k].size, 1]) for k in FEATURES_CAT}
    # Merges the two dictionaries into one.
    feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))
    if training == True:
        # Converts the label column into a constant Tensor.
        label = tf.constant(data_set[LABEL].values)
        # Returns the feature columns and the label.
        return feature_cols, label
    return feature_cols

# Model
regressor4 = tf.contrib.learn.DNNRegressor(feature_columns = engineered_features, 
                                          activation_fn = tf.nn.relu,
                                          hidden_units=[300, 150, 50, 25, 12]) #[200, 100, 50, 25, 12])

categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(training_set[k].size)],
                                                values = training_set[k].values,
                                                dense_shape = [training_set[k].size, 1]) for k in FEATURES_CAT}

# Deep Neural Network Regressor with the training set which contain the data split by train test split
regressor4.fit(input_fn = lambda: input_fn_new(training_set) , steps=nb_steps)

ev = regressor4.evaluate(input_fn=lambda: input_fn_new(testing_set, training = True), steps=1)

loss_score4 = ev["loss"]
# 1.029766
print("Final Loss on the testing set: {0:f}".format(loss_score4))


# Predictions
y = regressor4.predict(input_fn=lambda: input_fn_new(testing_set))
predictions = list(itertools.islice(y, testing_set.shape[0]))
#predictions = scale_y_train.inverse_transform(np.array(predictions).reshape(len(predictions),1))
#reality = pd.DataFrame(scale_train.inverse_transform(testing_set), columns = [COLUMNS]).ecart
reality = testing_set.ecart

plt.hist(predictions)
plt.show()
matplotlib.rc('xtick')
matplotlib.rc('ytick')
plt.plot(predictions, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predictions))
# RMSE = 0.9883072269310472
print('RMSE = ' + str(rmse))


# Check with validation set
test = pd.DataFrame(test, columns = FEATURES + FEATURES_CAT)
test['ecart'] = np.zeros(test.shape[0])
test[FEATURES_CAT] = test[FEATURES_CAT].applymap(str)
y = regressor4.predict(input_fn=lambda: input_fn_new(test))
predictions = list(itertools.islice(y, test.shape[0]))
reality = test_ecart
#predictions = [value for value in predictions if not math.isnan(value)]
plt.hist(predictions)
plt.show()
#predictions[predictions > 5] = 2
matplotlib.rc('xtick')
matplotlib.rc('ytick')
plt.plot(predictions, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predictions))
# RMSE = 1.0572891488237737
print('RMSE = ' + str(rmse))


# Submission
print((train.shape[0] / df.shape[0]) *100)
data_test = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test.csv", sep=";")
data_test = data_test.replace({',': '.'}, regex=True)
data_test = data_test.convert_objects(convert_numeric=True)
data_test = data_test.drop('date', 1)
data_test = data_test.drop('mois', 1)
data_test['insee'] = data_test['insee'].astype(object)
data_test['ddH10_rose4'] = data_test['ddH10_rose4'].astype(object)

test_numerical = data_test.select_dtypes(exclude=['object'])
test_numerical.fillna(0,inplace = True)
test_categoric = data_test.select_dtypes(include=['object'])
test_categoric.fillna('NONE',inplace = True)
data_test = test_numerical.merge(test_categoric, left_index = True, right_index = True)
data_test['insee'] = data_test['insee'].astype(object)
data_test['ddH10_rose4'] = data_test['ddH10_rose4'].astype(object)
data_test['ecart'] = np.zeros(data_test.shape[0])
th2 = np.array(data_test['tH2'], dtype=pd.Series)

m_test  = np.matrix(test_numerical)
m_x_test = m_test
test_num_scale = pd.DataFrame(scale_x.transform(m_x_test),columns = col_train_num_bis)
data_test = pd.concat([data_test[col_train_cat].reset_index(drop=True), test_num_scale], axis=1)
test_num_scale['ecart'] = np.zeros(data_test.shape[0])
data_test['ecart'] = np.zeros(data_test.shape[0])

data_test[FEATURES_CAT] = data_test[FEATURES_CAT].applymap(str)
y = regressor4.predict(input_fn=lambda: input_fn_new(data_test))
predictions = list(itertools.islice(y, data_test.shape[0]))
#predictions = scale_y_train.inverse_transform(np.array(predictions).reshape(len(predictions),1))
#predictions = predictions[:,0]
plt.hist(predictions)
plt.show()

X_filled = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test_answer_template.csv", sep=";")
X_filled = X_filled.drop('tH2_obs', 1)
X_filled['tH2_obs'] = pd.Series(th2 + predictions, index=data_test.index)
plt.hist(X_filled['tH2_obs'])
plt.show()
X_filled['tH2_obs']= X_filled['tH2_obs'].astype(str)
X_filled = X_filled.replace({'\.': ','}, regex=True)
#X_filled.to_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/sub_NN_Relu.csv", sep=';', index = False)





#----------
# X. Shallow Network
#----------



# Model
regressor5 = tf.contrib.learn.DNNRegressor(feature_columns = engineered_features, 
                                          activation_fn = tf.nn.relu, hidden_units=[1000])

# Deep Neural Network Regressor with the training set which contain the data split by train test split
regressor5.fit(input_fn = lambda: input_fn_new(training_set) , steps=nb_steps)

ev = regressor5.evaluate(input_fn=lambda: input_fn_new(testing_set, training = True), steps=1)
loss_score5 = ev["loss"]
# ~ 1.157061
print("Final Loss on the testing set: {0:f}".format(loss_score5))


# Predictions
y = regressor5.predict(input_fn=lambda: input_fn_new(testing_set))
predictions = list(itertools.islice(y, testing_set.shape[0]))
#predictions = scale_y_train.inverse_transform(np.array(predictions).reshape(len(predictions),1))
#reality = pd.DataFrame(scale_train.inverse_transform(testing_set), columns = [COLUMNS]).ecart
reality = testing_set.ecart

plt.hist(predictions)
plt.show()

matplotlib.rc('xtick')
matplotlib.rc('ytick')
plt.plot(predictions, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predictions))
# 1.07566762
print('RMSE = ' + str(rmse))


# Check with validation set
test = pd.DataFrame(test, columns = FEATURES + FEATURES_CAT)
test['ecart'] = np.zeros(test.shape[0])
test[FEATURES_CAT] = test[FEATURES_CAT].applymap(str)
y = regressor5.predict(input_fn=lambda: input_fn_new(test))
predictions = list(itertools.islice(y, test.shape[0]))
predictions = pd.DataFrame(predictions)
reality = test_ecart
#predictions = [value for value in predictions if not math.isnan(value)]
predictions[abs(predictions) > 8] = 0
plt.hist(predictions)
plt.show()
matplotlib.rc('xtick')
matplotlib.rc('ytick')
plt.plot(predictions, reality, 'ro')
plt.xlabel('Predictions')
plt.ylabel('Reality')
plt.title('Predictions x Reality on dataset Test')
plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()
rmse = sqrt(mean_squared_error(reality, predictions))
# 1.151139315
print('RMSE = ' + str(rmse))


# Submission
print((train.shape[0] / df.shape[0]) *100)
data_test = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test.csv", sep=";")
data_test = data_test.replace({',': '.'}, regex=True)
data_test = data_test.convert_objects(convert_numeric=True)
data_test = data_test.drop('date', 1)
data_test = data_test.drop('mois', 1)
data_test['insee'] = data_test['insee'].astype(object)
data_test['ddH10_rose4'] = data_test['ddH10_rose4'].astype(object)

test_numerical = data_test.select_dtypes(exclude=['object'])
test_numerical.fillna(0,inplace = True)
test_categoric = data_test.select_dtypes(include=['object'])
test_categoric.fillna('NONE',inplace = True)
data_test = test_numerical.merge(test_categoric, left_index = True, right_index = True)
data_test['insee'] = data_test['insee'].astype(object)
data_test['ddH10_rose4'] = data_test['ddH10_rose4'].astype(object)
data_test['ecart'] = np.zeros(data_test.shape[0])
th2 = np.array(data_test['tH2'], dtype=pd.Series)

m_test  = np.matrix(test_numerical)
m_x_test = m_test
test_num_scale = pd.DataFrame(scale_x.transform(m_x_test),columns = col_train_num_bis)
data_test = pd.concat([data_test[col_train_cat].reset_index(drop=True), test_num_scale], axis=1)
test_num_scale['ecart'] = np.zeros(data_test.shape[0])
data_test['ecart'] = np.zeros(data_test.shape[0])

data_test[FEATURES_CAT] = data_test[FEATURES_CAT].applymap(str)
y = regressor5.predict(input_fn=lambda: input_fn_new(data_test))
predictions = list(itertools.islice(y, data_test.shape[0]))
#predictions = scale_y_train.inverse_transform(np.array(predictions).reshape(len(predictions),1))
#predictions = predictions[:,0]
plt.hist(predictions)
plt.show()

X_filled = pd.read_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test_answer_template.csv", sep=";")
X_filled = X_filled.drop('tH2_obs', 1)
X_filled['tH2_obs'] = pd.Series(th2 + predictions, index=data_test.index)
plt.hist(X_filled['tH2_obs'])
plt.show()
X_filled['tH2_obs']= X_filled['tH2_obs'].astype(str)
X_filled = X_filled.replace({'\.': ','}, regex=True)
#X_filled.to_csv("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/sub_ShallowN_1.csv", sep=';', index = False)







# loss_score3 = 'Elu_cont'
list_score = [loss_score1, loss_score2, loss_score4, loss_score5]
list_model = ['Relu_cont', 'LRelu_cont', 'Relu_cont_categ', 'Shallow_1ku']


objects = list_model
y_pos = np.arange(len(objects))
performance = list_score
 
plt.barh(y_pos, performance, align='center', alpha=0.9)
plt.yticks(y_pos, objects)
plt.xlabel('Loss')
plt.title('Models compared')





