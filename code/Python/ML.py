# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from time import time
import datetime
from math import sqrt
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb
from xgboost import plot_importance
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.optimizers import RMSprop

t0 = time()


#########################
#                       #
#         SAKHIR        #
#   WEATHER CHALLENGE   #
#                       #
#########################


# ----------
# I. Load data
# ----------

Train = []
for i in range(1, 37):
    Train.append(pd.read_csv('C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/train/train_{}.csv'.format(i), sep=';', decimal=","))

train = pd.concat(Train)
train['test'] = 0

X_test = pd.read_csv('C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/data/test/test.csv', sep=';', decimal=",")
X_test['test'] = 1

df = pd.concat([train, X_test])

df['flvis1SOL0'] = df['flvis1SOL0'].apply(lambda x: float(str(x).replace(',', '.')))
df.head()

# plt.figure(figsize=(15,10))
# sns.pairplot(df[0:200], hue='insee')
# plt.show()
# g = sns.pairplot(df['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked'],
#                  hue='Survived',
#                  palette = 'seismic',
#                  size=1.2,
#                  diag_kind='kde',
#                  diag_kws=dict(shade=True),
#                  plot_kws=dict(s=10))
# g.set(xticklabels=[])


# ----------
# II. Add data
# ----------

df['seasons'] = df['mois'].map({'janvier': 'hiver', 'février': 'hiver', 'mars': 'printemps', 'avril': 'printemps', 'mai': 'printemps', 'juin': 'été', 'juillet': 'été', "août": "été", "septembre": "automne", "octobre": "automne", "novembre": "automne", "décembre": "hiver"})
df['year'] = df['date'].apply(lambda x: x[:4])

df['city'] = df['insee'].map({6088001: 'Nice', 31069001: 'Toulouse', 33281001: 'Bordeaux', 35281001: 'Rennes', 59343001: 'Lille', 67124001: 'Strasbourg', 75114001: 'Paris'})
df['lat'] = df['city'].map({'Nice': 43.66510, 'Toulouse': 43.63799, 'Bordeaux': 44.83137, 'Rennes': 48.06560, 'Lille': 50.57178, 'Strasbourg': 48.53943, 'Paris': 48.82231})
df['lon'] = df['city'].map({'Nice': 7.2133184, 'Toulouse': 1.3719825, 'Bordeaux': -0.6920608, 'Rennes': -1.7232451, 'Lille': 3.1060870, 'Strasbourg': 7.6280226, 'Paris': 2.3378563})
df['fleuve'] = df['city'].map({'Nice': 0, 'Toulouse': 1, 'Bordeaux': 1, 'Rennes': 0, 'Lille': 0, 'Strasbourg': 1, 'Paris': 1})
df['distmer'] = df['city'].map({'Nice': 0, 'Toulouse': 150, 'Bordeaux': 65, 'Rennes': 70, 'Lille': 80, 'Strasbourg': 540, 'Paris': 200})

# Source : http://www.linternaute.com/voyage/climat/comparaison/toulouse/rennes/2015/ville-31555/ville-35238
df['ensoleillement16'] = df['city'].map({'Nice': 2771, 'Toulouse': 2100, 'Bordeaux': 2022, 'Rennes': 1679, 'Lille': 1653, 'Strasbourg': 1625, 'Paris': 1726})
df['ensoleillement15'] = df['city'].map({'Nice': 2917, 'Toulouse': 2238, 'Bordeaux': 2151, 'Rennes': 1851, 'Lille': 1739, 'Strasbourg': 1886, 'Paris': 1953})
df['precipitations16'] = df['city'].map({'Nice': 484, 'Toulouse': 577, 'Bordeaux': 922, 'Rennes': 625, 'Lille': 836, 'Strasbourg': 730, 'Paris': 658})
df['precipitations15'] = df['city'].map({'Nice': 669, 'Toulouse': 511, 'Bordeaux': 592, 'Rennes': 596, 'Lille': 786, 'Strasbourg': 475, 'Paris': 507})

df = pd.concat([df, pd.get_dummies(df[['seasons', 'year']])], axis=1)
df = df.drop('seasons', 1)
df = df.drop('mois', 1)
df = df.drop('year', 1)
df = df.drop('city', 1)

df.head()
print('Size of df : '+str(len(df)))


# ----------
# III. Deal with missing values
# ----------

df.isnull().sum()
numeric_features = ['tH2', 'capeinsSOL0', 'ciwcH20', 'clwcH20', 'nH20',
                    'pMER0', 'rr1SOL0', 'rrH20', 'tpwHPA850', 'ux1H10',
                    'vapcSOL0', 'vx1H10', 'ddH10_rose4', 'ffH10', 'flir1SOL0',
                    'fllat1SOL0', 'flsen1SOL0', 'flvis1SOL0', 'hcoulimSOL0',
                    'huH2', 'iwcSOL0', 'nbSOL0_HMoy', 'ntSOL0_HMoy',
                    'tH2_VGrad_2.100', 'tH2_XGrad', 'tH2_YGrad']
qualitative_features = ['insee', 'mois']
imput = Imputer(missing_values='NaN', strategy='mean', axis=0)
imput.fit(df[numeric_features])
df[numeric_features] = imput.transform(df[numeric_features])

numeric_features.remove('tH2')

scale_num = StandardScaler()
scale_num.fit(df[numeric_features])
df[numeric_features] = scale_num.transform(df[numeric_features])

df.head()


# ----------
# IV. Split train & test & prepare for modeling
# ----------

X_test = df[df['test'] == 1].drop('test', 1)
train = df[df['test'] == 0].drop('test', 1)

date = train['date']


def to_time(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d').date()
train['date'] = date.map(to_time, date.all())

X_test_date = X_test['date']
X_test_ech = X_test['ech']
X_test_insee = X_test['insee']
X_test = X_test.drop('date', 1).drop('insee', 1).drop('tH2_obs', 1)

train['ecart'] = train['tH2_obs'] - train['tH2']
train['ecart'] = np.nan_to_num(train['ecart'])
print('Size of train : '+str(len(train)))

train = train.drop('tH2_obs', 1).drop('insee', 1).drop('date', 1)


# ----------
# V. Delete outliers from train set (isolation forest)
# ----------

# clf = IsolationForest(max_samples = 100, random_state = 3)
# clf.fit(train)
# y_noano = clf.predict(train)
# y_noano = pd.DataFrame(y_noano, columns = ['Top'])
# y_noano[y_noano['Top'] == 1].index.values
# train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
# train.reset_index(drop = True, inplace = True)
# print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
# print("Number of rows without outliers:", train.shape[0])


# ----------
# VI. Split train & validation set
# ----------

y_train = train['ecart']
X_train = train.drop('ecart', 1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
X_train.head()

obj_save_path = 'C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT/Sakhir/code/Python/obj_save/'
# pickle.dump((X_train, X_val, y_train, y_val), open(obj_save_path+'train_val_df.p', 'wb'))
X_train, X_val, y_train, y_val = pickle.load(open(obj_save_path+'train_val_df.p', 'rb'))
print('Ready to start ML part !')


# ----------
# VII. Random Forest
# ----------

print('ML part. I : starting Random Forest !')

model_rf = RandomForestRegressor(n_estimators=500,
                                 max_depth=15,
                                 min_samples_split=40,
                                 min_samples_leaf=20,
                                 max_leaf_nodes=1750,
                                 bootstrap=True, oob_score=True,
                                 random_state=321, n_jobs=4, verbose=1)

model_rf.fit(X_train, y_train)

# pickle.dump(model_rf, open(obj_save_path+'model_rf.p', 'wb'))
model_rf = pickle.load(open(obj_save_path+'model_rf.p', 'rb'))


def plot_imp_rf(model_rf, X):
    importances = model_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model_rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    names = X.columns[indices]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print(str(f+1)+'. feature '+str(names[f])+' ('+str(importances[indices[f]])+')')
    # Plot the feature importances of the forest
    plt.figure(figsize=(15, 10))
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), names, rotation=80)
    plt.xlim([-1, X.shape[1]])
    plt.show()

plot_imp_rf(model_rf, X_train)
# oob_error = 1 - model_rf.oob_score_


def verif_valid(model, X_val, y_val):
    if type(model) == Sequential:
        X_val = np.array(X_val)
    reality = y_val
    predictions = model.predict(X_val)
    if len(predictions.shape) == 2:
        predictions = predictions[:, 0]
    plt.hist(predictions)
    plt.title('Histogram of Predictions')
    plt.show()
    plt.plot(predictions, reality, 'ro')
    plt.xlabel('Predictions')
    plt.ylabel('Reality')
    plt.title('Predictions x Reality')
    plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
    plt.show()
    if type(model) != Sequential:
        print('Explained var (total) = '+str(explained_variance_score(X_val['tH2']+predictions, X_val['tH2']+reality)))
    print('Explained var (predictions) = '+str(explained_variance_score(reality, predictions)))
    print('RMSE = '+str(sqrt(mean_squared_error(reality, predictions))))

verif_valid(model_rf, X_val, y_val)

print('ML part. I : Random Forest, done !')


# ----------
# VIII. XGBoost
# ----------

print('ML part. II : starting XGBoost !')

model_xgb = xgb.XGBRegressor(base_score=0.5,
                             subsample=0.8,
                             max_delta_step=2,
                             max_depth=7,
                             min_child_weight=3,
                             n_estimators=580,
                             colsample_bytree=0.85,
                             gamma=0,
                             seed=321, silent=0)

model_xgb.fit(X_train, y_train)

# pickle.dump(model_xgb, open(obj_save_path+'model_xgb.p', 'wb'))
model_xgb = pickle.load(open(obj_save_path+'model_xgb.p', 'rb'))

plot_importance(model_xgb)
plt.figure(figsize=(15, 15))
plt.show()

verif_valid(model_xgb, X_val, y_val)

print('ML part. II : XGBoost, done !')


# ----------
# IX. ADABOOST
# ----------

print('ML part. III : starting Adaboost !')

model_adab = AdaBoostRegressor(base_estimator=RandomForestRegressor(),
                               n_estimators=300,
                               learning_rate=0.28,
                               loss='linear',
                               random_state=321)

model_adab.fit(X_train, y_train)

# pickle.dump(model_adab, open(obj_save_path+'model_adab.p', 'wb'))
model_adab = pickle.load(open(obj_save_path+'model_adab.p', 'rb'))

verif_valid(model_adab, X_val, y_val)

print('ML part. III : Adaboost, done !')


# ----------
# X. GRADIENT BOOSTING
# ----------

print('ML part. IV : starting Gradient Boosting !')

model_gradb = GradientBoostingRegressor(loss='huber',
                                        alpha=0.9,
                                        learning_rate=0.6,
                                        n_estimators=500,
                                        subsample=0.9,
                                        min_samples_leaf=450,
                                        max_depth=6,
                                        random_state=321, verbose=0)

model_gradb.fit(X_train, y_train)

# pickle.dump(model_gradb, open(obj_save_path+'model_gradb.p', 'wb'))
model_gradb = pickle.load(open(obj_save_path+'model_gradb.p', 'rb'))

verif_valid(model_gradb, X_val, y_val)

print('ML part. IV : Gradient Boosting, done !')


# ----------
# XIII. Neural Network
# ----------

print('ML part. V : starting Neural Network !')

scale_x = StandardScaler()
scale_x.fit(X_train)
X_train_scale = pd.DataFrame(scale_x.transform(X_train))
X_val_scale = pd.DataFrame(scale_x.transform(X_val))


def nn_model():
    seed = 321
    np.random.seed(seed)
    rmsprop = RMSprop(lr=0.0001)
    # sgd=SGD(lr=0.1)
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # for train, test in kfold.split(X, y):
    model_nn = Sequential()
    model_nn.add(Dense(300, input_dim=43, activation='relu', kernel_initializer='normal'))
    model_nn.add(Dropout(0.3))
    model_nn.add(Dense(200, activation='relu', kernel_initializer='normal'))
    model_nn.add(Dropout(0.3))
    model_nn.add(Dense(150, activation='relu', kernel_initializer='normal'))
    model_nn.add(Dropout(0.2))
    model_nn.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model_nn.compile(optimizer=rmsprop, loss='mean_squared_error')
    return model_nn

model_nn = nn_model()
model_nn.summary()
hist = model_nn.fit(np.array(X_train_scale), np.array(y_train),
                    epochs=300, validation_split=0.33,
                    shuffle=True, verbose=1)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model RMSE')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()

# model_nn.save(obj_save_path+'model_nn.p')
model_nn = load_model(obj_save_path+'model_nn.p')

verif_valid(model_nn, X_val_scale, y_val)

print('ML part. V : Neural Network, done !')


# ----------
# XIV. Stacking model (Neural Network)
# ----------

print('ML part. VI : starting Stacking Model (Neural Network) !')


def pred_ML(X):
    model_rf = pickle.load(open(obj_save_path+'model_rf.p', 'rb'))
    model_xgb = pickle.load(open(obj_save_path+'model_xgb.p', 'rb'))
    model_adab = pickle.load(open(obj_save_path+'model_adab.p', 'rb'))
    model_gradb = pickle.load(open(obj_save_path+'model_gradb.p', 'rb'))
    model_nn = load_model(obj_save_path+'model_nn.p')

    rf = pd.DataFrame(model_rf.predict(X))
    xgb = pd.DataFrame(model_xgb.predict(X))
    adab = pd.DataFrame(model_adab.predict(X))
    gradb = pd.DataFrame(model_gradb.predict(X))
    X_scale = pd.DataFrame(scale_x.transform(X))
    nn = pd.DataFrame(model_nn.predict(np.array(X_scale))[:, 0])
    X_stack = pd.concat([rf, xgb, adab, gradb, nn], axis=1)
    return X_stack

X_train_stack = pred_ML(X_train)
X_val_stack = pred_ML(X_val)
X_test_stack = pred_ML(X_test)

scale_x_stack = StandardScaler()
scale_x_stack.fit(X_train_stack)
X_train_stack_scale = pd.DataFrame(scale_x_stack.transform(X_train_stack))
X_val_stack_scale = pd.DataFrame(scale_x_stack.transform(X_val_stack))
X_test_stack_scale = pd.DataFrame(scale_x_stack.transform(X_test_stack))


def stacking_nn_model():
    seed = 321
    np.random.seed(seed)
    model_nn = Sequential()
    model_nn.add(Dense(5, input_dim=5, activation='relu', kernel_initializer='normal'))
    model_nn.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model_nn.compile(optimizer='adam', loss='mean_squared_error')
    return model_nn

model_stack_nn = stacking_nn_model()
model_stack_nn.summary()
hist = model_stack_nn.fit(np.array(X_train_stack_scale), np.array(y_train),
                          epochs=20, validation_split=0.33,
                          shuffle=True, verbose=1)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model RMSE')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()

# model_stack_nn.save(obj_save_path+'model_stack_nn.p')
model_stack_nn = load_model(obj_save_path+'model_stack_nn.p')

verif_valid(model_stack_nn, X_val_stack_scale, y_val)

print('ML part. VI : Stacking Model (Neural Network), done !')


# ----------
# XV. Stacking Model (mean)
# ----------

print('ML part. VII : starting Stacking Model (mean) !')


def check_predictions(predictions, reality):
    plt.hist(predictions)
    plt.title('Histogram of Predictions')
    plt.show()
    plt.plot(predictions, reality, 'ro')
    plt.xlabel('Predictions')
    plt.ylabel('Reality')
    plt.title('Predictions x Reality')
    plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
    plt.show()
    print('Explained var (predictions) = '+str(explained_variance_score(reality, predictions)))
    print('RMSE = '+str(sqrt(mean_squared_error(reality, predictions))))

pred_val_stack_avg = X_val_stack.mean(axis=1)
check_predictions(pred_val_stack_avg, y_val)

print('ML part. VII : Stacking Model (mean), done !')


# ----------
# XVI. Predictions on the test set
# ----------

# Différents models construits :
# [model_rf, model_xgb, model_adab, model_gradb, model_nn, model_stack_nn]

# Choix du plus performant :
model = model_stack_nn

X_test_scale = pd.DataFrame(scale_x.transform(X_test))
X_test_scale = np.array(X_test_scale)
X_test_stack_scale = np.array(X_test_stack_scale)

predictions = model.predict(X_test_stack_scale)
predictions = predictions[:, 0]
plt.hist(predictions)
plt.show()

predictions = X_test['tH2'] + predictions
plt.hist(predictions)
plt.show()

res = pd.DataFrame({
        'date': X_test_date,
        'insee': X_test_insee,
        'ech': X_test_ech,
        'tH2_obs': predictions},
        columns=['date', 'insee', 'ech', 'tH2_obs'])

res.to_csv('new_sub.csv', sep=";", index=False)

print('THE END : submission ready !')


tf = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
print('time end : ', tf)
print('time enlapsed : ', int((time()-t0)/60), 'min')
