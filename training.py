# -*- coding: utf-8 -*-
"""Higgs_final8.ipynb"""
## Importing Libraries"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
from collections import Counter
import warnings
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,normalize,MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

#Importing Libraries
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,normalize,MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
import seaborn as sns
import tensorflow as tf

"""## Loading Data"""

data = pd.read_csv('/content/drive/MyDrive/higgs/training.csv', delimiter=',')

data.shape

data_new=data.drop(["Weight"], axis=1)
data_new.head()

data_new= data_new[['DER_mass_MMC','DER_mass_transverse_met_lep','DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet','Label']]

data_new

"""## EDA"""

data_new.describe()

data_new.info()

data_new.dtypes

fig,axes=plt.subplots(figsize=(10,8))
print(data_new['Label'].value_counts())
sns.barplot(x = data_new['Label'].value_counts().index, y = data_new['Label'].value_counts().values)
plt.title('Counts of label')
plt.show()

#Finding Null Value
data_new.isnull().sum()

# Encoding the labels
label_data = preprocessing.LabelEncoder()
data_new['Label'] = label_data.fit_transform(data_new['Label'])
data_new.head()

"""## Corelation Matrix"""

corr = data_new.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 230, n=10),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)



"""## Box plot"""

#IQR Value of Each Column
Q1 = data_new.quantile(0.25)
Q3 =  data_new.quantile(0.75)
IQR = Q3 - Q1
print(IQR)



import math
cols = 3
rows = math.ceil(len(data_new.columns)/cols)


fig, axen = plt.subplots(rows, cols, figsize = (12, 50))
for v, ax in zip(data_new.columns, axen.ravel()):
    sns.histplot(data_new[v], ax=ax)
    ax.set_xscale('log')

fig, axen = plt.subplots(rows, cols, figsize = (12, 64))
for v, ax in zip(data_new.columns, axen.ravel()):
    sns.set(style='whitegrid')
    sns.boxplot(x="Label",
                y=data_new[v],
                data=data_new,ax=ax)

def remove_outlier(df):
    QI = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR =    Q3-Q1 
    fence_low  = Q1-1.5*IQR
    fence_high = Q3+1.5*IQR
    df_new= df[~((df > fence_low) & (df< fence_high))]
    return df_new

dr = remove_outlier(data_new)

dr.tail()

dr.head()

sns.set(style='whitegrid')
sns.boxplot(x="Label",
                y="DER_deltar_tau_lep",
                data=dr)

fig, axen = plt.subplots(rows, cols, figsize = (12, nrows*4))
for v, ax in zip(data_new.columns, axen.ravel()):
    dr1= remove_outlier(data_new)
    sns.set(style='whitegrid')
    sns.boxplot(x="Label",
                y=dr1[v],
                data=dr,ax=ax)

"""## Data Preprocessing"""

data_new.isnull().sum()

data_new1= data_new[['DER_mass_MMC','DER_mass_transverse_met_lep','DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet','Label']]

label_data = preprocessing.LabelEncoder()
data_new1['Label'] = label_data.fit_transform(data_new1['Label'])
data_new1.head()

data_new1

data_new1[data_new1==-999.000] = np.NaN

data_new1.fillna(data_new1.mean(), inplace = True)

data_new1

y_train = data_new1['Label'].values
x_train = data_new1.drop(['Label'], axis=1)

x_train.astype

data_new.astype

from sklearn.model_selection import train_test_split
X_train_SS, X_test_SS, y_train_SS, y_test_SS = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

#pip install --upgrade pandas

import numpy
X_train_SS= X_train_SS.to_numpy()
X_test_SS=X_test_SS.to_numpy()

import pandas
print(pandas.__version__)

X_test_SS=X_test_SS.to_numpy()

X_train_SS.shape

# reshape for rnn

X_train_SS = X_train_SS.reshape(X_train_SS.shape[0],X_train_SS.shape[1],1)
X_test_SS =X_test_SS.reshape(X_test_SS.shape[0],X_test_SS.shape[1],1)

columns_ = x_train.iloc[:].columns
columns_.shape

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
df_scaled = pd.DataFrame(scalar.fit_transform(x_train), columns = columns_)
df = pd.DataFrame(df_scaled)

df

from sklearn.decomposition import PCA
pca = PCA()
df_pca = pd.DataFrame(pca.fit_transform(df))
df_pca

import matplotlib.pyplot as plt
pd.DataFrame(pca.explained_variance_ratio_).plot.bar()
plt.legend('')
plt.xlabel('Principal Components')
plt.ylabel('Explained Varience');

pca_15 = PCA(n_components=4, random_state = 2020)
principalComponents_15 = pca_15.fit_transform(df)
print('Explained variation by 4 principal components: {}'.format(sum(pca_15.explained_variance_ratio_)*100))

principal_cols = ['Principal component '+str(i) for i in range(principalComponents_15.shape[1])]
pca_df = pd.DataFrame(data = principalComponents_15,columns=principal_cols)
pca_df.head()

"""## Building LSTM Model"""

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate,GRU,Dropout,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.io import FixedLenFeature
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, normalize
from keras.models import Sequential
from keras.layers import Bidirectional

def build_Lstm_model(train_x,train_y,test_x,test_y):
    inp = Input(shape=(train_x.shape[1],train_x.shape[2]))
    rnn_1st_model = LSTM(units=60, return_sequences=True,recurrent_dropout=0.1)(inp)
    rnn_2nd_model = LSTM(units=60,recurrent_dropout=0.1)(rnn_1st_model)
    dense_layer = Dense(128)(rnn_2nd_model)
    drop_out = Dropout(0.2)(dense_layer)
    output = Dense(1, activation= "sigmoid")(drop_out)
    model = Model(inp, output)
    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1, factor=0.6),
                 EarlyStopping(monitor='val_loss', patience=20),
                 ModelCheckpoint(filepath='best_model_LSTM.h5', monitor='val_loss', save_best_only=True)]
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam")
    history = model.fit(train_x, train_y, 
          epochs = 10, 
          batch_size = 128, 
          validation_data=(test_x,  test_y), 
          callbacks=callbacks)
    return history,model

def plot_Loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss over epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.show()

history_LSTM,Lstm_model = build_Lstm_model(X_train_SS,y_train_SS,X_test_SS,y_test_SS)

plot_Loss(history_LSTM)

Lstm_model.save("LSTM_Higgs_model.h5")

def prediction(model,input):
    input= np.array(input)
    prediction = model.predict(input)
    return 'b' if prediction[0][0] >= 0.5 else 's'



