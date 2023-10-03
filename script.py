import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from catboost import cv, Pool
from sklearn.metrics import roc_auc_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.metrics import AUC
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
RS = 121212
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import streamlit as st

df = pd.read_csv('train_dataset_Самолет.csv', low_memory=False, parse_dates=['report_date'])

df.head()

data = df.dropna(thresh=int(len(df)*0.95), axis=1).copy()
data.drop(columns=['col1454'], inplace=True)

data = data.fillna(data.mean())

data.head()

data_f = data.copy()

client_id = data['client_id']

data.drop(columns='client_id', inplace=True)

features = data.drop(columns=['target', 'report_date'])

target = data['target']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, stratify=target, random_state=RS)

features_train, target_train = SMOTE().fit_resample(features_train, target_train)

params = {
    'loss_function': 'Logloss',
    'iterations': 300,
    'custom_loss': ['AUC', 'F1'],
    'random_seed': RS,
    'learning_rate': 0.2
}

cv_data = cv(
    params=params,
    pool=Pool(features_train, label=target_train),
    fold_count=5, # Разбивка выборки на 5 кусочков
    shuffle=True, # Перемешаем наши данные
    partition_random_seed=RS,
    plot=False,
    stratified=True, 
    verbose=False
)

cv_data

baseline_auc_valid = cv_data['test-AUC-mean'].tail(1)

baseline_f1_valid = cv_data['test-F1-mean'].tail(1)

baseline_auc_valid

baseline_f1_valid

model = CatBoostClassifier(loss_function='Logloss',
    iterations=300,
    custom_loss=['AUC', 'F1'],
    random_seed= RS,
    learning_rate= 0.2,
    verbose=False)
   
model.fit(features_train, target_train)

pred = model.predict(features_test)

baseline_auc_test = roc_auc_score(target_test, pred)

st.text('ROC-AUC: ')

baseline_auc_test