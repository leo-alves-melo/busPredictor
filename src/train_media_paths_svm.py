import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from joblib import dump, load
import numpy as np

df = pd.read_csv('../data/train_df_90.csv', header=None)

y_df = df[90*3]
X_df = df.drop(columns=[90*3])

X_train_all, X_test, y_train_all, y_test = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

#X_train_all = adjustTrainDf(X_train_all.abs())
#X_test = adjustTrainDf(X_test.abs())

print('treinando')

model = SVC(gamma='auto', verbose=True).fit(X_train_all, y_train_all)

print('treinado')

dump(model, '../data/best_svm_90.joblib')
