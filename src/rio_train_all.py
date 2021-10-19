from lib.classes import *
from lib.data_visualization import *
from lib.rio_data_filter import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from joblib import dump, load
import numpy as np

paths_df = pd.read_csv('../data/rio_train_df_60_labeled.csv', header=None)

y_df = paths_df[60*3]
X_df = paths_df.drop(columns=[60*3])

X_train_all, X_test, y_train_all, y_test = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

print('treinando')

# Train SVM
#modelSVM = SVC(gamma='auto', verbose=True).fit(X_df, y_df)

# Train RF
#params = {'n_estimators': [40, 50, 60, 70], 'max_depth': [50, 60, 70], 'random_state': [0]}
#modelRF = GridSearchCV(RandomForestClassifier(), params, scoring='neg_root_mean_squared_error')
#modelRF.fit(X_df, y_df)
#dump(modelRF, '../data/rio_best_RF_60.joblib')

X_df = X_df.abs()
modelBayes = ComplementNB()
modelBayes.fit(X_df, y_df)
dump(modelBayes, '../data/rio_best_bayes_60.joblib')

print('treinado')

