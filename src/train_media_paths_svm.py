import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import pandas as pd
from lib.classes import *
from lib.data_filter import *
from lib.data_processor import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lib.data_normalization import *
from joblib import dump, load

df = pd.read_csv('../data/train_df.csv', header=None)

y_df = df[180]
X_df = df.drop(columns=[180])

X_train_all, X_test, y_train_all, y_test = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

X_train_all = adjustTrainDf(X_train_all.abs())
X_test = adjustTrainDf(X_test.abs())

print 'treinando'

model = SVC(gamma='auto', verbose=True).fit(X_train_all, y_train_all)

print 'treinado'
predicted = model.predict(X_test)
final_value = 100.0 * np.sum(predicted == y_test) / len(y_test.index)

print final_value

dump(model, '../data/best_svm.joblib')
