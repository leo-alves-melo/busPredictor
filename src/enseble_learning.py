import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import pandas as pd
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import numpy as np

df = pd.read_csv('../data/ensembled_data.csv')

X = df[['Convolutional NN', 'Random Forest', 'SVM', 'Dump']]
y = np.ravel(df[['Correct Line']])

model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3), random_state=1)
model.fit(X, y)

dump(model, '../data/nn_ensembled.joblib')
