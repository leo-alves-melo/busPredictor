import pandas as pd
#from lib.classes import *
#from lib.data_filter import *
#from lib.data_processor import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import numpy as np

sizes = [180]

for size in sizes:
    df = pd.read_csv('../data/train_df_' + str(size)+ '.csv', header=None)

    y_df = df[size*3]
    X_df = df.drop(columns=[size*3])

    X_train_all, X_test, y_train_all, y_test = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

    X_train, X_cross, y_train, y_cross = train_test_split(X_train_all, y_train_all, random_state=1, test_size=0.2, stratify=y_train_all)

    n_estimators = [40, 50, 60, 70]
    max_depths = [50, 60, 70]

    best_value = 0
    best_n_estimators = 0
    best_max_depth = 0

    model = RandomForestClassifier(n_estimators=50, max_depth=60, random_state=0).fit(X_train, y_train)

    for n_estimator in n_estimators:
        for max_depth in max_depths:

            print('n_estimators: ' + str(n_estimator))
            print('max_depths: ' + str(max_depth))
            model = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, random_state=0).fit(X_train, y_train)
            predicted = model.predict(X_cross)

            current_value = np.sum(predicted == y_cross)
            if current_value > best_value:
                best_value = current_value
                best_n_estimators = n_estimator
                best_max_depth = max_depth
                print('best value:', best_value)

    model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=0).fit(X_train, y_train)
    dump(model, '../data/random_forest_' + str(size) + '.joblib')

exit()
##dump(model, '../data/best_rf.joblib')
#
#model = load('../data/best_rf.joblib')
#
#predicted = model.predict(X_test)
#
#
#print((predicted == y_test).value_counts())
#
#exit()

#for neurons in number_of_neurons1:
#
#	print 'Number of neurons: ' + str(neurons)
#	model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(neurons), random_state=1).fit(X_train, y_train)
#	predicted = model.predict(X_cross)
#
#	current_value = np.sum(predicted == y_cross)
#	if current_value > best_value:
#		best_value = current_value
#		best_number_of_neurons = [neurons]


#

# Normalize
X_train=((X_train-X_train.min())/(X_train.max()-X_train.min()))
X_test=((X_test-X_test.min())/(X_test.max()-X_test.min()))

model = MultinomialNB().fit(X_train, y_train)
predicted = model.predict(X_test)
final_value = 100.0 * np.sum(predicted == y_test) / len(y_test.index)

print(final_value)