import pandas as pd
from lib.classes import *
from lib.data_filter import *
from lib.data_processor import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

df = pd.read_csv('../data/train_df.csv', header=None)

y_df = df[180]
X_df = df.drop(columns=[180])

X_train_all, X_test, y_train_all, y_test = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

#X_train, X_cross, y_train, y_cross = train_test_split(X_train_all, y_train_all, random_state=1, test_size=0.2, stratify=y_train_all)

#number_of_neurons1 = [500, 750, 1000, 1250, 1500, 2000]
#number_of_neurons2 = [50, 75, 100, 125, 150, 200]

best_value = 0
best_number_of_neurons = 0

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

#for neurons1 in number_of_neurons1:
#	for neurons2 in number_of_neurons2:
#
#		print 'Number of neurons 1: ' + str(neurons1)
#		print 'Number of neurons 2: ' + str(neurons2)
#		model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(neurons1, neurons2), random_state=1).fit(X_train, y_train)
#		predicted = model.predict(X_cross)
#
#		current_value = np.sum(predicted == y_cross)
#		if current_value > best_value:
#			best_value = current_value
#			best_number_of_neurons = [neurons1, neurons2]
#

print 'treinando'

model = SVC(gamma='auto').fit(X_train_all, y_train_all)

print 'treinado'
predicted = model.predict(X_test)
final_value = 100.0 * np.sum(predicted == y_test) / len(y_test.index)

print final_value
