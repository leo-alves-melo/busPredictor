#Encoding: UTF-8
import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import pandas as pd
from lib.classes import *
from lib.data_filter import *
from lib.data_processor import *
from lib.data_normalization import *
from lib.training_path import *
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from joblib import dump, load
import json

print('Inicializando...')
plot = False
if plot:
	with open('../data/correctness_percentage.json') as arq:
		correctness = json.load(arq)

	with open('../data/correctness_percentage_svm.json') as arq:
		correctness_svm = json.load(arq)

	print(correctness)

	max_lenght = 100
	percentage_svm = [0.0]*max_lenght
	for lenght in range(max_lenght):
		try:
			percentage_svm[lenght] = 100 * float(correctness_svm['correct_svm_' + str(lenght+1)])/float(correctness_svm['total_' + str(lenght+1)])
		except:
			pass

	print('plotando...')

	#plt.plot(percentage_dump, marker='', color='green', label="Algoritmo de Proximidade")
	#plt.plot(percentage_nn, marker='', markerfacecolor='blue', label="Rede Neural")
	plt.plot(percentage_svm, marker='', color='olive', label="SVM")
	#plt.plot(percentage_rf, marker='', color='red', label="Random Forest")
	plt.legend(loc='upper left')
	plt.xlabel(u'Porcentagem de completude do caminho')
	plt.ylabel(u'Porcentagem de acerto')
	plt.title(u'Porcentagem de acerto x Porcentagem de completude do caminho para SVM')
	plt.grid(True)
	plt.show()

	exit()

train_df = pd.read_csv('../data/train_df.csv', header=None).iloc[::-1]

y_df = train_df[180]
X_df = train_df.drop(columns=[180])

X_train_all, X_test, y_train_all, y_test = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

X_train_all = X_train_all.abs()
X_test = X_test.abs()

X_train_all = adjustTrainDf(X_train_all)
X_test = adjustTrainDf(X_test)

paths_df = pd.read_csv('../data/media_paths_date.csv').abs()

total = paths_df.index_path.max() + 1
count_int = 0

minimum_path = int(paths_df.index_path.min())
maximum_path = int(paths_df.index_path.max()) + 1

correctness = {}
model_svm = load('../data/best_svm.joblib')

max_lenght = 0

print('Criando dados do grafico...')

for path_id in range(minimum_path + 1, maximum_path, 32):

	print('path_id: ' + str(path_id))

	current_path_df = paths_df[paths_df.index_path == path_id]

	line = current_path_df.iloc[0].id_line
	
	for lenght in range(1, len(current_path_df.index) + 1, 2):

		if lenght > max_lenght:
			max_lenght = lenght

		current_train_df = pd.DataFrame([create_training_path(current_path_df.head(lenght))])
		current_train_df = adjustTrainDf(current_train_df)
		predicted_svm = model_svm.predict(current_train_df)
		
		# Parse to percent 
		lenght = int(100*lenght/len(current_path_df.index) + 1)

		# Append the new number of total of this lenght and add if it was correct
		if correctness.get('total_' + str(lenght)) == None:
			correctness['total_' + str(lenght)] = 1
			
			if predicted_svm[0] == line:
				correctness['correct_svm_' + str(lenght)] = 1
			else:
				correctness['correct_svm_' + str(lenght)] = 0
			
		else:
			correctness['total_' + str(lenght)] += 1
			
			if predicted_svm[0] == line:
				correctness['correct_svm_' + str(lenght)] += 1


	if 100 * path_id / maximum_path > count_int:
		count_int += 1
		print(count_int)

print('Criando porcentagem...')

max_lenght = 100
percentage_svm = [0.0]*max_lenght
for lenght in range(100):
	try:
		percentage_svm[lenght] = 100 * float(correctness['correct_svm_' + str(lenght)])/float(correctness['total_' + str(lenght)])
	except:
		print('algo de errado nao esta certo')

print('plotando...')

print(correctness)

with open('../data/correctness_percentage_svm.json', 'w') as file:
	file.write(str(correctness))

plt.plot(percentage_svm, marker='', color='olive', label="SVM")
plt.legend(loc='upper left')
plt.xlabel(u'Número de leituras')
plt.ylabel(u'Porcentagem de acerto')
plt.title(u'Porcentagem de acerto x Número de leituras para SVM')
plt.grid(True)
plt.show()

