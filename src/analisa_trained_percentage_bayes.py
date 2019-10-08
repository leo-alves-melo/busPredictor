#Encoding: UTF-8
import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import pandas as pd
from lib.classes import *
from lib.data_filter import *
from lib.data_processor import *
from lib.training_path import *
from lib.data_visualization import *
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

	with open('../data/correctness_percentage_bayes_multinomial_reversed.json') as arq:
		correctness_bayes = json.load(arq)

	print(correctness)

	max_lenght = 100
	percentage_nn = [0.0]*max_lenght
	percentage_bayes = [0.0]*max_lenght
	percentage_rf = [0.0]*max_lenght
	percentage_dump = [0.0]*max_lenght
	for lenght in range(max_lenght):
		try:
			#percentage_nn[lenght] = 100 * float(correctness['correct_nn_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
			percentage_bayes[lenght] = 100 * float(correctness_bayes['correct_bayes_' + str(lenght+1)])/float(correctness_bayes['total_' + str(lenght+1)])
			#percentage_rf[lenght] = 100 * float(correctness['correct_rf_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
			#percentage_dump[lenght] = 100 * float(correctness['correct_dump_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
		except:
			pass

	print('plotando...')

	#plt.plot(percentage_dump, marker='', color='green', label="Algoritmo de Proximidade")
	#plt.plot(percentage_nn, marker='', markerfacecolor='blue', label="Rede Neural")
	plt.plot(percentage_bayes, marker='', color='olive', label="Bayes")
	#plt.plot(percentage_rf, marker='', color='red', label="Random Forest")
	plt.legend(loc='upper left')
	plt.xlabel(u'Porcentagem de completude do caminho')
	plt.ylabel(u'Porcentagem de acerto')
	plt.title(u'Porcentagem de acerto x Porcentagem de completude do caminho para ComplementNB')
	plt.grid(True)
	plt.show()

	exit()

train_df = pd.read_csv('../data/train_df.csv', header=None)

y_df = train_df[180]
X_df = train_df.drop(columns=[180])

X_train_all, X_test, y_train_all, y_test = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

X_train_all = X_train_all.abs()
X_test = X_test.abs()

paths_df = pd.read_csv('../data/media_paths_date.csv')

total = paths_df.index_path.max() + 1
count_int = 0

minimum_path = int(paths_df.index_path.min())
maximum_path = int(paths_df.index_path.max()) + 1

correctness = {}
model_bayes = ComplementNB().fit(X_train_all, y_train_all)

max_lenght = 0

print('Criando dados do grafico...')

wrong_80_to_100 = []

for path_id in range(minimum_path + 1, maximum_path, 32):

	print('path_id: ' + str(path_id))

	current_path_df = paths_df[paths_df.index_path == path_id]

	line = current_path_df.iloc[0].id_line
	
	for lenght in range(1, len(current_path_df.index) + 1, 2):

		if lenght > max_lenght:
			max_lenght = lenght

		current_train_df = pd.DataFrame([create_training_path(current_path_df.head(lenght))]).abs()
		predicted_bayes = model_bayes.predict(current_train_df)
		# Parse to percent 
		lenght = int(100*lenght/len(current_path_df.index) + 1)

		# Append the new number of total of this lenght and add if it was correct
		if correctness.get('total_' + str(path_id) + '_' + str(lenght)) == None:
			correctness['total_' + str(path_id) + '_' + str(lenght)] = 1
			
			if predicted_bayes[0] == line:
				correctness['correct_bayes_' + str(path_id) + '_' + str(lenght)] = 1
			else:
				correctness['correct_bayes_' + str(path_id) + '_' + str(lenght)] = 0
			
		else:
			correctness['total_' + str(path_id) + '_' + str(lenght)] += 1
			
			if predicted_bayes[0] == line:
				correctness['correct_bayes_' + str(path_id) + '_' + str(lenght)] += 1

	print('vreificando...')
	close_to_80 = None
	index_80 = 80
	while close_to_80 == None:
		close_to_80 = correctness.get('correct_bayes_' + str(path_id) + '_' + str(index_80))
		index_80 += 1
	
	close_to_100 = None
	index_100 = 100
	while close_to_100 == None:
		close_to_100 = correctness.get('correct_bayes_' + str(path_id) + '_' + str(index_100))
		index_100 -= 1
	 
	print(close_to_80)
	print(close_to_100)
	
	if close_to_80 > 0 and close_to_100 == 0:
		print('Acheeeeeeeeei')
		print(path_id)
		wrong_80_to_100.append(path_id)
	if 100 * path_id / maximum_path > count_int:
		count_int += 1
		print(count_int)

	correctness = {}

print(wrong_80_to_100)
with open('../data/wrong_80_to_100.txt', 'w') as arq:
	arq.write(str(wrong_80_to_100))

print('Criando porcentagem...')

max_lenght = 100
percentage_bayes = [0.0]*max_lenght
for lenght in range(100):
	try:
		percentage_bayes[lenght] = 100 * float(correctness['correct_bayes_' + str(lenght)])/float(correctness['total_' + str(lenght)])
	except:
		print('algo de errado nao esta certo')

print('plotando...')

print(correctness)

with open('../data/correctness_percentage_bayes_multinomial_reversed.json', 'w') as file:
	file.write(str(correctness))

plt.plot(percentage_bayes, marker='', color='olive', label="Bayes")
plt.legend(loc='upper left')
plt.xlabel(u'Número de leituras')
plt.ylabel(u'Porcentagem de acerto')
plt.title(u'Porcentagem de acerto x Número de leituras para Multinomial Bayes')
plt.grid(True)
plt.show()

