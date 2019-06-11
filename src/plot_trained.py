#Encoding: UTF-8

import pandas as pd
from lib.classes import *
from lib.data_filter import *
from lib.data_processor import *
from lib.training_path import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from joblib import dump, load

print('Inicializando...')

train_df = pd.read_csv('../data/train_df.csv', header=None)

y_df = train_df[180]
X_df = train_df.drop(columns=[180])

X_train_all, X_test, y_train_all, y_test = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

X_train, X_cross, y_train, y_cross = train_test_split(X_train_all, y_train_all, random_state=1, test_size=0.2, stratify=y_train_all)

paths_df = pd.read_csv('../data/media_paths_date.csv')

total = paths_df.index_path.max() + 1
count_int = 0

minimum_path = int(paths_df.index_path.min())
maximum_path = int(paths_df.index_path.max()) + 1

correctness = {}
max_lenght = 0
model = load('../data/best_nn.joblib') 

print('Criando dados do grafico...')

for path_id in range(minimum_path + 1, maximum_path, 64):

	print('path_id: ' + str(path_id))

	current_path_df = paths_df[paths_df.index_path == path_id]

	line = current_path_df.iloc[0].id_line
	
	for lenght in range(1, len(current_path_df.index) + 1):

		if lenght > max_lenght:
			max_lenght = lenght

		current_train_df = pd.DataFrame([create_training_path(current_path_df.head(lenght))])
		predicted = model.predict(current_train_df)
		# Append the new number of total of this lenght and add if it was correct
		if correctness.get('total_' + str(lenght)) == None:
			correctness['total_' + str(lenght)] = 1
			if predicted[0] == line:
				correctness['correct_' + str(lenght)] = 1
			else:
				correctness['correct_' + str(lenght)] = 0
		else:
			correctness['total_' + str(lenght)] += 1
			if predicted[0] == line:
				correctness['correct_' + str(lenght)] += 1


	if 100 * path_id / maximum_path > count_int:
		count_int += 1
		print(count_int)

print('Criando porcentagem...')

max_lenght += 1
percentage = [0.0]*max_lenght
for lenght in range(max_lenght):
	try:
		percentage[lenght] = 100 * float(correctness['correct_' + str(lenght)])/float(correctness['total_' + str(lenght)])
	except:
		pass

print('plotando...')

plt.plot(percentage)
plt.xlabel(u'Número de leituras')
plt.ylabel(u'Porcentagem de acerto')
plt.title(u'Porcentagem de acerto x Número de leituras para Rede Neural')
plt.show()

