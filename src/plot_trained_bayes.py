#Encoding: UTF-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from joblib import dump, load
import json
from math import sin, cos, sqrt, atan2, radians
import json


def calculate_total_distance(path_df):
	total_distance = 0
	last_row = path_df.iloc[0]
	for index, row in path_df.tail(-1).iterrows():
		total_distance += distance_between(last_row, row)
		last_row = row
	return total_distance

# Create the training path vector using the total distance as a divisor
def create_training_path(path_df):
	empty_coordinate = [0.0, 0.0, 0.0]
	total_distance = calculate_total_distance(path_df)
	path_size = len(path_df.index)
	cluster_distance = total_distance/path_size
	new_path = []
	path_index = 0

	for index in range(0, 60):

		if path_index >= path_size:
			new_path += empty_coordinate
			continue

		current_distance = 0
		cluster = pd.DataFrame()
		last_row = path_df.iloc[path_index]
		cluster = cluster.append(last_row)
		path_index += 1

		if path_index >= path_size:
			new_path += [cluster.iloc[0].date, cluster.iloc[0].latitude, cluster.iloc[0].longitude]
			continue

		while current_distance < cluster_distance:
			current_row = path_df.iloc[path_index]
			current_distance += distance_between(last_row, current_row)
			cluster = cluster.append(current_row)
			path_index += 1
			last_row = current_row

			if path_index >= path_size:
				break

		mean_cluster = cluster.mean()
		cluster_coordinate = [mean_cluster.date, mean_cluster.latitude, mean_cluster.longitude]

		new_path += cluster_coordinate

	return new_path

def distance_between(position1, position2):
	# approximate radius of earth in km
	R = 6373.0

	lat1 = radians(position1.latitude)
	lon1 = radians(position1.longitude)
	lat2 = radians(position2.latitude)
	lon2 = radians(position2.longitude)

	dlon = lon2 - lon1
	dlat = lat2 - lat1
	
	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))
	
	distance = R * c
	
	return distance

print('Inicializando...')

with open('../data/correctness_bayes.json') as arq:
	correctness = json.load(arq)
#
#print(correctness['correct_rf_1'])
#max_lenght = 350
#percentage_nn = [0.0]*max_lenght
percentage_bayes = [0.0]*100
#percentage_rf = [0.0]*max_lenght
#percentage_dump = [0.0]*max_lenght
for lenght in range(1, 100+1):
#	try:
#		percentage_nn[lenght-1] = 100 * float(correctness['correct_nn_' + str(lenght)])/float(correctness['total_' + str(lenght)])
	percentage_bayes[lenght-1] = 100 * float(correctness['correct_bayes_' + str(lenght + 100)])/float(correctness['total_' + str(lenght + 100)])
#		percentage_rf[lenght-1] = 100 * float(correctness['correct_rf_' + str(lenght)])/float(correctness['total_' + str(lenght)])
#		percentage_dump[lenght-1] = 100 * float(correctness['correct_dump_' + str(lenght)])/float(correctness['total_' + str(lenght)])
#	except:
#		break
#
#print('plotando...')
#
#plt.plot(percentage_nn, marker='', markerfacecolor='blue', label="Rede Neural")
plt.plot(percentage_bayes, marker='', color='olive', label="Bayes")
#plt.plot(percentage_rf, marker='', color='red', label="Random Forest")
#plt.plot(percentage_dump, marker='', color='green', label="Algoritmo Dump")
plt.legend(loc='upper left')
plt.xlabel(u'Número de leituras')
plt.ylabel(u'Porcentagem de acerto')
plt.title(u'Porcentagem de acerto x Número de leituras para diferentes algoritmos')
plt.grid(True)
plt.show()

exit()

train_df = pd.read_csv('../data/train_df.csv', header=None).abs()

y_df = train_df[180]
X_df = train_df.drop(columns=[180])

X_train_all, X_test, y_train_all, y_test = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

#X_train_all = ((X_train_all-X_train_all.min())/(X_train_all.max()-X_train_all.min()))
#X_test = ((X_test-X_test.min())/(X_test.max()-X_test.min()))
#X_train, X_cross, y_train, y_cross = train_test_split(X_train_all, y_train_all, random_state=1, test_size=0.2, stratify=y_train_all)

paths_df = pd.read_csv('../data/media_paths_date.csv').abs()

total = paths_df.index_path.max() + 1
count_int = 0

minimum_path = int(paths_df.index_path.min())
maximum_path = int(paths_df.index_path.max()) + 1

print('Fitando')

correctness = {}
model_bayes = MultinomialNB().fit(X_train_all, y_train_all)

max_lenght = 100

print('Criando dados do grafico...')

for path_id in range(minimum_path + 1, maximum_path, 32):

	print('path_id: ' + str(path_id))

	current_path_df = paths_df[paths_df.index_path == path_id]

	line = current_path_df.iloc[0].id_line
	
	for lenght in range(1, len(current_path_df.index) + 1):

		current_train_df = pd.DataFrame([create_training_path(current_path_df.head(lenght))])
		predicted_bayes = model_bayes.predict(current_train_df)

		lenght = int((lenght/len(current_path_df.index) + 1) * 100) 
		
		# Append the new number of total of this lenght and add if it was correct
		if correctness.get('total_' + str(lenght)) == None:
			correctness['total_' + str(lenght)] = 1
			
			if predicted_bayes[0] == line:
				correctness['correct_bayes_' + str(lenght)] = 1
			else:
				correctness['correct_bayes_' + str(lenght)] = 0
		else:
			correctness['total_' + str(lenght)] += 1
			if predicted_bayes[0] == line:
				correctness['correct_bayes_' + str(lenght)] += 1


	if 100 * path_id / maximum_path > count_int:
		count_int += 1
		print(count_int)

print('Criando porcentagem...')

percentage_bayes = [0.0]*max_lenght
for lenght in range(max_lenght-1):
	try:
		percentage_bayes[lenght] = 100 * float(correctness['correct_bayes_' + str(lenght)])/float(correctness['total_' + str(lenght)])
	except:
		print('algo de errado nao esta certo')

print('plotando...')

with open('../data/correctness_bayes.json', 'w') as file:
	json.dump(correctness, file)

plt.plot(percentage_bayes, marker='', color='olive', label="Bayes")
plt.legend(loc='lower right')
plt.xlabel(u'Número de leituras')
plt.ylabel(u'Porcentagem de acerto')
plt.title(u'Porcentagem de acerto x Número de leituras para diferentes algoritmos')
plt.show()