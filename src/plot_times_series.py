#Encoding: UTF-8
import sys
import pandas as pd
#from lib.classes import *
#from lib.data_filter import *
#from lib.data_processor import *
#from lib.data_normalization import *
#from lib.training_path import *
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from joblib import dump, load
from math import sin, cos, sqrt, atan2, radians
import json

print('Inicializando...')
plot = True
if plot:
	with open('../data/correctness_percentage_rio_bus.json') as arq:
		correctness = json.load(arq)

	max_lenght = 100

	percentage_total = [0.0]*(max_lenght)
	percentage_1 = [0.0]*(max_lenght)
	percentage_2 = [0.0]*(max_lenght)
	percentage_3 = [0.0]*(max_lenght)
	percentage_4 = [0.0]*(max_lenght)
	
	for lenght in range(1, max_lenght):
		
		#try:
		correct_1 = correctness['correct_conv_nn_' + str(lenght) + '_1.0']
		correct_2 = correctness['correct_conv_nn_' + str(lenght) + '_2.0']
		correct_3 = correctness['correct_conv_nn_' + str(lenght) + '_3.0']
		correct_4 = correctness['correct_conv_nn_' + str(lenght) + '_4.0']

		total_1 = correctness['total_' + str(lenght) + '_1.0']
		total_2 = correctness['total_' + str(lenght) + '_2.0']
		total_3 = correctness['total_' + str(lenght) + '_3.0']
		total_4 = correctness['total_' + str(lenght) + '_4.0']

		total = total_1 + total_2 + total_3 + total_4
		correct = correct_1 + correct_2 + correct_3 + correct_4

		percentage_total[lenght] = 100 * float(correct)/float(total)

		percentage_1[lenght] = 100 * float(correct_1)/float(total_1)
		percentage_2[lenght] = 100 * float(correct_2)/float(total_2)
		percentage_3[lenght] = 100 * float(correct_3)/float(total_3)
		percentage_4[lenght] = 100 * float(correct_4)/float(total_4)
			
		#print('plotando...')

	plt.plot(percentage_total, marker='', label=u"CNN Rio Bus Data")
	plt.plot(percentage_1, marker='', label=u"CNN Rio Bus Data Linha 1")
	plt.plot(percentage_2, marker='', label=u"CNN Rio Bus Data Linha 2")
	plt.plot(percentage_3, marker='', label=u"CNN Rio Bus Data Linha 3")
	plt.plot(percentage_4, marker='', label=u"CNN Rio Bus Data Linha 4")
	plt.legend(loc='best')
	plt.xlabel(u'Porcentagem de completude do caminho')
	plt.ylabel(u'Porcentagem de acerto')
	plt.title(u'Porcentagem de acerto x Porcentagem de completude do caminho')
	plt.grid(True)
	plt.show()
	exit()

	with open('../data/correctness_percentage_all_ensembled.json') as arq:
		correctness = json.load(arq)
		
	max_lenght = 100

	percentage_svm = [0.0]*(max_lenght)
	percentage_nn = [0.0]*(max_lenght)
	percentage_dump = [0.0]*(max_lenght)
	percentage_rf = [0.0]*(max_lenght)
	percentage_ensembled_nn = [0.0]*(max_lenght)
	percentage_ensembled_rf = [0.0]*(max_lenght)
	percentage_ensembled_bayes = [0.0]*(max_lenght)
	percentage_ensembled_svm = [0.0]*(max_lenght)
	for lenght in range(max_lenght):
		
		try:
			percentage_svm[lenght] = 100 * float(correctness['correct_svm_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
			percentage_dump[lenght] = 100 * float(correctness['correct_dump_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
			percentage_rf[lenght] = 100 * float(correctness['correct_rf_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
			percentage_nn[lenght] = 100 * float(correctness['correct_conv_nn_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
			percentage_ensembled_nn[lenght] = 100 * float(correctness['correct_ensembled_nn_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
			percentage_ensembled_rf[lenght] = 100 * float(correctness['correct_ensembled_rf_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
			percentage_ensembled_bayes[lenght] = 100 * float(correctness['correct_ensembled_bayes_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
			percentage_ensembled_svm[lenght] = 100 * float(correctness['correct_ensembled_svm_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
		except:
			pass

	print('plotando...')

	plt.plot(percentage_dump, marker='', color='green', label=u"Algoritmo de Georreferência")
	plt.plot(percentage_nn, marker='', markerfacecolor='blue', label="Rede Neural Convolucional")
	plt.plot(percentage_svm, marker='', color='olive', label="SVM")
	plt.plot(percentage_rf, marker='', color='red', label="Random Forest")
	#plt.plot(percentage_ensembled_nn, marker='', color='yellow', label="Ensembled Learning NN")
	plt.plot(percentage_ensembled_rf, marker='', color='black', label="Ensembled Learning RF")
	#plt.plot(percentage_ensembled_svm, marker='', color='orange', label="Ensembled Learning SVM")
	#plt.plot(percentage_ensembled_bayes, marker='', color='purple', label="Ensembled Learning Bayes")
	plt.legend(loc='best')
	plt.xlabel(u'Porcentagem de completude do caminho')
	plt.ylabel(u'Porcentagem de acerto')
	plt.title(u'Porcentagem de acerto x Porcentagem de completude do caminho')
	plt.grid(True)
	plt.show()
	exit()

def calculate_total_distance(path_df):
	total_distance = 0
	last_row = path_df.iloc[0]
	for index, row in path_df.tail(-1).iterrows():
		total_distance += distance_between(last_row, row)
		last_row = row
	return total_distance

# Create the training path vector using the total distance as a divisor
def create_training_path(path_df, x_length):
	empty_coordinate = [0.0, 0.0, 0.0]
	total_distance = calculate_total_distance(path_df)
	path_size = len(path_df.index)
	cluster_distance = total_distance/path_size
	new_path = []
	path_index = 0

	for index in range(0, x_length):

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

paths_df = pd.read_csv('../data/media_paths_date.csv')

total = paths_df.index_path.max() + 1

minimum_path = int(paths_df.index_path.min())
maximum_path = int(paths_df.index_path.max()) + 1

sizes = [60]

for size in sizes:
	print('size:', size)
	count_int = 0
	correctness = {}
	model_rf = load('../data/random_forest_' + str(size) + '.joblib')

	max_lenght = 0

	print('Criando dados do grafico...')

	for path_id in range(minimum_path + 2, maximum_path, 32):

		print('path_id: ' + str(path_id))

		current_path_df = paths_df[paths_df.index_path == path_id]

		line = current_path_df.iloc[0].id_line
	
		for lenght in range(1, len(current_path_df.index) + 1, 2):

			if lenght > max_lenght:
				max_lenght = lenght

			current_train_df = pd.DataFrame([create_training_path(current_path_df.head(lenght), size)])

			predicted_rf = model_rf.predict(current_train_df)
		
			# Parse to percent 
			lenght = int(100*lenght/len(current_path_df.index) + 1)

			# Append the new number of total of this lenght and add if it was correct
			if correctness.get('total_' + str(lenght) + '_' + str(int(line))) == None:
				correctness['total_' + str(lenght) + '_' + str(int(line))] = 1
			
				if predicted_rf[0] == line:
					correctness['correct_rf_' + str(lenght) + '_' + str(int(line))] = 1
				else:
					correctness['correct_rf_' + str(lenght) + '_' + str(int(line))] = 0
			
			else:
				correctness['total_' + str(lenght) + '_' + str(int(line))] += 1
				
				if predicted_rf[0] == line:
					correctness['correct_rf_' + str(lenght) + '_' + str(int(line))] += 1


		if 100 * path_id / maximum_path > count_int:
			count_int += 1
			print(count_int)
	with open('../data/correct_size_' + str(size) + '.json', 'w') as fp:
		json.dump(correctness, fp)
exit()
print('Criando porcentagem...')

#max_lenght = 100
#percentage_svm = [0.0]*max_lenght
#for lenght in range(100):
#	try:
#		percentage_svm[lenght] = 100 * float(correctness['correct_svm_' + str(lenght)])/float(correctness['total_' + str(lenght)])
#	except:
#		print('algo de errado nao esta certo')
#
#print('plotando...')
#
#print(correctness)
#
with open('../data/correctness_percentage_all_2.json', 'w') as file:
	file.write(str(correctness))
#
#plt.plot(percentage_svm, marker='', color='olive', label="SVM")
#plt.legend(loc='upper left')
#plt.xlabel(u'Número de leituras')
#plt.ylabel(u'Porcentagem de acerto')
#plt.title(u'Porcentagem de acerto x Número de leituras para SVM')
#plt.grid(True)
#plt.show()

