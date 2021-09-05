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
import keras

print('Inicializando...')
plot = True
if plot:
	sizes = [ 5, 15, 30, 45, 60, 90, 120, 180]

	for size in sizes:
		with open('../data/correct_size_' + str(size) + '.json') as arq:
			correctness = json.load(arq)
		
		max_lenght = 100
		percentage_total_rf = [0.0]*(max_lenght)
		for lenght in range(1, max_lenght):

			

			correct_1_rf = correctness['correct_rf_' + str(lenght) + '_1']
			correct_2_rf = correctness['correct_rf_' + str(lenght) + '_2']
			correct_3_rf = correctness['correct_rf_' + str(lenght) + '_3']
			correct_4_rf = correctness['correct_rf_' + str(lenght) + '_4']

			total_1_rf = correctness['total_' + str(lenght) + '_1']
			total_2_rf = correctness['total_' + str(lenght) + '_2']
			total_3_rf = correctness['total_' + str(lenght) + '_3']
			total_4_rf = correctness['total_' + str(lenght) + '_4']

			total_rf = total_1_rf + total_2_rf + total_3_rf + total_4_rf
			correct_rf = correct_1_rf + correct_2_rf + correct_3_rf + correct_4_rf

			percentage_total_rf[lenght] = 100 * float(correct_rf)/float(total_rf)

		plt.plot(percentage_total_rf, marker='', label=u"Janela " + str(size))
	sizes = []
	for size in sizes:
		with open('../data/correct_size_' + str(size) + '_nn_svm.json') as arq:
			correctness = json.load(arq)

		max_lenght = 100

		percentage_total_svm = [0.0]*(max_lenght)
		percentage_total_nn = [0.0]*(max_lenght)
		
		
		for lenght in range(1, max_lenght):
			
			#try:
			correct_1_svm = correctness['correct_svm_' + str(lenght) + '_1']
			correct_2_svm = correctness['correct_svm_' + str(lenght) + '_2']
			correct_3_svm = correctness['correct_svm_' + str(lenght) + '_3']
			correct_4_svm = correctness['correct_svm_' + str(lenght) + '_4']

			total_1_svm = correctness['total_' + str(lenght) + '_1']
			total_2_svm = correctness['total_' + str(lenght) + '_2']
			total_3_svm = correctness['total_' + str(lenght) + '_3']
			total_4_svm = correctness['total_' + str(lenght) + '_4']

			total_svm = total_1_svm + total_2_svm + total_3_svm + total_4_svm
			correct_svm = correct_1_svm + correct_2_svm + correct_3_svm + correct_4_svm

			percentage_total_svm[lenght] = 100 * float(correct_svm)/float(total_svm)



			correct_1_nn = correctness['correct_nn_' + str(lenght) + '_1']
			correct_2_nn = correctness['correct_nn_' + str(lenght) + '_2']
			correct_3_nn = correctness['correct_nn_' + str(lenght) + '_3']
			correct_4_nn = correctness['correct_nn_' + str(lenght) + '_4']

			total_1_nn = correctness['total_' + str(lenght) + '_1']
			total_2_nn = correctness['total_' + str(lenght) + '_2']
			total_3_nn = correctness['total_' + str(lenght) + '_3']
			total_4_nn = correctness['total_' + str(lenght) + '_4']

			total_nn = total_1_nn + total_2_nn + total_3_nn + total_4_nn
			correct_nn = correct_1_nn + correct_2_nn + correct_3_nn + correct_4_nn

			percentage_total_nn[lenght] = 100 * float(correct_nn)/float(total_nn)
				
			#print('plotando...')

		plt.plot(percentage_total_svm, marker='', label=u"Janela " + str(size) + " SVM")
		plt.plot(percentage_total_nn, marker='', label=u"Janela " + str(size) + " RNC")
		
	plt.legend(loc='best')
	plt.xlabel(u'Porcentagem de completude do trajeto', fontsize=14)
	plt.ylabel(u'Acurácia', fontsize=14)
	plt.title(u'Acurácia x Porcentagem de completude do trajeto', fontsize=16)
	plt.grid(True)
	plt.show()
	plt.clf()

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

def getMaxIndex(result):
	maxValue = 1
	for index in range(4):

		if result[0][index] > result[0][maxValue - 1]:
			maxValue = index + 1
	return maxValue

paths_df = pd.read_csv('../data/media_paths_date.csv')

total = paths_df.index_path.max() + 1

minimum_path = int(paths_df.index_path.min())
maximum_path = int(paths_df.index_path.max()) + 1

sizes = [45, 90]

for size in sizes:
	print('size:', size)
	count_int = 0
	correctness = {}
	#model_rf = load('../data/random_forest_' + str(size) + '.joblib')
	model_svm = load('../data/best_svm_' + str(size) + '.joblib')
	model_nn = keras.models.load_model('../data/best_cnn_' + str(size))

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

			predicted_svm = model_svm.predict(current_train_df)
			predicted_nn = getMaxIndex(model_nn.predict(current_train_df))

			print(line, predicted_svm, predicted_nn)


			# Parse to percent 
			lenght = int(100*lenght/len(current_path_df.index) + 1)

			# Append the new number of total of this lenght and add if it was correct
			if correctness.get('total_' + str(lenght) + '_' + str(int(line))) == None:
				correctness['total_' + str(lenght) + '_' + str(int(line))] = 1
			
				if predicted_svm[0] == line:
					correctness['correct_svm_' + str(lenght) + '_' + str(int(line))] = 1
				else:
					correctness['correct_svm_' + str(lenght) + '_' + str(int(line))] = 0

				if predicted_nn == line:
					correctness['correct_nn_' + str(lenght) + '_' + str(int(line))] = 1
				else:
					correctness['correct_nn_' + str(lenght) + '_' + str(int(line))] = 0
			
			else:
				correctness['total_' + str(lenght) + '_' + str(int(line))] += 1
				
				if predicted_svm[0] == line:
					correctness['correct_svm_' + str(lenght) + '_' + str(int(line))] += 1
				if predicted_nn == line:
					correctness['correct_nn_' + str(lenght) + '_' + str(int(line))] += 1


		if 100 * path_id / maximum_path > count_int:
			count_int += 1
			print(count_int)
	with open('../data/correct_size_' + str(size) + '_nn_svm.json', 'w') as fp:
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

