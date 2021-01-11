#Encoding: UTF-8
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from keras.models import load_model
import matplotlib.pyplot as plt
from joblib import dump, load
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

def adjustTimeColumn(time):
	if time == 0:
		return 0
	return (time - lowest_time)/(hightest_time - lowest_time)

def adjustLatitudeColumn(latitude):
	if latitude == 0:
		return 0
	return (latitude - lowest_latitude)/(highest_latitude - lowest_latitude)

def adjustLongitudeColumn(longitude):
	if longitude == 0:
		return 0
	return (longitude - lowest_longitude)/(highest_longitude - lowest_longitude)

def adjustTrainDf(df):

	for column in df:
		# For time columns
		if column % 3 == 0:
			df[column] = df[column].apply(adjustTimeColumn)
		elif column % 3 == 1:
			df[column] = df[column].apply(adjustLatitudeColumn)
		elif column % 3 == 2:
			df[column] = df[column].apply(adjustLongitudeColumn)

	return df

def convertTo3D(df):
	time_train_df = []
	lat_train_df = []
	lon_train_df = []
	for line in range(len(df.index)):

		time_train_df.append(df.iloc[line][df.iloc[line].index % 3 == 0].to_numpy())
		lat_train_df.append(df.iloc[line][df.iloc[line].index % 3 == 1].to_numpy())
		lon_train_df.append(df.iloc[line][df.iloc[line].index % 3 == 2].to_numpy())
	new_df = np.array([np.array(time_train_df), np.array(lat_train_df), np.array(lon_train_df)])
	return new_df.reshape((new_df.shape[1], 60, 3))

def getMaxIndex(result):
	maxValue = 1
	for index in range(4):

		if result[0][index] > result[0][maxValue - 1]:
			maxValue = index + 1
	return maxValue

print('Inicializando...')
plot = False
if plot:
	with open('../data/correctness_percentage.json') as arq:
		correctness = json.load(arq)

	with open('../data/correctness_percentage_conv_nn_2.json') as arq:
		correctness_conv_nn = json.load(arq)

	with open('../data/correctness_percentage_svm.json') as arq:
		correctness_svm = json.load(arq)

	print(correctness)

	max_lenght = 101
	percentage_nn = [0.0]*max_lenght
	percentage_conv_nn = [0.0]*max_lenght
	percentage_rf = [0.0]*max_lenght
	percentage_dump = [0.0]*max_lenght
	percentage_svm = [0.0]*max_lenght
	for lenght in range(max_lenght):
		try:
			percentage_svm[lenght+1] = 100 * float(correctness_svm['correct_svm_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
			percentage_conv_nn[lenght+1] = 100 * float(correctness_conv_nn['correct_conv_nn_' + str(lenght)])/float(correctness_conv_nn['total_' + str(lenght)])
			percentage_rf[lenght+1] = 100 * float(correctness['correct_rf_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
			percentage_dump[lenght+1] = 100 * float(correctness['correct_dump_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
		except:
			pass

	print('plotando...')

	plt.plot(percentage_dump, marker='', color='green', label="Algoritmo de Proximidade")
	plt.plot(percentage_svm, marker='', markerfacecolor='blue', label="SVM")
	plt.plot(percentage_conv_nn, marker='', color='olive', label="Convolution NN")
	plt.plot(percentage_rf, marker='', color='red', label="Random Forest")
	plt.legend(loc='upper left')
	plt.xlabel(u'Porcentagem de completude do caminho')
	plt.ylabel(u'Porcentagem de acerto')
	plt.title(u'Porcentagem de acerto x Porcentagem de completude do caminho para diferentes algoritmos')
	plt.grid(True)
	plt.show()

	exit()

paths_df = pd.read_csv('../data/media_paths_date.csv')

total = paths_df.index_path.max() + 1
count_int = 0

minimum_path = int(paths_df.index_path.min())
maximum_path = int(paths_df.index_path.max()) + 1

correctness = {}
model_conv_nn = load_model('../data/best_nn_cross_rio_bus_data_augmented.h5')

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
		current_train_df = convertTo3D(current_train_df)
		predicted_conv_nn = getMaxIndex(model_conv_nn.predict(current_train_df))

		# Parse to percent 
		lenght = int(100*lenght/len(current_path_df.index) + 1)

		# Append the new number of total of this lenght and add if it was correct
		if correctness.get('total_' + str(lenght)) == None:
			correctness['total_' + str(lenght)] = 1
			
			if predicted_conv_nn == line:
				correctness['correct_conv_nn_' + str(lenght)] = 1
			else:
				correctness['correct_conv_nn_' + str(lenght)] = 0
			
		else:
			correctness['total_' + str(lenght)] += 1
			
			if predicted_conv_nn == line:
				correctness['correct_conv_nn_' + str(lenght)] += 1


	if 100 * path_id / maximum_path > count_int:
		count_int += 1
		print(count_int)

print('Criando porcentagem...')

max_lenght = 100
percentage_conv_nn = [0.0]*max_lenght
for lenght in range(100):
	try:
		percentage_conv_nn[lenght] = 100 * float(correctness['correct_conv_nn_' + str(lenght)])/float(correctness['total_' + str(lenght)])
	except:
		print('algo de errado nao esta certo')

print('plotando...')

print(correctness)

with open('../data/correctness_riobus_aumented.json', 'w') as file:
	json.dumb(file, correctness)

plt.plot(percentage_conv_nn, marker='', color='olive', label="Convolution NN")
plt.legend(loc='upper left')
plt.xlabel(u'Número de leituras')
plt.ylabel(u'Porcentagem de acerto')
plt.title(u'Porcentagem de acerto x Número de leituras para Convolution NN')
plt.grid(True)
plt.show()

