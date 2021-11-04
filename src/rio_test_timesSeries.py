#Encoding: UTF-8
import sys
import pandas as pd
import numpy as np
from lib.classes import *
from lib.rio_data_filter import *
from lib.training_path import *
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from joblib import dump, load
from math import sin, cos, sqrt, atan2, radians
import math
import json
from statsmodels.tsa.vector_ar.var_model import VAR

print('Inicializando...')
plot = False
if plot:
	with open('data/correctness_percentage_all_ensembled.json') as arq:
		correctness = json.load(arq)

	max_lenght = 100
	for line in range(1, 5):

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
		plt.clf()

	exit()

def create_model(path, id_line, dict):
    line_path = path[path.line == id_line].drop(['line'], axis=1)
    model = VAR(line_path).fit()
    dict[id_line] = model


small_riobus = '../data/rio_paths_10.csv'
paths_df = pd.read_csv(small_riobus, parse_dates=['datetime'], dtype={'line': object})
paths_df = paths_df.drop(['order'], axis=1)
paths_df.datetime = ((paths_df.datetime - paths_df.datetime.dt.normalize()) / pd.Timedelta('1 second')).astype(int)

print(paths_df)

paths_train_df = paths_df[paths_df.index_path % 4 == 0]
paths_train_df = paths_train_df.drop(['index_path'], axis=1)
paths_test_df = paths_df[paths_df.index_path % 4 == 3]

models = {}

for line in paths_train_df.line.unique():
    create_model(paths_train_df, line, models)

print(models)

exit()

count_int = 0
correctness = {}
model_rf = load('../data/random_forest_times_seriesFull.joblib')

max_lenght = 0

print('Criando dados do grafico...')

for path_id in range(minimum_path + 2, maximum_path, 32):

	print('path_id: ' + str(path_id))

	current_path_df = paths_df[paths_df.index_path == path_id]

	line = current_path_df.iloc[0].id_line
	total_dist_1 = 0
	total_dist_2 = 0
	total_dist_3 = 0
	total_dist_4 = 0
	for lenght in range(1, len(current_path_df.index) - 1, 2):

		if lenght > max_lenght:
			max_lenght = lenght
		
		next_line = np.array(current_path_df.drop(columns=['id_line', 'index_path']).iloc[lenght + 1])
		#print(current_path_df)
		array_route = np.array(current_path_df.head(lenght).drop(columns=['id_line', 'index_path']))
		#print(array_route)
		current_train_df = pd.DataFrame([create_training_path(current_path_df.head(lenght), size)])

		prediction_1 = model_1.forecast(array_route, steps=1)[0]
		prediction_2 = model_2.forecast(array_route, steps=1)[0]
		prediction_3 = model_3.forecast(array_route, steps=1)[0]
		prediction_4 = model_4.forecast(array_route, steps=1)[0]

		# Define closest prediction
		dist_1 = distance_between_literal(prediction_1[1], prediction_1[2], next_line[1], next_line[2])
		dist_2 = distance_between_literal(prediction_2[1], prediction_2[2], next_line[1], next_line[2])
		dist_3 = distance_between_literal(prediction_3[1], prediction_3[2], next_line[1], next_line[2])
		dist_4 = distance_between_literal(prediction_4[1], prediction_4[2], next_line[1], next_line[2])

		# Use time as distance too, centering using the value of the day
		next_line_time = next_line[0]/(24*60*60)**2
		dist_1 = math.sqrt((prediction_1[0]/(24*60*60) - next_line_time)**2 + dist_1**2)
		dist_2 = math.sqrt((prediction_2[0]/(24*60*60) - next_line_time)**2 + dist_2**2)
		dist_3 = math.sqrt((prediction_3[0]/(24*60*60) - next_line_time)**2 + dist_3**2)
		dist_4 = math.sqrt((prediction_4[0]/(24*60*60) - next_line_time)**2 + dist_4**2)

		total_dist_1 += dist_1
		total_dist_2 += dist_2
		total_dist_3 += dist_3
		total_dist_4 += dist_4

		current = current_train_df.values[0]
		
		inputValue = np.array([np.concatenate((prediction_1, prediction_2, prediction_3, prediction_4, np.array([total_dist_1, total_dist_2, total_dist_3, total_dist_4]), current), axis=None)])
		
		predicted_rf = model_rf.predict(inputValue)
	
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
with open('../data/correct_times_seriesFull.json', 'w') as fp:
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

