#Encoding: UTF-8
import sys
import pandas as pd
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
plot = True
if plot:
	with open('../data/correctness_percentage_all_ensembled.json') as arq:
		correctness = json.load(arq)
	with open('../data/correctness_bayes.json') as arq:
		correctness_bayes = json.load(arq)
	with open('../data/correct_times_series.json') as arq:
		correctness_time_series = json.load(arq)
	with open('../data/correctness_riobus_completed.json') as arq:
		correctness_rio_bus = json.load(arq)

	max_lenght = 100
	percentage_svm = [0.0]*(max_lenght)
	percentage_nn = [0.0]*(max_lenght)
	percentage_dump = [0.0]*(max_lenght)
	percentage_rf = [0.0]*(max_lenght)
	percentage_ensembled_rf = [0.0]*(max_lenght)
	percentage_bayes = [0.0]*(max_lenght)
	percentage_time_series = [0.0]*(max_lenght)
	percentage_riobus = [0.0]*(max_lenght)
	percentage_manual = [97.86]*(max_lenght)
	for lenght in range(max_lenght):
		percentage_svm[lenght] = 100 * float(correctness['correct_svm_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
		percentage_dump[lenght] = 100 * float(correctness['correct_dump_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
		percentage_rf[lenght] = 100 * float(correctness['correct_rf_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
		percentage_nn[lenght] = 100 * float(correctness['correct_conv_nn_' + str(lenght+1)])/float(correctness['total_' + str(lenght+1)])
		percentage_bayes[lenght] = 100 * float(correctness_bayes['correct_bayes_' + str(lenght+100)])/float(correctness_bayes['total_' + str(lenght+100)])
		percentage_time_series[lenght] = 100 * float(       correctness_time_series['correct_rf_' + str(lenght+1) + "_1"] + correctness_time_series['correct_rf_' + str(lenght+1) + "_2"] + correctness_time_series['correct_rf_' + str(lenght+1) + "_3"] + correctness_time_series['correct_rf_' + str(lenght+1) + "_4"] )/float( correctness_time_series['total_' + str(lenght+1) + "_1"] + correctness_time_series['total_' + str(lenght+1) + "_2"] + correctness_time_series['total_' + str(lenght+1) + "_3"] + correctness_time_series['total_' + str(lenght+1) + "_4"]  )
		percentage_riobus[lenght] = 100 * float(correctness_rio_bus['correct_conv_nn_' + str(lenght+1)])/float(correctness_rio_bus['total_' + str(lenght+1)])

	print('plotando...')

	#plt.plot(percentage_manual, marker='', color='black', label=u"Configuração manual")
	plt.plot(percentage_dump, marker='', color='green', label=u"Alg. Georreferência")
	plt.plot(percentage_nn, marker='', markerfacecolor='blue', label=u"RNC unidimensional")
	plt.plot(percentage_svm, marker='', color='olive', label="SVM")
	plt.plot(percentage_rf, marker='', color='red', label="Floresta Aleatória")
	plt.plot(percentage_bayes, marker='', color='brown', label="Naïve Bayes")
	plt.plot(percentage_time_series, marker='', color='orange', label=u"Séries Temporais")
	plt.plot(percentage_riobus, marker='', color='purple', label="RNC do RioBus")
	plt.legend(loc='best')
	plt.xlabel(u'Porcentagem de completude do trajeto', fontsize=14)
	plt.ylabel(u'Acurácia', fontsize=14)
	plt.title(u'Acurácia x Porcentagem de completude do trajeto', fontsize=16)
	plt.grid(True)
	plt.show()
	plt.clf()

	exit()

paths_df = pd.read_csv('../data/media_paths_date.csv')

total = paths_df.index_path.max() + 1
count_int = 0

minimum_path = int(paths_df.index_path.min())
maximum_path = int(paths_df.index_path.max()) + 1

correctness = {}
model_rf = load('../data/best_rf.joblib')
model_svm = load('../data/best_svm.joblib')
model_dump = DumpAlgorithm()

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
		predicted_dump = model_dump.predict(current_path_df.head(lenght))
		predicted_rf = model_rf.predict(current_train_df)
		predicted_svm = model_svm.predict(adjustTrainDf(current_train_df.abs()))
		
		# Parse to percent 
		lenght = int(100*lenght/len(current_path_df.index) + 1)

		# Append the new number of total of this lenght and add if it was correct
		if correctness.get('total_' + str(lenght) + '_' + str(line)) == None:
			correctness['total_' + str(lenght) + '_' + str(line)] = 1
			
			if predicted_svm[0] == line:
				correctness['correct_svm_' + str(lenght) + '_' + str(line)] = 1
			else:
				correctness['correct_svm_' + str(lenght) + '_' + str(line)] = 0
			if predicted_dump == line:
				correctness['correct_dump_' + str(lenght) + '_' + str(line)] = 1
			else:
				correctness['correct_dump_' + str(lenght) + '_' + str(line)] = 0
			if predicted_rf[0] == line:
				correctness['correct_rf_' + str(lenght) + '_' + str(line)] = 1
			else:
				correctness['correct_rf_' + str(lenght) + '_' + str(line)] = 0
			
		else:
			correctness['total_' + str(lenght) + '_' + str(line)] += 1
			
			if predicted_svm[0] == line:
				correctness['correct_svm_' + str(lenght) + '_' + str(line)] += 1
			if predicted_dump == line:
				correctness['correct_dump_' + str(lenght) + '_' + str(line)] += 1
			if predicted_rf[0] == line:
				correctness['correct_rf_' + str(lenght) + '_' + str(line)] += 1


	if 100 * path_id / maximum_path > count_int:
		count_int += 1
		print(count_int)

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

