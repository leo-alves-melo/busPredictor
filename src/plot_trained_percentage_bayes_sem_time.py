#Encoding: UTF-8
import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import pandas as pd
from lib.classes import *
from lib.data_filter import *
from lib.data_processor import *
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

train_df = pd.read_csv('../data/train_df.csv', header=None).iloc[::-1]

y_df = train_df[180]
X_df = train_df.drop(columns=[180])

X_train_all, X_test, y_train_all, y_test = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

X_train_all = X_train_all.abs()
X_test = X_test.abs()

# Normalizing X considering the absolute values of the coordinates
highest_latitude = 22.836006
lowest_latitude = 22.801396
highest_longitude = 47.095658
lowest_longitude = 47.046078
hightest_time = 86400
lowest_time = 0
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

# Remove the time from the df
def remove_time(df):
	return df.drop(columns=list(range(0, 180, 3)))

X_train_all = remove_time(adjustTrainDf(X_train_all))
X_test = remove_time(adjustTrainDf(X_test))

paths_df = pd.read_csv('../data/media_paths_date.csv').abs()

total = paths_df.index_path.max() + 1
count_int = 0

minimum_path = int(paths_df.index_path.min())
maximum_path = int(paths_df.index_path.max()) + 1

correctness = {}
model_bayes = ComplementNB().fit(X_train_all, y_train_all)

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
		current_train_df = remove_time(adjustTrainDf(current_train_df))
		predicted_bayes = model_bayes.predict(current_train_df)
		
		# Parse to percent 
		lenght = int(100*lenght/len(current_path_df.index) + 1)

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
plt.title(u'Porcentagem de acerto x Número de leituras para Complement Bayes')
plt.grid(True)
plt.show()

