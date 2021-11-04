# encoding: utf-8

from matplotlib.pyplot import axis
from pandas.core.frame import DataFrame
from lib.classes import *
from lib.data_visualization import *
from lib.rio_data_filter import *
import pandas as pd
import matplotlib.pyplot as plt
from lib.training_path import *
from joblib import dump, load
import json
from sklearn import preprocessing

def compare_line(model, path, method_to_apply, line, percentage, corrects, name):
	predicted = model.predict(method_to_apply(path))
	key = name + "_" + str(percentage)
	if predicted[0] == line:
		if corrects.get(key) == None:
			corrects[key] = 1
		else:
			corrects[key] += 1
	else:
		if corrects.get(key) == None:
			corrects[key] = 0	

def test_all(df, models):
	# test will be indexed in 3, 7, 11...

	lines = df.line.unique()
	jump = 64

	corrects = {}

	for line in lines:
		print('line:', line)
		current_df = df[df.line == line]

		possible_index_paths = current_df.index_path.unique()
		total = possible_index_paths[-1] - possible_index_paths[0]
		count = 0
		for index_path in range(possible_index_paths[3], possible_index_paths[-1], jump):
			
			count += (1*jump)
			current_path = current_df[current_df.index_path == index_path]
			
			path_lenght = len(current_path.index)

			for lenght in range(1, path_lenght, 4):

				current_path_lenght = current_path.head(lenght)

				percentage = int(100*lenght/path_lenght)

				#create_training_path(current_path_lenght)

				for model in models:
					compare_line(model["model"], current_path_lenght, model["method"], line, percentage, corrects, model["name"])

				name_total = "total_" + str(percentage)
				if corrects.get(name_total) == None:
					corrects[name_total] = 1
				else:
					corrects[name_total] += 1
				
			print(count/total *100)
	
	return corrects

def test_dump(df, model):

	# test will be indexed in 3, 7, 11...

	lines = df.line.unique()
	jump = 64

	correct = {}
	total_try = {}

	for line in lines:
		print('line:', line)
		current_df = df[df.line == line]

		possible_index_paths = current_df.index_path.unique()
		total = possible_index_paths[-1] - possible_index_paths[0]
		count = 0
		for index_path in range(possible_index_paths[3], possible_index_paths[-1], jump):
			
			count += (1*jump)
			current_path = current_df[current_df.index_path == index_path]
			
			path_lenght = len(current_path.index)

			for lenght in range(0, path_lenght, 4):

				predicted = model.predict(current_path.head(lenght))

				if predicted == line:
					if correct.get(int(100*lenght/path_lenght)) == None:
						correct[int(100*lenght/path_lenght)] = 1
					else:
						correct[int(100*lenght/path_lenght)] += 1
				else:
					if correct.get(int(100*lenght/path_lenght)) == None:
						correct[int(100*lenght/path_lenght)] = 0	
				
				if total_try.get(int(100*lenght/path_lenght)) == None:
					total_try[int(100*lenght/path_lenght)] = 1
				else:
					total_try[int(100*lenght/path_lenght)] += 1
				

			print(count/total *100, '- predicted:', predicted, '- correct:', correct, '- tries:', total_try)
	
	return (correct, total_try)

def test_dump_all_possibilities(df, model):

	# test will be indexed in 3, 7, 11...

	lines = df.line.unique()
	jump = 64

	correct = {}
	total_try = {}

	errors = {}
	super_errors = {}

	for line in lines:
		print('line:', line)
		current_df = df[df.line == line]

		errors[line] = {}
		super_errors[line] = {}

		possible_index_paths = current_df.index_path.unique()
		total = possible_index_paths[-1] - possible_index_paths[0]
		count = 0
		for index_path in range(possible_index_paths[3], possible_index_paths[-1], jump):
			
			count += (1*jump)
			current_path = current_df[current_df.index_path == index_path]
			
			path_lenght = len(current_path.index)

			predictions = model.predict_all_possibilities(current_path)

			if line in predictions:
				for prediction in predictions:
					if prediction != line:
						if errors[line].get(prediction) == None:
							errors[line][prediction] = [index_path]
						else:
							errors[line][prediction].append(index_path)
			else:
				for prediction in predictions:
					if super_errors[line].get(prediction) == None:
						super_errors[line][prediction] = [index_path]
					else:
						super_errors[line][prediction].append(index_path)
				

			print(count/total *100, '- errors:', errors, '- super_errors:', super_errors)
	
	return (correct, total_try)

def separate_go_back(df):
	file = open('../data/riobus_coordinates.json')
	data = json.load(file)

	final_stops = {}
	for key in data.keys():
		final_stops[key] = [Coordinate(latitude=data[key][0][0], longitude=data[key][0][1]), Coordinate(latitude=data[key][1][0], longitude=data[key][1][1])]
	df.line = df.apply(lambda x: x.line + '_ida' if distance_between(df[df.index_path == x.index_path].iloc[0], final_stops[x.line][0]) < 0.1 else x.line + '_volta', axis=1)
	return df

def create_paths_models(df):

	# Models will be indexed in 0, 64, 128...

	lines = df.line.unique()
	jump = 64

	new_df = DataFrame()

	for line in lines:
		print('line:', line)
		current_df = df[df.line == line]

		new_df = new_df.append(current_df.iloc[0])

		possible_index_paths = current_df.index_path.unique()
		total = possible_index_paths[-1] - possible_index_paths[0]
		count = 0
		for index_path in range(possible_index_paths[0], possible_index_paths[-1], jump):
			
			count += (1*jump)
			current_path = current_df[current_df.index_path == index_path]
			print(count/total *100)
			for row in current_path.iterrows():
				row = row[1]
				if not has_distance_from_coordinate(new_df[new_df.line == line], row, distance=minimum_distance):
					new_df = new_df.append(row)
					

	return new_df.drop(['index_path', 'order'], axis=1)
		

class DumpRioBusAlgorithm():
	def __init__(self, paths):
		self.paths = paths
		self.possible_lines = paths.line.unique()

	def fit(self, X, y):
		pass

	def predict(self, path):
		for row in path.iterrows():

			current_not_possible_lines = []

			row = row[1]

			for line in self.possible_lines:
				if not has_distance_from_coordinate(self.paths[self.paths.line == line], row, distance=0.1):
					current_not_possible_lines.append(line)

			if len(current_not_possible_lines) < len(self.possible_lines):
				self.possible_lines = list(set(self.possible_lines) - set(current_not_possible_lines))
			
			if len(self.possible_lines) == 1:
				break
		
		choosed_line = self.possible_lines[0]

		self.possible_lines = self.paths.line.unique()
		return choosed_line

	def predict_all_possibilities(self, path):
		for row in path.iterrows():

			current_not_possible_lines = []

			row = row[1]

			for line in self.possible_lines:
				if not has_distance_from_coordinate(self.paths[self.paths.line == line], row, distance=0.1):
					current_not_possible_lines.append(line)

			if len(current_not_possible_lines) < len(self.possible_lines):
				self.possible_lines = list(set(self.possible_lines) - set(current_not_possible_lines))
			
			if len(self.possible_lines) == 1:
				break
		
		current_possible_lines = self.possible_lines

		self.possible_lines = self.paths.line.unique()
		return current_possible_lines

def plot_dump_results(corrects, tries):
	result = []
	
	for index in range(4, 100, 5):
		corrects_sum = corrects[index] + corrects[index - 1] + corrects[index - 2] + corrects[index - 3] + corrects[index - 4]
		total_sum = tries[index] + tries[index - 1] + tries[index - 2] + tries[index - 3] + tries[index - 4]
		result.append(100*corrects_sum/total_sum)

	result.append(88.6) # This result was computed with completed paths
	plt.plot(list(range(0, 101, 5)), result, label=u'Georreferência Automático', marker='', markerfacecolor='blue')
	
	plt.xlabel('Porcentagem de Completude')
	plt.ylabel(u'Acurácia')
	plt.title(u'Porcentagem de Completude X Acurácia do Trajeto')
	plt.legend(loc='best')
	plt.grid(True)

	plt.show()

def plot_all_results(corrects_dump, tries_dump, corrects):
	result_dump = []
	result_rf = []
	result_svm = []
	result_bayes = []
	
	for index in range(4, 100, 5):
		corrects_dump_sum = corrects_dump[index] + corrects_dump[index - 1] + corrects_dump[index - 2] + corrects_dump[index - 3] + corrects_dump[index - 4]
		corrects_rf_sum = corrects['rf_' + str(index)] + corrects['rf_' + str(index - 1)] + corrects['rf_' + str(index - 2)] + corrects['rf_' + str(index - 3)] + corrects['rf_' + str(index - 4)]
		corrects_bayes_sum = corrects['bayes_' + str(index)] + corrects['bayes_' + str(index - 1)] + corrects['bayes_' + str(index - 2)] + corrects['bayes_' + str(index - 3)] + corrects['bayes_' + str(index - 4)]
		corrects_svm_sum = corrects['svm_' + str(index)] + corrects['svm_' + str(index - 1)] + corrects['svm_' + str(index - 2)] + corrects['svm_' + str(index - 3)] + corrects['svm_' + str(index - 4)]
		
		total_dump = tries_dump[index] + tries_dump[index - 1] + tries_dump[index - 2] + tries_dump[index - 3] + tries_dump[index - 4]
		total_sum = corrects['total_' + str(index)] + corrects['total_' + str(index - 1)] + corrects['total_' + str(index - 2)] + corrects['total_' + str(index - 3)] + corrects['total_' + str(index - 4)]
		result_dump.append(100*corrects_dump_sum/total_dump)
		result_rf.append(100*corrects_rf_sum/total_sum)
		result_svm.append(100*corrects_svm_sum/total_sum)
		result_bayes.append(100*corrects_bayes_sum/total_sum)

	#result_dump.append(88.6) # This result was computed with completed paths

	plt.plot(list(range(0, 100, 5)), result_dump, label=u'Georreferência Automático', marker='', markerfacecolor='blue')
	plt.plot(list(range(0, 100, 5)), result_rf, label=u'Floresta Aleatoria', marker='', markerfacecolor='green')
	plt.plot(list(range(0, 100, 5)), result_svm, label=u'SVM', marker='', markerfacecolor='brown')
	plt.plot(list(range(0, 100, 5)), result_bayes, label=u'Bayes', marker='', markerfacecolor='orange')
	
	plt.xlabel('Porcentagem de Completude')
	plt.ylabel(u'Acurácia')
	plt.title(u'Porcentagem de Completude X Acurácia do Trajeto')
	plt.legend(loc='best')
	plt.grid(True)

	plt.show()

def create_errors_table(errors):
	df = pd.DataFrame(columns=['keys'] + list(errors.keys()))
	for key in errors.keys():
		row = errors[key]
		
		for inside_keys in row.keys():
			row[inside_keys] = [len(row[inside_keys])]
		row['keys'] = [key]

		row = pd.DataFrame(row)
		df = df.append(row)
	df = df.fillna(0)
	return df

def testErrors(df, errors, box_coordinates):
	humans = 0
	others = 0
	for line in errors.keys():
		print('line:', line, '- humans:', humans, '- others:', others)
		for inline in errors[line].keys():
			indexes = errors[line][inline]
			print('--inline:', inline)
			for index in indexes:
				current_df = df[df.index_path == index]
				print('-- -- foi configurado', line, 'mas foi predito', inline)
				plot_path_with_map(current_df, '../data/mapa_rio.png', box_coordinates)
				if input('tipo de erro:') == 'h':
					humans += 1
				else:
					others += 1
	print('humans:', humans, '- others:', others)
	
def testTime(df):
	size = len(df.index) - 1
	counter = 0
	histogram_dict = {}
	for index in range(len(df.index) - 1):
		if int(100*index/size) > counter:
			counter += 1
			print(counter)
		if df.index_path.iloc[index+1] == df.index_path.iloc[index]:
			time = int((df.datetime.iloc[index+1] - df.datetime.iloc[index]).seconds/60)

			if histogram_dict.get(time) == None:
				histogram_dict[time] = 1
			else:
				histogram_dict[time] += 1
	return histogram_dict

def testTimeForTravel(df):
	size = df.index_path.max()
	counter = 0
	histogram_dict = {}
	for index in df.index_path.unique():
		if int(100*index/size) > counter:
			counter += 1
			print(counter)
		
		current_df = df[df.index_path == index]
		time = int((current_df.datetime.iloc[-1] - current_df.datetime.iloc[0]).seconds/60)

		if histogram_dict.get(time) == None:
			histogram_dict[time] = 1
		else:
			histogram_dict[time] += 1
	return histogram_dict

def create_model(method, name, model):
	return {'method': method, 'name': name, 'model': model}

small_riobus = '../data/rio_paths_10.csv'
minimum_percent = 0.1
minimum_distance = 0.1

#models_df = pd.read_csv('../data/dump_models.csv', parse_dates=['datetime'], dtype={'line': object})
#df = pd.read_csv(small_riobus, parse_dates=['datetime'])
#df.datetime = ((df.datetime - df.datetime.dt.normalize()) / pd.Timedelta('1 second')).astype(int)
#le = preprocessing.LabelEncoder()
#le.fit(df.line)
#df['line'] = le.transform(df.line)

#model = DumpRioBusAlgorithm(models_df)

#test_dump_all_possibilities(df, model)

#model_rf = create_model(lambda x: [create_training_path(x)], 'rf', load('../data/rio_best_RF_60.joblib'))
#model_bayes = create_model(lambda x: [create_training_path(x)], 'bayes', load('../data/rio_best_bayes_60.joblib'))
#model_svm = create_model(lambda x: [create_training_path(x)], 'svm', load('../data/rio_best_svm_60.joblib'))

#models = [model_rf, model_bayes, model_svm]
#print(test_all(df, models))

min_lon = -43.4168
max_lon = -42.9173
min_lat = -23.0440
max_lat = -22.7799

box_coordinates = (min_lon, max_lon, min_lat, max_lat)

plot_all_results(corrects_dump, tries_dump, corrects)

#print(testTimeForTravel(df))

#plt.bar(times_for_each_travel.keys(), times_for_each_travel.values())


#testErrors(df, super_errors, box_coordinates)

#errors_table = create_errors_table(errors)
#super_errors_table = create_errors_table(super_errors)

#errors_table.to_csv('../data/rio_errors_table.csv', index=False)
#super_errors_table.to_csv('../data/rio_super_errors_table.csv', index=False)

